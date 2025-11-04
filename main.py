import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import torch.nn.functional as F
import random
from config import *
from task_model import EmbeddedDAG, TaskSplitter
from layered_optimization import MODRLScheduler, MultiObjectiveReward, TaskMigrationStrategy
from experiment import EmbeddedDatasetGenerator, BaselineScheduler
from special_scenarios import DynamicTaskBuffer, MultiSoftwareScheduler


def main():
    # 设备初始化
    print(f"使用设备: {DEVICE}")

    # 1. 数据集生成
    print("===== 生成嵌入式任务数据集 =====")
    dataset = EmbeddedDatasetGenerator.generate_dataset()

    # 2. 初始化多软件调度器（多软件并发场景）
    software_priorities = {
        "工业控制软件": 1.0,
        "边缘AI软件": 0.8,
        "传感器融合软件": 0.7
    }
    multi_soft_scheduler = MultiSoftwareScheduler(software_priorities)

    # 注册软件DAG任务
    for task_type in TASK_TYPES:
        software_name = {
            "工业控制": "工业控制软件",
            "边缘AI": "边缘AI软件",
            "传感器融合": "传感器融合软件"
        }[task_type]
        for dag in dataset["split"][task_type]:
            multi_soft_scheduler.register_software_dag(software_name, dag)

    # 3. 初始化MODRL调度器
    print("===== 初始化MODRL调度器 =====")
    task_feat_dim = HARDWARE_NUM + HARDWARE_NUM + 2  # 计算开销 + 能耗 + 优先级 + 截止时间
    adj_dim = MAX_TASK_NUM  # 邻接矩阵维度
    hardware_feat_dim = 4  # 硬件特征维度（算力、内存、能耗系数、负载阈值）
    modrl_scheduler = MODRLScheduler(task_feat_dim, adj_dim, hardware_feat_dim).to(DEVICE)

    # 优化器
    optimizer = torch.optim.Adam(modrl_scheduler.parameters(), lr=LEARNING_RATE)

    # 4. 动态任务缓冲区初始化
    dynamic_buffer = DynamicTaskBuffer(100)
    for task in dataset["dynamic"]:
        dynamic_buffer.add_task(task, time.time())

        # 5. 训练过程
    print("===== 开始训练MODRL调度器 =====")
    total_episodes = 100
    replay_buffer = []
    metrics = {
        "makespan": [],
        "load_balance": [],
        "energy_consumption": [],
        "deadline_satisfaction": []
    }

    for episode in tqdm(range(total_episodes), desc="训练进度"):
        # 重置环境状态
        current_load = np.random.uniform(0.2, 0.5, size=HARDWARE_NUM)  # 初始负载
        last_makespan = 0.0
        last_energy = 0.0
        total_energy = 0.0
        deadline_satisfied = 0
        total_tasks = 0

        # 添加episode级别的统计信息
        episode_makespan_history = []  # 记录该episode中每个DAG的makespan
        episode_energy_history = []  # 记录该episode中每个DAG的能耗

        # 随机选择一个DAG批次
        task_type = random.choice(TASK_TYPES)
        dag_batch = random.sample(dataset["split"][task_type], 5)  # 每次训练5个DAG

        for dag in dag_batch:
            # 提取任务特征 - 修改部分：构建更丰富的任务特征
            task_ids = sorted(dag.task_nodes.keys())
            task_num = len(task_ids)

            # 构建任务特征矩阵 (task_num, feature_dim)
            # 特征包括：计算开销(hardware_num) + 能耗(hardware_num) + 优先级(1) + 截止时间(1) = hardware_num*2 + 2
            task_feat_dim = HARDWARE_NUM + HARDWARE_NUM + 2  # 与 ExperimentEvaluator.evaluate_modrl 保持一致
            task_feat = np.zeros((task_num, task_feat_dim))

            for i, task_id in enumerate(task_ids):
                task = dag.task_nodes[task_id]
                task_idx = i

                # 计算开销特征
                for j, hw in enumerate(HARDWARE_TYPES):
                    task_feat[task_idx, j] = dag.comp_matrix[task_idx, j]

                # 能耗特征
                for j, hw in enumerate(HARDWARE_TYPES):
                    task_feat[task_idx, j + HARDWARE_NUM] = dag.energy_matrix[task_idx, j]

                # 优先级特征
                task_feat[task_idx, -2] = task.priority

                # 截止时间特征
                task_feat[task_idx, -1] = task.deadline

            task_feat = torch.tensor(task_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)


            adj = torch.tensor(nx.to_numpy_array(dag.graph), dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # 替换原来的seq_order生成代码
            priorities = [dag.task_nodes[tid].priority for tid in task_ids]
            # 创建按照优先级排序的索引
            sorted_indices = sorted(range(len(priorities)), key=lambda i: priorities[i], reverse=True)
            seq_order = torch.tensor(sorted_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

            # 提取硬件特征
            hardware_feat = []
            for hw in HARDWARE_TYPES:
                hw_info = EMBEDDED_HARDWARES[hw]
                hardware_feat.append([hw_info["算力"], hw_info["内存"], hw_info["能耗系数"], hw_info["负载阈值"]])
            hardware_feat = torch.tensor(hardware_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            load_states = torch.tensor(current_load, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # 生成随机历史资源数据（模拟过去10个时间步）
            history_resource = torch.randn(1, 10, 4, dtype=torch.float32).to(DEVICE)

            # MODRL调度决策
            q_values = modrl_scheduler(task_feat, adj, seq_order, hardware_feat, load_states, history_resource)
            action = torch.argmax(q_values, dim=-1).cpu().numpy()[0]  # 取出 batch 维度的第一个样本

            # 执行调度并计算奖励
            current_makespan, task_energy = execute_scheduling(dag, action, current_load)
            total_energy += task_energy

            # 计算负载状态
            load_states_np = current_load.copy()
            hardware_thresholds = [EMBEDDED_HARDWARES[hw]["负载阈值"] for hw in HARDWARE_TYPES]

            # 使用episode历史计算更稳定的奖励
            if len(episode_makespan_history) > 0:
                last_makespan_avg = np.mean(episode_makespan_history[-5:])  # 使用最近5个DAG的平均值
            else:
                last_makespan_avg = last_makespan  # 使用episode初始值

            if len(episode_energy_history) > 0:
                last_energy_avg = np.mean(episode_energy_history[-5:])  # 使用最近5个DAG的平均值
            else:
                last_energy_avg = last_energy  # 使用episode初始值

            # 计算多目标奖励
            reward = MultiObjectiveReward.calculate_reward(
                current_makespan, last_makespan_avg,
                load_states_np, hardware_thresholds,
                total_energy, last_energy_avg,
                dag.task_nodes[0].priority,  # 任务优先级
                dag.task_type  # 添加任务类型参数
            )

            # 更新episode历史
            episode_makespan_history.append(current_makespan)
            episode_energy_history.append(total_energy)

            # 检查截止时间满足情况
            if current_makespan <= dag.task_nodes[0].deadline:
                deadline_satisfied += 1
            total_tasks += 1

            # 更新状态
            last_makespan = current_makespan
            last_energy = total_energy

            # 经验回放存储
            replay_buffer.append((task_feat, adj, seq_order, hardware_feat, load_states, q_values, reward))
            if len(replay_buffer) > REPLAY_BUFFER_SIZE:
                replay_buffer.pop(0)

            # 资源状态预测与任务迁移（硬件层管控）
            if np.any(current_load > 0.85):  # 负载过高触发迁移
                overloaded_hw = np.argmax(current_load)
                task_data_size = np.random.uniform(1, 10)  # 随机任务数据量(MB)
                target_hw, migration_cost = TaskMigrationStrategy.select_target_hardware(
                    overloaded_hw, task_data_size, current_load, hardware_thresholds
                )
                # 执行迁移（更新负载）
                current_load[overloaded_hw] -= 0.1
                current_load[target_hw] += 0.1

        # 动态任务处理
        while not dynamic_buffer.is_empty():
            dynamic_task = dynamic_buffer.get_next_task()
            # 简化处理：直接调度动态任务
            total_tasks += 1
            if schedule_dynamic_task(dynamic_task, current_load):
                deadline_satisfied += 1

        # 训练更新
        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            loss = train_batch(batch, modrl_scheduler, optimizer)
            modrl_scheduler.soft_update_target()

        # 记录指标
        metrics["makespan"].append(last_makespan)
        metrics["load_balance"].append(np.var(current_load))
        metrics["energy_consumption"].append(total_energy)
        metrics["deadline_satisfaction"].append(deadline_satisfied / total_tasks if total_tasks > 0 else 0)

    # 6. 与基线算法对比实验
    print("===== 运行基线算法对比实验 =====")
    baseline_results = {}
    for algo in BASELINE_ALGORITHMS:
        baseline_metrics = evaluate_baseline(algo, dataset)
        baseline_results[algo] = baseline_metrics

    # 7. 结果可视化
    visualize_results(metrics, baseline_results)
    print("实验完成！")


# main.py 中的 execute_scheduling 函数需要修改
def execute_scheduling(dag, actions, current_load):
    """执行调度并返回makespan和能耗"""
    task_ids = sorted(dag.task_nodes.keys())
    makespan = 0.0
    total_energy = 0.0
    hw_finish_times = np.zeros(HARDWARE_NUM)  # 记录每个硬件的完成时间

    # 修改部分：确保 actions 是一个数组，而不是单个值
    if isinstance(actions, int):
        actions = [actions] * len(task_ids)
    elif isinstance(actions, np.ndarray) and actions.ndim == 0:
        actions = [actions.item()] * len(task_ids)
    elif isinstance(actions, torch.Tensor) and actions.dim() == 0:
        actions = [actions.item()] * len(task_ids)

    for i, task_id in enumerate(task_ids):
        task_idx = i
        hw_idx = actions[i]
        exec_time = dag.comp_matrix[task_idx, hw_idx]
        energy = dag.energy_matrix[task_idx, hw_idx]

        # 更新硬件完成时间（考虑任务依赖）
        start_time = float(hw_finish_times[hw_idx])  # 确保转换为标量
        # 考虑前驱任务的完成时间
        for pred_id in dag.graph.predecessors(task_id):
            pred_idx = task_ids.index(pred_id)
            # 简化处理：假设在同一硬件上执行
            start_time = max(start_time, float(hw_finish_times[hw_idx]))  # 确保类型一致

        finish_time = start_time + exec_time
        hw_finish_times[hw_idx] = finish_time

        # 更新负载
        current_load[hw_idx] = min(current_load[hw_idx] + exec_time / 100, 1.0)
        makespan = max(makespan, finish_time)
        total_energy += energy

    return makespan, total_energy



def schedule_dynamic_task(task, current_load):
    """调度动态任务并返回是否满足截止时间"""
    hw_candidates = [i for i, hw in enumerate(HARDWARE_TYPES) if hw in task.hardware_tags]
    if not hw_candidates:
        return False  # 无可用硬件

    # 选择负载最低的硬件
    hw_idx = min(hw_candidates, key=lambda x: current_load[x])
    exec_time = task.comp_cost[HARDWARE_TYPES[hw_idx]]

    # 更新负载
    current_load[hw_idx] = min(current_load[hw_idx] + exec_time / 100, 1.0)
    return exec_time <= task.deadline


def train_batch(batch, scheduler, optimizer):
    """训练批次数据"""
    optimizer.zero_grad()
    total_loss = 0.0

    for sample in batch:
        task_feat, adj, seq_order, hardware_feat, load_states, _, reward = sample
        device = task_feat.device
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)

        # 重新计算当前Q值（用于训练）
        history_resource = torch.randn(task_feat.shape[0], 10, 4, dtype=task_feat.dtype, device=device)
        current_q_values = scheduler(task_feat, adj, seq_order, hardware_feat, load_states, history_resource)

        # 使用当前Q值计算目标Q值（简化方法）
        with torch.no_grad():
            next_q_values = scheduler(task_feat, adj, seq_order, hardware_feat, load_states, history_resource)
            target_q = reward_tensor + GAMMA * torch.max(next_q_values, dim=-1)[0]

        # 计算损失
        current_q_max = torch.max(current_q_values, dim=-1)[0]
        loss = F.mse_loss(current_q_max, target_q)
        total_loss += loss

    total_loss.backward()
    optimizer.step()
    return total_loss.item() / len(batch)



def evaluate_baseline(algorithm, dataset):
    """评估基线算法性能"""
    metrics = {
        "makespan": [],
        "load_balance": [],
        "energy": [],
        "deadline": []
    }

    for task_type in TASK_TYPES:
        for dag in dataset["split"][task_type]:
            if algorithm == "HEFT":
                makespan, load_balance = BaselineScheduler.heft_schedule(dag)
            elif algorithm == "RM":
                makespan, load_balance = BaselineScheduler.rm_schedule(dag)
            elif algorithm == "EDF":
                makespan, load_balance = BaselineScheduler.edf_schedule(dag)  # 假设已实现
            else:
                makespan, load_balance = 0.0, 0.0  # 其他算法实现略

            # 计算能耗和截止时间满足率
            energy = np.sum(dag.energy_matrix)
            deadline_met = 1 if makespan <= dag.task_nodes[0].deadline else 0

            metrics["makespan"].append(makespan)
            metrics["load_balance"].append(load_balance)
            metrics["energy"].append(energy)
            metrics["deadline"].append(deadline_met)

    # 计算平均值
    return {
        "makespan": np.mean(metrics["makespan"]),
        "load_balance": np.mean(metrics["load_balance"]),
        "energy": np.mean(metrics["energy"]),
        "deadline": np.mean(metrics["deadline"])
    }


def visualize_results(modrl_metrics, baseline_results):
    """可视化实验结果"""
    plt.figure(figsize=(16, 12))

    # 1. Makespan对比
    plt.subplot(2, 2, 1)
    baseline_makespan = [baseline_results[algo]["makespan"] for algo in BASELINE_ALGORITHMS]
    plt.bar(BASELINE_ALGORITHMS + ["MODRL"], baseline_makespan + [np.mean(modrl_metrics["makespan"])])
    plt.title("Makespan Comparison (Lower is Better)")
    plt.xticks(rotation=45)

    # 2. 负载均衡对比
    plt.subplot(2, 2, 2)
    baseline_load = [baseline_results[algo]["load_balance"] for algo in BASELINE_ALGORITHMS]
    plt.bar(BASELINE_ALGORITHMS + ["MODRL"], baseline_load + [np.mean(modrl_metrics["load_balance"])])
    plt.title("Load Balance Comparison (Lower is Better)")
    plt.xticks(rotation=45)

    # 3. 能耗对比
    plt.subplot(2, 2, 3)
    baseline_energy = [baseline_results[algo]["energy"] for algo in BASELINE_ALGORITHMS]
    plt.bar(BASELINE_ALGORITHMS + ["MODRL"], baseline_energy + [np.mean(modrl_metrics["energy_consumption"])])
    plt.title("Energy Consumption Comparison (Lower is Better)")
    plt.xticks(rotation=45)

    # 4. 截止时间满足率
    plt.subplot(2, 2, 4)
    baseline_deadline = [baseline_results[algo]["deadline"] for algo in BASELINE_ALGORITHMS]
    plt.bar(BASELINE_ALGORITHMS + ["MODRL"], baseline_deadline + [np.mean(modrl_metrics["deadline_satisfaction"])])
    plt.title("Deadline Satisfaction Rate (Higher is Better)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("experiment_results.png")
    plt.show()



if __name__ == "__main__":
    main()