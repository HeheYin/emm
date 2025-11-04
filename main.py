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


class StableEpsilonGreedyStrategy:
    """稳定的ε-greedy策略"""

    def __init__(self, start=EPS_START, end=EPS_END, decay=EPS_DECAY):
        self.start = start
        self.end = end
        self.decay = decay
        self.current = start
        self.steps = 0

    def get_epsilon(self):
        """线性衰减而不是指数衰减，更稳定"""
        if self.steps >= self.decay:
            self.current = self.end
        else:
            self.current = self.start - (self.start - self.end) * (self.steps / self.decay)
        self.steps += 1
        return self.current

    def reset(self):
        self.current = self.start
        self.steps = 0


def prepare_dag_input(dag):
    """准备DAG的输入特征"""
    task_ids = sorted(dag.task_nodes.keys())
    task_num = len(task_ids)

    # 构建任务特征矩阵
    task_feat = np.zeros((task_num, HARDWARE_NUM * 2 + 2))

    for i, task_id in enumerate(task_ids):
        task = dag.task_nodes[task_id]

        # 计算开销特征
        for j in range(HARDWARE_NUM):
            task_feat[i, j] = dag.comp_matrix[i, j] / 100.0  # 归一化

        # 能耗特征
        for j in range(HARDWARE_NUM):
            task_feat[i, j + HARDWARE_NUM] = dag.energy_matrix[i, j] / 100.0  # 归一化

        # 优先级和截止时间特征
        task_feat[i, -2] = task.priority / 10.0  # 归一化到[0,1]
        task_feat[i, -1] = task.deadline / 1000.0  # 归一化

    task_feat = torch.tensor(task_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    adj = torch.tensor(nx.to_numpy_array(dag.graph), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 任务优先级顺序
    priorities = [dag.task_nodes[tid].priority for tid in task_ids]
    sorted_indices = sorted(range(len(priorities)), key=lambda i: priorities[i], reverse=True)
    seq_order = torch.tensor(sorted_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 硬件特征
    hardware_feat = []
    for hw in HARDWARE_TYPES:
        hw_info = EMBEDDED_HARDWARES[hw]
        hardware_feat.append([
            hw_info["算力"] / 100.0,  # 归一化
            hw_info["内存"] / 10.0,  # 归一化
            hw_info["能耗系数"],
            hw_info["负载阈值"]
        ])
    hardware_feat = torch.tensor(hardware_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 负载状态和历史资源数据
    load_states = torch.tensor(np.zeros(HARDWARE_NUM), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    history_resource = torch.randn(1, 10, 4, dtype=torch.float32).to(DEVICE) * 0.1

    return task_feat, adj, seq_order, hardware_feat, load_states, history_resource


def main():
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # 设备初始化
    print(f"使用设备: {DEVICE}")

    # 1. 数据集生成
    print("===== 生成嵌入式任务数据集 =====")
    dataset = EmbeddedDatasetGenerator.generate_dataset()

    # 验证DAG质量
    print("===== 验证DAG质量 =====")
    is_valid = validate_dag_quality(dataset)
    if not is_valid:
        print("警告：存在循环依赖的DAG，建议重新生成数据集")
        # 可以选择重新生成或继续
        user_input = input("是否重新生成数据集? (y/n): ")
        if user_input.lower() == 'y':
            dataset = EmbeddedDatasetGenerator.generate_dataset()
            validate_dag_quality(dataset)

    # 3. 初始化MODRL调度器
    print("===== 初始化MODRL调度器 =====")
    task_feat_dim = HARDWARE_NUM + HARDWARE_NUM + 2  # 计算开销 + 能耗 + 优先级 + 截止时间
    adj_dim = MAX_TASK_NUM  # 邻接矩阵维度
    hardware_feat_dim = 4  # 硬件特征维度（算力、内存、能耗系数、负载阈值）
    modrl_scheduler = MODRLScheduler(task_feat_dim, adj_dim, hardware_feat_dim).to(DEVICE)

    # 优化器 - 使用权重衰减
    optimizer = torch.optim.AdamW(
        modrl_scheduler.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    # 4. 动态任务缓冲区初始化
    dynamic_buffer = DynamicTaskBuffer(50)

    # 5. ε-greedy策略初始化
    epsilon_strategy = StableEpsilonGreedyStrategy()

    # 5. 训练过程
    print("===== 开始训练MODRL调度器 =====")
    total_episodes = 100
    replay_buffer = []
    metrics = {
        "makespan": [],
        "load_balance": [],
        "energy_consumption": [],
        "deadline_satisfaction": [],
        "epsilon": [],
        "loss": [],
        "reward": []
    }
    # 训练进度条
    pbar = tqdm(range(total_episodes), desc="训练进度")
    for episode in pbar:
        # 获取当前ε值
        epsilon = epsilon_strategy.get_epsilon()
        metrics["epsilon"].append(epsilon)

        # 重置环境状态
        current_load = np.random.uniform(0.1, 0.3, size=HARDWARE_NUM)  # 初始负载
        episode_reward = 0.0
        episode_makespan = 0.0
        deadline_satisfied = 0
        total_tasks = 0

        # 每次只训练一个DAG，更稳定
        task_type = random.choice(TASK_TYPES)
        dag = random.choice(dataset["split"][task_type])

        # 提取任务特征
        task_ids = sorted(dag.task_nodes.keys())
        task_num = len(task_ids)

        # 构建任务特征矩阵
        task_feat = np.zeros((task_num, HARDWARE_NUM * 2 + 2))

        for i, task_id in enumerate(task_ids):
            task = dag.task_nodes[task_id]

            # 计算开销特征
            for j in range(HARDWARE_NUM):
                task_feat[i, j] = dag.comp_matrix[i, j] / 100.0  # 归一化

            # 能耗特征
            for j in range(HARDWARE_NUM):
                task_feat[i, j + HARDWARE_NUM] = dag.energy_matrix[i, j] / 100.0  # 归一化

            # 优先级和截止时间特征
            task_feat[i, -2] = task.priority / 10.0  # 归一化到[0,1]
            task_feat[i, -1] = task.deadline / 1000.0  # 归一化

        task_feat = torch.tensor(task_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        adj = torch.tensor(nx.to_numpy_array(dag.graph), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 任务优先级顺序
        priorities = [dag.task_nodes[tid].priority for tid in task_ids]
        sorted_indices = sorted(range(len(priorities)), key=lambda i: priorities[i], reverse=True)
        seq_order = torch.tensor(sorted_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

        # 硬件特征
        hardware_feat = []
        for hw in HARDWARE_TYPES:
            hw_info = EMBEDDED_HARDWARES[hw]
            hardware_feat.append([
                hw_info["算力"] / 100.0,  # 归一化
                hw_info["内存"] / 10.0,  # 归一化
                hw_info["能耗系数"],
                hw_info["负载阈值"]
            ])
        hardware_feat = torch.tensor(hardware_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        load_states = torch.tensor(current_load, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 历史资源数据
        history_resource = torch.randn(1, 10, 4, dtype=torch.float32).to(DEVICE) * 0.1  # 减小噪声

        # MODRL调度决策
        q_values = modrl_scheduler(task_feat, adj, seq_order, hardware_feat, load_states, history_resource)

        # 对每个任务应用ε-greedy策略
        actions = []
        current_time = 0.0  # 简化处理
        for task_idx in range(len(task_ids)):
            task = dag.task_nodes[task_ids[task_idx]]
            action = improved_select_action(
                q_values[0, task_idx],
                epsilon,
                dag.comp_matrix[task_idx],
                task.deadline,
                current_time
            )
            actions.append(action)

        actions = np.array(actions)

        # 执行调度
        current_makespan, task_energy, task_finish_times = improved_execute_scheduling(dag, actions, current_load)
        episode_makespan = current_makespan

        # 计算负载状态
        load_states_np = current_load.copy()
        hardware_thresholds = [EMBEDDED_HARDWARES[hw]["负载阈值"] for hw in HARDWARE_TYPES]

        # 计算截止时间满足率
        deadline_met_count = 0
        for task_id, finish_time in task_finish_times.items():
            task = dag.task_nodes[task_id]
            if finish_time <= task.deadline:
                deadline_met_count += 1
        deadline_satisfaction_rate = deadline_met_count / len(task_ids) if task_ids else 0

        # 计算奖励 - 简化版本，避免接口不匹配
        reward = calculate_simple_reward(
            current_makespan,
            load_states_np,
            hardware_thresholds,
            task_energy,
            dag.task_nodes[0].priority,
            deadline_satisfaction_rate
        )

        episode_reward = reward

        # 检查截止时间
        if current_makespan <= dag.task_nodes[0].deadline:
            deadline_satisfied += 1
        total_tasks += 1

        # 经验回放存储
        replay_buffer.append((task_feat, adj, seq_order, hardware_feat, load_states, history_resource, actions, reward))

        if len(replay_buffer) > REPLAY_BUFFER_SIZE:
            replay_buffer.pop(0)

        # 训练更新 - 每步都训练，但使用小学习率
        episode_loss = 0.0
        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            episode_loss = improved_train_stable_batch(batch, modrl_scheduler, optimizer)
            modrl_scheduler.soft_update_target()

        # 记录指标
        metrics["makespan"].append(episode_makespan)
        metrics["load_balance"].append(np.var(current_load))
        metrics["energy_consumption"].append(task_energy)
        metrics["deadline_satisfaction"].append(deadline_satisfaction_rate)
        metrics["loss"].append(episode_loss)
        metrics["reward"].append(episode_reward)

        # 更新进度条
        pbar.set_postfix({
            'ε': f'{epsilon:.3f}',
            'Loss': f'{episode_loss:.4f}' if not np.isnan(episode_loss) else 'nan',
            'Makespan': f'{episode_makespan:.2f}',
            'Reward': f'{episode_reward:.4f}'
        })

        # 每20个episode保存一次模型
        if episode % 20 == 0 and episode > 0:
            torch.save(modrl_scheduler.state_dict(), f'modrl_checkpoint_ep{episode}.pth')

        # 6. 保存最终模型
    torch.save(modrl_scheduler.state_dict(), 'modrl_final.pth')

    # 7. 与基线算法对比
    print("===== 运行基线算法对比实验 =====")
    baseline_results = {}
    for algo in BASELINE_ALGORITHMS:
        print(f"评估 {algo}...")
        baseline_metrics = evaluate_baseline(algo, dataset)
        baseline_results[algo] = baseline_metrics
        print(f"{algo}: Makespan={baseline_metrics['makespan']:.2f}")

    # 8. 评估MODRL
    print("评估MODRL调度器...")
    modrl_metrics = improved_evaluate_modrl(modrl_scheduler, dataset)

    # 9. 结果可视化
    visualize_stable_results(metrics, baseline_results, modrl_metrics)
    print("实验完成！")


def calculate_simple_reward(makespan, load_states, hardware_thresholds, energy, priority, deadline_satisfaction):
    """修正的奖励计算，使用动态权重"""
    # 归一化各项指标
    makespan_reward = 1.0 / (1.0 + makespan / 100.0)
    load_reward = 1.0 - np.std(load_states)
    energy_reward = 1.0 / (1.0 + energy / 500.0)

    # 可靠性惩罚
    reliability_penalty = 0.0
    for load, threshold in zip(load_states, hardware_thresholds):
        if load > threshold:
            reliability_penalty += (load - threshold) / threshold

    reliability_reward = 1.0 - min(reliability_penalty / len(load_states), 1.0)

    # 修复：使用config中的动态权重，修正键名
    weights = WEIGHTS[CURRENT_MODE]

    # 综合奖励 - 修正键名和权重分配
    total_reward = (
        makespan_reward * weights["makespan"] +
        deadline_satisfaction * weights["reliability"] +  # 截止时间满足率使用reliability权重
        load_reward * weights["load"] +
        energy_reward * weights["energy"] +
        reliability_reward * weights["reliability"]  # 可靠性奖励也使用reliability权重
    ) * (0.5 + priority / 20.0)

    return float(total_reward)

def improved_select_action(q_values, epsilon, comp_costs, task_deadline, current_time):
    """改进的动作选择，考虑截止时间"""
    valid_actions = [i for i, cost in enumerate(comp_costs) if cost != float('inf')]
    if not valid_actions:
        return random.randint(0, HARDWARE_NUM - 1)

    # 探索阶段
    if random.random() < epsilon:
        return random.choice(valid_actions)

    # 利用阶段 - 改进的策略
    q_values_np = q_values.detach().cpu().numpy()

    # 创建有效动作的Q值
    masked_q_values = np.full(HARDWARE_NUM, -1e6)
    for i in valid_actions:
        masked_q_values[i] = q_values_np[i]

    # 对于紧急任务，优先选择执行时间短的硬件
    time_remaining = task_deadline - current_time
    is_urgent = time_remaining < np.mean(comp_costs[valid_actions]) * 2

    if is_urgent:
        # 紧急任务：选择执行时间最短的硬件
        best_time = float('inf')
        best_action = valid_actions[0]
        for action in valid_actions:
            if comp_costs[action] < best_time:
                best_time = comp_costs[action]
                best_action = action
        return best_action
    else:
        # 非紧急任务：选择Q值最大的动作
        return np.argmax(masked_q_values)


def improved_execute_scheduling(dag, actions, current_load):
    """改进的调度执行，增强循环依赖处理"""
    task_ids = sorted(dag.task_nodes.keys())
    makespan = 0.0
    total_energy = 0.0
    hw_finish_times = np.zeros(HARDWARE_NUM)
    task_finish_times = {}

    # 创建图的副本
    graph_copy = dag.graph.copy()

    # 增强的拓扑排序处理
    try:
        topological_order = list(nx.topological_sort(graph_copy))
    except nx.NetworkXUnfeasible:
        print(f"严重警告：DAG {dag.dag_id} 存在无法解决的循环依赖")
        # 使用节点ID顺序作为备选
        topological_order = task_ids
        # 记录问题以便后续分析
        with open("cyclic_dags.log", "a") as f:
            f.write(f"DAG {dag.dag_id} 存在循环依赖，使用ID顺序\n")

    # 初始化完成时间
    for task_id in task_ids:
        task_finish_times[task_id] = 0.0

    # 按照拓扑顺序执行任务
    for task_id in topological_order:
        task_idx = task_ids.index(task_id)
        hw_idx = actions[task_idx] if task_idx < len(actions) else 0

        # 硬件支持检查
        if dag.comp_matrix[task_idx, hw_idx] == float('inf'):
            valid_hw = [j for j in range(HARDWARE_NUM) if dag.comp_matrix[task_idx, j] != float('inf')]
            hw_idx = valid_hw[0] if valid_hw else 0

        exec_time = dag.comp_matrix[task_idx, hw_idx]
        energy = dag.energy_matrix[task_idx, hw_idx]

        # 计算开始时间
        start_time = hw_finish_times[hw_idx]

        # 检查前驱任务完成时间
        pred_finish_times = []
        for pred_id in list(graph_copy.predecessors(task_id)):  # 转换为list避免在迭代中修改
            if pred_id in task_finish_times:
                pred_finish = task_finish_times[pred_id]
                pred_idx = task_ids.index(pred_id)
                comm_delay = np.mean(dag.comm_matrix[pred_idx, task_idx]) if dag.comm_matrix[
                    pred_idx, task_idx].any() else 0
                pred_finish_times.append(pred_finish + comm_delay)

        if pred_finish_times:
            start_time = max(start_time, max(pred_finish_times))

        finish_time = start_time + exec_time
        hw_finish_times[hw_idx] = finish_time
        task_finish_times[task_id] = finish_time

        # 更新负载
        current_load[hw_idx] = min(current_load[hw_idx] + 0.1, 1.0)
        makespan = max(makespan, finish_time)
        total_energy += energy

    return makespan, total_energy, task_finish_times


def validate_dag_quality(dataset):
    """验证生成的DAG质量（增强版本）"""
    cyclic_count = 0
    total_dags = 0
    problematic_dags = []

    for task_type in TASK_TYPES:
        for dag in dataset["split"][task_type]:
            total_dags += 1
            try:
                # 检查是否有环
                list(nx.topological_sort(dag.graph))

                # 额外检查：确保图是连通的（至少有一个入口节点）
                entry_nodes = [node for node in dag.graph.nodes() if dag.graph.in_degree(node) == 0]
                if not entry_nodes:
                    print(f"警告：DAG {dag.dag_id} 没有入口节点")
                    problematic_dags.append(dag.dag_id)

            except nx.NetworkXUnfeasible:
                cyclic_count += 1
                problematic_dags.append(dag.dag_id)
                print(f"循环依赖DAG: {dag.dag_id}, 类型: {task_type}")

    print(f"\n=== DAG质量详细报告 ===")
    print(f"总DAG数: {total_dags}")
    print(f"有循环依赖的DAG数: {cyclic_count}")
    print(f"无环DAG比例: {(total_dags - cyclic_count) / total_dags * 100:.2f}%")

    if problematic_dags:
        print(f"有问题的DAG ID: {problematic_dags[:10]}")  # 只显示前10个

    return cyclic_count == 0
def improved_train_stable_batch(batch, scheduler, optimizer):
    """改进的批次训练，增加稳定性检查"""
    if len(batch) == 0:
        return 0.0

    optimizer.zero_grad()
    total_loss = 0.0
    valid_samples = 0

    for sample in batch:
        task_feat, adj, seq_order, hardware_feat, load_states, history_resource, actions, reward = sample

        # 更严格的样本验证
        if (np.isnan(reward) or np.isinf(reward) or
                abs(reward) > 10.0 or  # 奖励值异常大
                task_feat.isnan().any() or task_feat.isinf().any()):
            continue

        device = task_feat.device
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)

        try:
            # 计算当前Q值
            current_state_features = scheduler.prepare_state_features(
                task_feat, adj, seq_order, hardware_feat, load_states, history_resource
            )
            current_q_values = scheduler.q_network(current_state_features)

            # 目标Q值计算
            with torch.no_grad():
                next_q_values_target = scheduler.target_q_network(current_state_features)
                next_q_max = torch.max(next_q_values_target, dim=-1)[0]
                target_q = reward_tensor + GAMMA * next_q_max

            # 动作处理
            batch_size, task_num, _ = current_q_values.shape
            if isinstance(actions, (int, np.int64)):
                actions = [actions]
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)

            if actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(0)
            if actions_tensor.dim() == 2:
                actions_tensor = actions_tensor.unsqueeze(-1)

            # 确保维度匹配
            if actions_tensor.shape[1] != task_num:
                # 调整动作序列长度
                if actions_tensor.shape[1] < task_num:
                    # 填充
                    padding = torch.zeros(batch_size, task_num - actions_tensor.shape[1], 1,
                                          dtype=torch.long, device=device)
                    actions_tensor = torch.cat([actions_tensor, padding], dim=1)
                else:
                    # 截断
                    actions_tensor = actions_tensor[:, :task_num, :]

            current_q = current_q_values.gather(-1, actions_tensor).squeeze(-1)

            # 计算损失
            loss = F.smooth_l1_loss(current_q, target_q)

            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 100.0:
                total_loss += loss
                valid_samples += 1

        except Exception as e:
            print(f"训练批次出错: {e}")
            continue

    if valid_samples > 0:
        avg_loss = total_loss / valid_samples
        avg_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(scheduler.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        return avg_loss.item()
    else:
        return 0.0


def improved_evaluate_modrl(scheduler, dataset):
    """改进的MODRL评估"""
    scheduler.eval()
    makespans = []
    energies = []
    deadline_rates = []

    with torch.no_grad():
        for task_type in TASK_TYPES:
            for dag in dataset["split"][task_type][:20]:  # 评估更多样本
                # 使用改进的调度执行
                task_ids = sorted(dag.task_nodes.keys())
                current_load = np.zeros(HARDWARE_NUM)

                # 构建输入特征
                task_feat, adj, seq_order, hardware_feat, load_states, history_resource = \
                    prepare_dag_input(dag)

                # 获取Q值并选择动作
                q_values = scheduler(task_feat, adj, seq_order, hardware_feat, load_states, history_resource)
                actions = torch.argmax(q_values, dim=-1).cpu().numpy()[0]

                # 执行调度
                makespan, energy, task_finish_times = improved_execute_scheduling(dag, actions, current_load)

                # 计算截止时间满足率
                deadline_met_count = 0
                for task_id, finish_time in task_finish_times.items():
                    task = dag.task_nodes[task_id]
                    if finish_time <= task.deadline:
                        deadline_met_count += 1

                deadline_rate = deadline_met_count / len(task_ids) if task_ids else 0

                makespans.append(makespan)
                energies.append(energy)
                deadline_rates.append(deadline_rate)

    scheduler.train()
    return {
        "makespan": np.mean(makespans),
        "energy": np.mean(energies),
        "deadline": np.mean(deadline_rates)
    }


def evaluate_baseline(algorithm, dataset):
    """评估基线算法"""
    metrics = {
        "makespan": [],
        "load_balance": [],
        "energy": [],
        "deadline": []
    }

    for task_type in TASK_TYPES:
        for dag in dataset["split"][task_type][:10]:  # 只评估前10个
            if algorithm == "HEFT":
                makespan, load_balance = BaselineScheduler.heft_schedule(dag)
            elif algorithm == "RM":
                makespan, load_balance = BaselineScheduler.rm_schedule(dag)
            elif algorithm == "EDF":
                makespan, load_balance = BaselineScheduler.edf_schedule(dag)
            else:
                makespan, load_balance = 0.0, 0.0

            energy = np.sum(dag.energy_matrix)
            deadline_met = 1 if makespan <= dag.task_nodes[0].deadline else 0

            metrics["makespan"].append(makespan)
            metrics["load_balance"].append(load_balance)
            metrics["energy"].append(energy)
            metrics["deadline"].append(deadline_met)

    return {
        "makespan": np.mean(metrics["makespan"]),
        "load_balance": np.mean(metrics["load_balance"]),
        "energy": np.mean(metrics["energy"]),
        "deadline": np.mean(metrics["deadline"])
    }


def visualize_stable_results(train_metrics, baseline_results, modrl_results):
    """改进的可视化"""
    plt.figure(figsize=(20, 12))

    # 1. 训练过程指标
    plt.subplot(2, 4, 1)
    plt.plot(train_metrics["makespan"])
    plt.title("Training Makespan")
    plt.xlabel("Episode")
    plt.ylabel("Makespan")

    plt.subplot(2, 4, 2)
    plt.plot(train_metrics["loss"])
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.subplot(2, 4, 3)
    plt.plot(train_metrics["reward"])
    plt.title("Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(2, 4, 4)
    plt.plot(train_metrics["epsilon"])
    plt.title("Exploration Rate ε")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    # 2. 算法对比
    plt.subplot(2, 4, 5)
    algorithms = list(baseline_results.keys()) + ["MODRL"]
    makespans = [baseline_results[algo]["makespan"] for algo in baseline_results] + [modrl_results["makespan"]]
    plt.bar(algorithms, makespans, color=['gray'] * len(baseline_results) + ['red'])
    plt.title("Makespan Comparison")
    plt.xticks(rotation=45)

    plt.subplot(2, 4, 6)
    energies = [baseline_results[algo]["energy"] for algo in baseline_results] + [modrl_results["energy"]]
    plt.bar(algorithms, energies, color=['gray'] * len(baseline_results) + ['red'])
    plt.title("Energy Consumption")
    plt.xticks(rotation=45)

    plt.subplot(2, 4, 7)
    deadlines = [baseline_results[algo]["deadline"] for algo in baseline_results] + [modrl_results["deadline"]]
    plt.bar(algorithms, deadlines, color=['gray'] * len(baseline_results) + ['red'])
    plt.title("Deadline Satisfaction Rate")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("stable_experiment_results.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 输出详细结果
    print("\n=== Detailed Results Comparison ===")
    for algo in baseline_results:
        print(f"{algo}: Makespan={baseline_results[algo]['makespan']:.2f}, "
              f"Energy={baseline_results[algo]['energy']:.2f}, "
              f"Deadline={baseline_results[algo]['deadline']:.2%}")

    print(f"MODRL: Makespan={modrl_results['makespan']:.2f}, "
          f"Energy={modrl_results['energy']:.2f}, "
          f"Deadline={modrl_results['deadline']:.2%}")


if __name__ == "__main__":
    main()