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
from experiment import EmbeddedDatasetGenerator, BaselineScheduler, DetailedSchedulerVisualizer
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
    """修正的状态特征准备"""
    task_ids = sorted(dag.task_nodes.keys())
    task_num = len(task_ids)

    # 构建任务特征矩阵
    task_feat = np.zeros((task_num, HARDWARE_NUM * 2 + 2))

    for i, task_id in enumerate(task_ids):
        task = dag.task_nodes[task_id]

        # 计算开销特征
        for j in range(HARDWARE_NUM):
            task_feat[i, j] = dag.comp_matrix[i, j] / 100.0

        # 能耗特征
        for j in range(HARDWARE_NUM):
            task_feat[i, j + HARDWARE_NUM] = dag.energy_matrix[i, j] / 100.0

        # 优先级和截止时间特征
        task_feat[i, -2] = task.priority / 10.0
        task_feat[i, -1] = task.deadline / 1000.0

    task_feat = torch.tensor(task_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    adj = torch.tensor(nx.to_numpy_array(dag.graph), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 任务优先级顺序
    priorities = [dag.task_nodes[tid].priority for tid in task_ids]
    sorted_indices = sorted(range(len(priorities)), key=lambda i: priorities[i], reverse=True)
    seq_order = torch.tensor(sorted_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 修正硬件特征 - 确保正确维度
    hardware_feat = []
    for hw in HARDWARE_TYPES:
        hw_info = EMBEDDED_HARDWARES[hw]
        hardware_feat.append([
            hw_info["算力"] / 100.0,
            hw_info["内存"] / 10.0,
            hw_info["能耗系数"],
            hw_info["负载阈值"]
        ])

    # 修正：硬件特征应该是 (batch_size, hardware_num, feature_dim)
    hardware_feat = torch.tensor(hardware_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 修正：负载状态应该是 (batch_size, hardware_num)
    current_load = np.random.uniform(0.1, 0.3, size=HARDWARE_NUM)
    load_states = torch.tensor(current_load, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 历史资源数据
    history_resource = torch.randn(1, 10, 4, dtype=torch.float32).to(DEVICE) * 0.1

    return task_feat, adj, seq_order, hardware_feat, load_states, history_resource

def reset_hardware_usage():
    """重置硬件使用计数"""
    if hasattr(improved_select_action, 'hw_usage_count'):
        improved_select_action.hw_usage_count = [0] * HARDWARE_NUM

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
        reset_hardware_usage()
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
        # 为每个DAG创建独立的硬件使用计数
        episode_hw_usage_count = [0] * HARDWARE_NUM

        # MODRL调度决策
        q_values = modrl_scheduler(task_feat, adj, seq_order, hardware_feat, load_states, history_resource)

        # 对每个任务应用ε-greedy策略
        actions = []
        current_time = 0.0

        for task_idx in range(len(task_ids)):
            task = dag.task_nodes[task_ids[task_idx]]
            action = improved_select_action(
                q_values[0, task_idx],  # 每个任务对应的Q值
                epsilon,
                dag.comp_matrix[task_idx],  # 该任务在各硬件上的计算成本
                task.deadline,
                current_time,
                task_idx,  # 传递任务索引
                episode_hw_usage_count  # 传递episode特定的使用计数
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
    """简化奖励函数，专注于核心目标"""

    # 1. Makespan奖励（最重要）
    makespan_reward = 1.0 / (1.0 + makespan / 100.0)

    # 2. 负载均衡奖励（促进并行）
    load_balance = 1.0 - np.std(load_states)

    # 3. 截止时间奖励
    deadline_reward = deadline_satisfaction

    # 4. 综合奖励（大幅简化）
    total_reward = (
            makespan_reward * 0.5 +
            load_balance * 0.3 +
            deadline_reward * 0.2
    )

    return float(total_reward)


def improved_select_action(q_values, epsilon, comp_costs, task_deadline, current_time, task_idx=None,
                           hw_usage_count=None):
    """改进的动作选择，强制负载均衡和并行化"""

    # 获取有效硬件（计算成本不为无穷大）
    valid_actions = [i for i, cost in enumerate(comp_costs) if cost != float('inf')]
    if not valid_actions:
        return random.randint(0, HARDWARE_NUM - 1)

    # 如果未提供硬件使用计数，创建新的（每个DAG独立）
    if hw_usage_count is None:
        hw_usage_count = [0] * HARDWARE_NUM

    # 如果是DAG的第一个任务，重置硬件使用计数（确保每个DAG独立）
    if task_idx == 0:
        hw_usage_count = [0] * HARDWARE_NUM

    # 计算当前负载状态（基于使用计数）
    total_usage = sum(hw_usage_count) if sum(hw_usage_count) > 0 else 1
    hw_usage_ratio = [count / total_usage for count in hw_usage_count]

    # 探索阶段 - 强制负载均衡和并行化
    if random.random() < epsilon:
        # 并行化优先策略：选择使用率最低的硬件
        min_usage = float('inf')
        best_hws = []

        for hw in valid_actions:
            # 强烈偏好使用率低的硬件
            usage_score = hw_usage_ratio[hw]

            if usage_score < min_usage:
                min_usage = usage_score
                best_hws = [hw]
            elif usage_score == min_usage:
                best_hws.append(hw)

        # 如果有多个相同使用率的硬件，随机选择
        if best_hws:
            best_hw = random.choice(best_hws)
        else:
            best_hw = random.choice(valid_actions)

        # 更新使用计数
        hw_usage_count[best_hw] += 1
        return best_hw

    # 利用阶段 - 结合Q值、负载均衡和并行化约束
    q_values_np = q_values.detach().cpu().numpy()

    # 创建并行化感知的Q值调整
    masked_q_values = np.full(HARDWARE_NUM, -1e6)

    for hw in valid_actions:
        # 1. 基础Q值
        base_q = q_values_np[hw]

        # 2. 负载均衡因子（强烈惩罚过载硬件）
        load_balance_factor = 1.5 - (hw_usage_ratio[hw] * 2.0)  # 使用率越高，惩罚越大

        # 3. 并行化奖励（鼓励分散到不同硬件）
        parallelization_bonus = 0.0
        if hw_usage_ratio[hw] < 0.3:  # 如果硬件使用率低于30%，给予奖励
            parallelization_bonus = 0.8 * (0.3 - hw_usage_ratio[hw])

        # 4. 紧急任务处理
        time_remaining = task_deadline - current_time
        avg_comp_cost = np.mean([comp_costs[i] for i in valid_actions])
        is_urgent = time_remaining < avg_comp_cost * 2

        if is_urgent:
            # 紧急任务：优先选择执行时间最短的硬件
            time_factor = 1.0 / (1.0 + comp_costs[hw] / 10.0)
            adjusted_q_value = base_q * (1.0 + time_factor * 0.6)  # 时间因素占60%权重
        else:
            # 正常任务：综合考虑Q值、负载均衡和并行化
            adjusted_q_value = base_q * (1.0 + load_balance_factor * 0.4 + parallelization_bonus)

        masked_q_values[hw] = adjusted_q_value

    # 选择最佳动作
    max_value = np.max(masked_q_values)
    best_actions = [i for i in valid_actions if masked_q_values[i] == max_value]

    # 处理多个最佳动作的情况
    if not best_actions:
        # 如果没有有效动作，选择使用率最低的硬件
        min_usage = float('inf')
        best_hw = valid_actions[0]
        for hw in valid_actions:
            if hw_usage_ratio[hw] < min_usage:
                min_usage = hw_usage_ratio[hw]
                best_hw = hw
    elif len(best_actions) > 1:
        # 多个相同Q值，选择使用率最低的（进一步促进并行化）
        min_usage = float('inf')
        best_hw = best_actions[0]
        for hw in best_actions:
            if hw_usage_ratio[hw] < min_usage:
                min_usage = hw_usage_ratio[hw]
                best_hw = hw
    else:
        best_hw = best_actions[0]

    # 更新使用计数
    hw_usage_count[best_hw] += 1

    return best_hw

def improved_execute_scheduling(dag, actions, current_load):
    """改进的并行调度执行，支持多硬件并行执行"""
    task_ids = sorted(dag.task_nodes.keys())
    makespan = 0.0
    total_energy = 0.0

    # 初始化硬件状态 - 每个硬件维护任务时间线
    hw_timelines = {
        i: {
            'tasks': [],  # 已安排的任务列表 (start_time, finish_time, task_id)
            'next_available': 0.0  # 下一个可用时间
        } for i in range(HARDWARE_NUM)
    }

    task_finish_times = {}
    task_start_times = {}
    completed_tasks = set()

    # 获取拓扑顺序
    try:
        topological_order = list(nx.topological_sort(dag.graph))
    except nx.NetworkXUnfeasible:
        print(f"警告：DAG {dag.dag_id} 存在循环依赖，使用ID顺序")
        topological_order = task_ids

    current_time = 0.0
    max_iterations = 10000  # 防止无限循环
    iteration = 0

    while len(completed_tasks) < len(task_ids) and iteration < max_iterations:
        iteration += 1
        scheduled_this_round = False

        # 按拓扑顺序检查每个任务是否可以开始执行
        for task_id in topological_order:
            if task_id in completed_tasks:
                continue

            task_idx = task_ids.index(task_id)
            hw_idx = actions[task_idx] if task_idx < len(actions) else 0

            # 检查硬件支持
            if dag.comp_matrix[task_idx, hw_idx] == float('inf'):
                valid_hw = [j for j in range(HARDWARE_NUM) if dag.comp_matrix[task_idx, j] != float('inf')]
                if valid_hw:
                    hw_idx = valid_hw[0]  # 选择第一个有效的硬件
                else:
                    continue  # 没有可用硬件，跳过

            # 检查前驱任务是否都已完成
            can_start = True
            earliest_start = hw_timelines[hw_idx]['next_available']

            for pred_id in dag.graph.predecessors(task_id):
                if pred_id not in completed_tasks:
                    can_start = False
                    break
                else:
                    # 考虑通信延迟
                    pred_idx = task_ids.index(pred_id)
                    pred_hw_idx = actions[pred_idx] if pred_idx < len(actions) else 0
                    comm_delay = dag.comm_matrix[pred_idx, task_idx, pred_hw_idx, hw_idx]
                    pred_finish = task_finish_times[pred_id] + comm_delay
                    earliest_start = max(earliest_start, pred_finish)

            if can_start and task_id not in task_start_times:
                # 任务可以开始执行
                exec_time = dag.comp_matrix[task_idx, hw_idx]
                start_time = max(earliest_start, current_time)
                finish_time = start_time + exec_time

                # 更新任务时间
                task_start_times[task_id] = start_time
                task_finish_times[task_id] = finish_time

                # 更新硬件时间线
                hw_timelines[hw_idx]['tasks'].append({
                    'task_id': task_id,
                    'start_time': start_time,
                    'finish_time': finish_time
                })
                hw_timelines[hw_idx]['next_available'] = finish_time

                # 计算能耗
                energy = dag.energy_matrix[task_idx, hw_idx]
                total_energy += energy

                scheduled_this_round = True

        # 完成当前时间点已完成的任务
        completed_this_round = set()
        for task_id in topological_order:
            if (task_id in task_finish_times and
                    task_id not in completed_tasks and
                    task_finish_times[task_id] <= current_time):
                completed_tasks.add(task_id)
                completed_this_round.add(task_id)

        # 推进时间到下一个最早完成时间
        if not scheduled_this_round and not completed_this_round:
            # 找到下一个最早完成的任务
            next_completion = float('inf')
            for task_id in topological_order:
                if task_id in task_finish_times and task_id not in completed_tasks:
                    next_completion = min(next_completion, task_finish_times[task_id])

            if next_completion != float('inf'):
                current_time = next_completion
            else:
                current_time += 1.0  # 没有任务在运行，推进一小步
        else:
            # 有任务被调度或完成，保持当前时间继续检查
            pass

    # 计算最终makespan
    makespan = max(task_finish_times.values()) if task_finish_times else 0.0

    # 计算负载状态 - 基于硬件实际使用时间
    for hw_idx in range(HARDWARE_NUM):
        hw_total_time = 0.0
        for task_info in hw_timelines[hw_idx]['tasks']:
            hw_total_time += (task_info['finish_time'] - task_info['start_time'])

        if makespan > 0:
            current_load[hw_idx] = hw_total_time / makespan
        else:
            current_load[hw_idx] = 0.0

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


def execute_scheduling_with_details(dag, actions, current_load):
    """执行调度并返回详细结果（支持并行执行）"""
    task_ids = sorted(dag.task_nodes.keys())

    # 使用改进的并行调度
    makespan, total_energy, task_finish_times = improved_execute_scheduling(dag, actions, current_load)

    # 构建详细调度信息
    task_schedule = {}
    hardware_usage = {i: {"tasks": [], "total_time": 0.0} for i in range(HARDWARE_NUM)}

    # 重新计算任务调度详情
    for task_id in task_ids:
        task_idx = task_ids.index(task_id)
        hw_idx = actions[task_idx] if task_idx < len(actions) else 0

        # 硬件支持检查
        if dag.comp_matrix[task_idx, hw_idx] == float('inf'):
            valid_hw = [j for j in range(HARDWARE_NUM) if dag.comp_matrix[task_idx, j] != float('inf')]
            hw_idx = valid_hw[0] if valid_hw else 0

        exec_time = dag.comp_matrix[task_idx, hw_idx]

        # 使用从并行调度中计算的时间
        start_time = task_finish_times[task_id] - exec_time
        finish_time = task_finish_times[task_id]

        # 记录详细调度信息
        task_schedule[task_id] = (hw_idx, start_time, finish_time)
        hardware_usage[hw_idx]["tasks"].append({
            "task_id": task_id,
            "start": start_time,
            "finish": finish_time,
            "duration": exec_time
        })
        hardware_usage[hw_idx]["total_time"] += exec_time

    # 计算截止时间满足率
    deadline_met_count = 0
    for task_id, finish_time in task_finish_times.items():
        task = dag.task_nodes[task_id]
        if finish_time <= task.deadline:
            deadline_met_count += 1

    deadline_satisfaction_rate = deadline_met_count / len(task_ids) if task_ids else 0

    # 计算负载均衡（基于硬件完成时间的方差）
    hw_finish_times = [hardware_usage[i]["total_time"] for i in range(HARDWARE_NUM)]
    load_balance = np.var(hw_finish_times)

    return {
        "makespan": makespan,
        "total_energy": total_energy,
        "deadline_satisfaction_rate": deadline_satisfaction_rate,
        "load_balance": load_balance,
        "task_schedule": task_schedule,
        "hardware_usage": hardware_usage,
        "task_finish_times": task_finish_times
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
    """Improved visualization with English labels"""
    plt.figure(figsize=(20, 12))

    # 1. Training process metrics
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

    # 2. Algorithm comparison
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

    # Output detailed results
    print("\n=== Detailed Results Comparison ===")
    for algo in baseline_results:
        print(f"{algo}: Makespan={baseline_results[algo]['makespan']:.2f}, "
              f"Energy={baseline_results[algo]['energy']:.2f}, "
              f"Deadline={baseline_results[algo]['deadline']:.2%}")

    print(f"MODRL: Makespan={modrl_results['makespan']:.2f}, "
          f"Energy={modrl_results['energy']:.2f}, "
          f"Deadline={modrl_results['deadline']:.2%}")


def debug_single_dag_example():
    """调试单个DAG例子"""
    print("\n===== 调试单个DAG调度过程 =====")

    # 生成数据集
    dataset = EmbeddedDatasetGenerator.generate_dataset()

    # 选择一个具体的DAG进行调试
    task_type = "工业控制"
    dag = dataset["split"][task_type][0]  # 选择第一个DAG

    print(f"调试DAG {dag.dag_id} ({task_type}), 任务数: {len(dag.task_nodes)}")

    # 初始化可视化器
    visualizer = DetailedSchedulerVisualizer()

    # 测试不同算法
    algorithm_results = {}

    # 1. 加载训练好的MODRL模型
    print("加载MODRL模型...")
    task_feat_dim = HARDWARE_NUM * 2 + 2
    adj_dim = MAX_TASK_NUM
    hardware_feat_dim = 4
    modrl_scheduler = MODRLScheduler(task_feat_dim, adj_dim, hardware_feat_dim).to(DEVICE)

    try:
        modrl_scheduler.load_state_dict(torch.load('modrl_final.pth', map_location=DEVICE))
        modrl_scheduler.eval()
        print("MODRL模型加载成功")

        # 2. MODRL调度
        print("运行MODRL调度...")
        task_feat, adj, seq_order, hardware_feat, load_states, history_resource = prepare_dag_input(dag)

        with torch.no_grad():
            q_values = modrl_scheduler(task_feat, adj, seq_order, hardware_feat, load_states, history_resource)
            modrl_actions = torch.argmax(q_values, dim=-1).cpu().numpy()[0]

        modrl_result = execute_scheduling_with_details(dag, modrl_actions, np.zeros(HARDWARE_NUM))
        algorithm_results["MODRL"] = modrl_result

    except FileNotFoundError:
        print("未找到训练好的MODRL模型，请先运行训练")
        return

    # 3. 随机调度（作为基线）
    print("运行随机调度...")
    random_actions = np.random.randint(0, HARDWARE_NUM, len(dag.task_nodes))
    random_result = execute_scheduling_with_details(dag, random_actions, np.zeros(HARDWARE_NUM))
    algorithm_results["Random"] = random_result

    # 可视化两种调度结果
    for algo, result in algorithm_results.items():
        visualizer.visualize_single_dag_schedule(dag, result, algo)

    # 输出对比结果
    print(f"\n=== DAG {dag.dag_id} 调度结果对比 ===")
    for algo, result in algorithm_results.items():
        print(f"{algo}:")
        print(f"  - Makespan: {result['makespan']:.2f}ms")
        print(f"  - 能耗: {result['total_energy']:.2f}J")
        print(f"  - 截止时间满足率: {result['deadline_satisfaction_rate']:.2%}")
        print(f"  - 负载均衡方差: {result['load_balance']:.4f}")

        # 输出任务分配统计
        hw_counts = {hw: 0 for hw in HARDWARE_TYPES}
        for task_id, (hw_idx, start, finish) in result["task_schedule"].items():
            hw_counts[HARDWARE_TYPES[hw_idx]] += 1
        print(f"  - 任务分配: {hw_counts}")

    # 算法对比可视化
    visualizer.compare_algorithms_on_dag(dag, algorithm_results)

# 在main()函数末尾调用
if __name__ == "__main__":
    main()
    debug_single_dag_example()  # 添加这行

