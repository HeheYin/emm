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
    epsilon_strategy =  StableEpsilonGreedyStrategy()

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
        for task_idx in range(len(task_ids)):
            action = select_stable_action(
                q_values[0, task_idx],
                epsilon,
                dag.comp_matrix[task_idx]
            )
            actions.append(action)

        actions = np.array(actions)

        # 执行调度
        current_makespan, task_energy = execute_stable_scheduling(dag, actions, current_load)
        episode_makespan = current_makespan
        episode_reward = 0.0  # 会在后面计算

        # 计算负载状态
        load_states_np = current_load.copy()
        hardware_thresholds = [EMBEDDED_HARDWARES[hw]["负载阈值"] for hw in HARDWARE_TYPES]

        # 计算奖励
        reward = MultiObjectiveReward.calculate_reward(
            current_makespan,
            load_states_np,
            hardware_thresholds,
            task_energy,
            dag.task_nodes[0].priority,
            dag.task_type
        )

        episode_reward = reward

        # 检查截止时间
        if current_makespan <= dag.task_nodes[0].deadline:
            deadline_satisfied += 1
        total_tasks += 1

        # 经验回放存储
        replay_buffer.append((
            task_feat, adj, seq_order, hardware_feat, load_states,
            history_resource, actions, reward
        ))

        if len(replay_buffer) > REPLAY_BUFFER_SIZE:
            replay_buffer.pop(0)

        # 训练更新 - 每步都训练，但使用小学习率
        episode_loss = 0.0
        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            episode_loss = train_stable_batch(batch, modrl_scheduler, optimizer)
            modrl_scheduler.soft_update_target()

        # 记录指标
        metrics["makespan"].append(episode_makespan)
        metrics["load_balance"].append(np.var(current_load))
        metrics["energy_consumption"].append(task_energy)
        metrics["deadline_satisfaction"].append(deadline_satisfied / total_tasks if total_tasks > 0 else 0)
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
    modrl_metrics = evaluate_modrl(modrl_scheduler, dataset)

    # 9. 结果可视化
    visualize_stable_results(metrics, baseline_results, modrl_metrics)
    print("实验完成！")


def select_stable_action(q_values, epsilon, comp_costs):
    """稳定的动作选择"""
    # 创建有效动作掩码
    valid_actions = [i for i, cost in enumerate(comp_costs) if cost != float('inf')]
    if not valid_actions:
        return random.randint(0, HARDWARE_NUM - 1)

    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        q_values_np = q_values.detach().cpu().numpy()

        # 对无效动作设置很小的值
        masked_q_values = q_values_np.copy()
        for i in range(HARDWARE_NUM):
            if i not in valid_actions:
                masked_q_values[i] = -1e6

        return np.argmax(masked_q_values)


def execute_stable_scheduling(dag, actions, current_load):
    """稳定的调度执行"""
    task_ids = sorted(dag.task_nodes.keys())
    makespan = 0.0
    total_energy = 0.0
    hw_finish_times = np.zeros(HARDWARE_NUM)

    # 确保actions格式正确
    if isinstance(actions, (int, np.int64)):
        actions = [actions] * len(task_ids)
    elif isinstance(actions, np.ndarray) and actions.ndim == 0:
        actions = [actions.item()] * len(task_ids)

    if len(actions) != len(task_ids):
        actions = [actions[0]] * len(task_ids) if actions else [0] * len(task_ids)

    for i, task_id in enumerate(task_ids):
        task_idx = i
        hw_idx = actions[i]

        # 检查硬件支持
        if dag.comp_matrix[task_idx, hw_idx] == float('inf'):
            valid_hw = [j for j in range(HARDWARE_NUM) if dag.comp_matrix[task_idx, j] != float('inf')]
            if valid_hw:
                hw_idx = valid_hw[0]
            else:
                continue

        exec_time = dag.comp_matrix[task_idx, hw_idx]
        energy = dag.energy_matrix[task_idx, hw_idx]

        # 计算开始时间
        start_time = float(hw_finish_times[hw_idx])

        # 考虑任务依赖
        for pred_id in dag.graph.predecessors(task_id):
            start_time = max(start_time, float(hw_finish_times[hw_idx]))

        finish_time = start_time + exec_time
        hw_finish_times[hw_idx] = finish_time

        # 更新负载
        current_load[hw_idx] = min(current_load[hw_idx] + exec_time / 1000.0, 1.0)  # 更温和的负载更新
        makespan = max(makespan, finish_time)
        total_energy += energy

    return makespan, total_energy


def train_stable_batch(batch, scheduler, optimizer):
    """稳定的批次训练"""
    optimizer.zero_grad()
    total_loss = 0.0
    valid_samples = 0

    for sample in batch:
        task_feat, adj, seq_order, hardware_feat, load_states, history_resource, actions, reward = sample

        # 跳过无效样本
        if np.isnan(reward) or np.isinf(reward):
            continue

        device = task_feat.device
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)

        # 计算当前Q值
        current_state_features = scheduler.prepare_state_features(
            task_feat, adj, seq_order, hardware_feat, load_states, history_resource
        )
        current_q_values = scheduler.q_network(current_state_features)

        # 使用当前状态作为下一个状态（简化处理）
        with torch.no_grad():
            next_q_values_target = scheduler.target_q_network(current_state_features)
            next_q_max = torch.max(next_q_values_target, dim=-1)[0]
            target_q = reward_tensor + GAMMA * next_q_max

        # 计算当前Q值
        batch_size, task_num, _ = current_q_values.shape
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)

        if actions_tensor.dim() == 1:
            actions_tensor = actions_tensor.unsqueeze(0).unsqueeze(-1)

        current_q = current_q_values.gather(-1, actions_tensor).squeeze(-1)

        # 计算Huber损失，对异常值更稳定
        loss = F.smooth_l1_loss(current_q, target_q)

        # 检查损失是否为NaN
        if not torch.isnan(loss) and not torch.isinf(loss):
            total_loss += loss
            valid_samples += 1

    if valid_samples > 0:
        (total_loss / valid_samples).backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(scheduler.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        return (total_loss / valid_samples).item()
    else:
        return 0.0


def evaluate_modrl(scheduler, dataset):
    """评估MODRL性能"""
    makespans = []
    energies = []
    deadline_rates = []

    scheduler.eval()  # 切换到评估模式

    with torch.no_grad():
        for task_type in TASK_TYPES:
            for dag in dataset["split"][task_type][:10]:  # 只评估前10个以节省时间
                # 构建输入（与训练时相同）
                task_ids = sorted(dag.task_nodes.keys())
                task_num = len(task_ids)

                task_feat = np.zeros((task_num, HARDWARE_NUM * 2 + 2))
                for i, task_id in enumerate(task_ids):
                    task = dag.task_nodes[task_id]
                    for j in range(HARDWARE_NUM):
                        task_feat[i, j] = dag.comp_matrix[i, j] / 100.0
                        task_feat[i, j + HARDWARE_NUM] = dag.energy_matrix[i, j] / 100.0
                    task_feat[i, -2] = task.priority / 10.0
                    task_feat[i, -1] = task.deadline / 1000.0

                task_feat = torch.tensor(task_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                adj = torch.tensor(nx.to_numpy_array(dag.graph), dtype=torch.float32).unsqueeze(0).to(DEVICE)

                priorities = [dag.task_nodes[tid].priority for tid in task_ids]
                sorted_indices = sorted(range(len(priorities)), key=lambda i: priorities[i], reverse=True)
                seq_order = torch.tensor(sorted_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

                hardware_feat = []
                for hw in HARDWARE_TYPES:
                    hw_info = EMBEDDED_HARDWARES[hw]
                    hardware_feat.append([
                        hw_info["算力"] / 100.0,
                        hw_info["内存"] / 10.0,
                        hw_info["能耗系数"],
                        hw_info["负载阈值"]
                    ])
                hardware_feat = torch.tensor(hardware_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                load_states = torch.tensor(np.zeros(HARDWARE_NUM), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                history_resource = torch.randn(1, 10, 4, dtype=torch.float32).to(DEVICE) * 0.1

                # 贪婪策略选择动作
                q_values = scheduler(task_feat, adj, seq_order, hardware_feat, load_states, history_resource)
                actions = torch.argmax(q_values, dim=-1).cpu().numpy()[0]

                # 执行调度
                current_load = np.zeros(HARDWARE_NUM)
                makespan, energy = execute_stable_scheduling(dag, actions, current_load)

                makespans.append(makespan)
                energies.append(energy)

                # 计算截止时间满足率
                deadline_met = 1 if makespan <= dag.task_nodes[0].deadline else 0
                deadline_rates.append(deadline_met)

    scheduler.train()  # 切换回训练模式

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