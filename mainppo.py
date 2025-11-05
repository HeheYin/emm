# main_ppo.py
import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import random
from collections import deque
from config import *
from experiment import EmbeddedDatasetGenerator, BaselineScheduler, DetailedSchedulerVisualizer


class ActorNetwork(nn.Module):
    """PPO Actor网络 - 策略网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return torch.softmax(self.network(state), dim=-1)


class CriticNetwork(nn.Module):
    """PPO Critic网络 - 价值网络"""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state)


class PPOEmbeddedScheduler:
    """基于PPO的嵌入式任务调度器"""

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 网络初始化
        self.actor = ActorNetwork(state_dim, action_dim).to(DEVICE)
        self.critic = CriticNetwork(state_dim).to(DEVICE)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        # 经验缓冲区
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)

        # PPO超参数
        self.clip_epsilon = 0.2
        self.ppo_epochs = 4
        self.batch_size = BATCH_SIZE

    def get_state_representation(self, dag, current_load_states, current_time=0):
        """获取状态表示"""
        task_ids = sorted(dag.task_nodes.keys())
        task_num = len(task_ids)

        # 任务特征
        task_features = []
        for task_id in task_ids:
            task = dag.task_nodes[task_id]
            task_idx = task_ids.index(task_id)

            # 计算任务在各硬件上的特征
            hw_features = []
            for hw_idx in range(HARDWARE_NUM):
                comp_cost = dag.comp_matrix[task_idx, hw_idx]
                energy_cost = dag.energy_matrix[task_idx, hw_idx]
                hw_features.extend([comp_cost / 100.0, energy_cost / 100.0])

            # 任务属性特征
            task_attr = [
                task.priority / 10.0,
                task.deadline / 1000.0,
                (task.deadline - current_time) / 1000.0 if current_time > 0 else 1.0
            ]

            task_features.append(hw_features + task_attr)

        # 全局特征
        global_features = list(current_load_states)  # 当前硬件负载

        # 硬件能力特征
        for hw in HARDWARE_TYPES:
            hw_info = EMBEDDED_HARDWARES[hw]
            global_features.extend([
                hw_info["算力"] / 100.0,
                hw_info["内存"] / 10.0,
                hw_info["能耗系数"],
                hw_info["负载阈值"]
            ])

        # 合并特征
        if task_features:
            avg_task_features = np.mean(task_features, axis=0)
            state = np.concatenate([avg_task_features, global_features])
        else:
            state = np.concatenate([np.zeros(len(task_features[0]) if task_features else [0]), global_features])

        return state

    def select_action(self, state, task_comp_costs, epsilon=0.1):
        """选择动作 - 考虑硬件约束"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action_probs = self.actor(state_tensor).cpu().numpy()[0]

        # 屏蔽不支持该任务的硬件
        masked_probs = action_probs.copy()
        for hw_idx in range(self.action_dim):
            if task_comp_costs[hw_idx] == float('inf'):
                masked_probs[hw_idx] = 0.0

        # 重新归一化概率
        if np.sum(masked_probs) > 0:
            masked_probs /= np.sum(masked_probs)
        else:
            # 如果没有可用硬件，均匀分布
            valid_actions = [i for i in range(self.action_dim) if task_comp_costs[i] != float('inf')]
            if valid_actions:
                masked_probs = np.zeros(self.action_dim)
                for i in valid_actions:
                    masked_probs[i] = 1.0 / len(valid_actions)
            else:
                masked_probs = np.ones(self.action_dim) / self.action_dim

        # ε-greedy探索
        if random.random() < epsilon:
            valid_actions = [i for i in range(self.action_dim) if masked_probs[i] > 0]
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = random.randint(0, self.action_dim - 1)
        else:
            action = np.random.choice(self.action_dim, p=masked_probs)

        return action, masked_probs

    def store_experience(self, state, action, action_prob, reward, next_state, done):
        """存储经验"""
        self.memory.append({
            'state': state,
            'action': action,
            'action_prob': action_prob,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def update(self):
        """PPO更新"""
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0

        # 准备训练数据
        states = torch.FloatTensor([exp['state'] for exp in self.memory]).to(DEVICE)
        actions = torch.LongTensor([exp['action'] for exp in self.memory]).to(DEVICE)
        old_probs = torch.FloatTensor([exp['action_prob'] for exp in self.memory]).to(DEVICE)
        rewards = torch.FloatTensor([exp['reward'] for exp in self.memory]).to(DEVICE)
        next_states = torch.FloatTensor([exp['next_state'] for exp in self.memory]).to(DEVICE)
        dones = torch.FloatTensor([exp['done'] for exp in self.memory]).to(DEVICE)

        # 计算优势函数
        with torch.no_grad():
            current_values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            targets = rewards + GAMMA * next_values * (1 - dones)
            advantages = targets - current_values

        # 多轮PPO更新
        actor_losses = []
        critic_losses = []

        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(self.memory))

            for start in range(0, len(self.memory), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_targets = targets[batch_indices]

                # 更新Actor
                current_probs = self.actor(batch_states)
                batch_current_probs = current_probs.gather(1, batch_actions.unsqueeze(1)).squeeze()

                ratio = batch_current_probs / (batch_old_probs + 1e-8)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
                self.actor_optimizer.step()

                # 更新Critic
                current_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_targets)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        # 清空经验池
        self.memory.clear()

        return np.mean(actor_losses), np.mean(critic_losses)

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class PPOSchedulingEnvironment:
    """PPO调度环境"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.current_dag = None
        self.current_time = 0.0
        self.completed_tasks = set()
        self.hw_timelines = None
        self.task_finish_times = {}
        self.task_start_times = {}
        self.total_energy = 0.0  # 添加总能耗记录
        self.ppo_scheduler = None  # 添加PPO调度器引用

    def get_state(self, current_load_states):
        """获取当前状态"""
        if self.ppo_scheduler is None:
            raise ValueError("PPO调度器未设置，请先设置env.ppo_scheduler")

        return self.ppo_scheduler.get_state_representation(
            self.current_dag, current_load_states, self.current_time
        )

    def reset(self, task_type=None):
        """重置环境"""
        if task_type is None:
            task_type = random.choice(TASK_TYPES)

        self.current_dag = random.choice(self.dataset["split"][task_type])
        self.current_time = 0.0
        self.completed_tasks = set()
        self.task_finish_times = {}
        self.task_start_times = {}
        self.total_energy = 0.0  # 重置能耗

        # 初始化硬件时间线
        self.hw_timelines = {
            i: {
                'tasks': [],
                'next_available': 0.0
            } for i in range(HARDWARE_NUM)
        }

        # 获取初始状态
        current_load_states = np.zeros(HARDWARE_NUM)
        state = self.get_state(current_load_states)

        return state, self.current_dag

    def step(self, actions, current_load_states):
        """执行一步调度"""
        task_ids = sorted(self.current_dag.task_nodes.keys())
        makespan = 0.0
        scheduled_count = 0

        # 获取拓扑顺序
        try:
            topological_order = list(nx.topological_sort(self.current_dag.graph))
        except nx.NetworkXUnfeasible:
            topological_order = task_ids

        # 执行调度
        for task_id in topological_order:
            if task_id in self.completed_tasks or task_id in self.task_start_times:
                continue

            task_idx = task_ids.index(task_id)
            hw_idx = actions[task_idx] if task_idx < len(actions) else 0

            # 检查硬件支持
            if self.current_dag.comp_matrix[task_idx, hw_idx] == float('inf'):
                valid_hw = [j for j in range(HARDWARE_NUM)
                            if self.current_dag.comp_matrix[task_idx, j] != float('inf')]
                if not valid_hw:
                    continue
                hw_idx = valid_hw[0]

            # 检查前驱任务是否完成
            can_start = True
            earliest_start = self.hw_timelines[hw_idx]['next_available']

            for pred_id in self.current_dag.graph.predecessors(task_id):
                if pred_id not in self.completed_tasks:
                    can_start = False
                    break
                else:
                    # 考虑通信延迟
                    pred_idx = task_ids.index(pred_id)
                    pred_hw_idx = actions[pred_idx] if pred_idx < len(actions) else 0
                    comm_delay = self.current_dag.comm_matrix[pred_idx, task_idx, pred_hw_idx, hw_idx]
                    pred_finish = self.task_finish_times[pred_id] + comm_delay
                    earliest_start = max(earliest_start, pred_finish)

            if can_start:
                # 执行任务
                exec_time = self.current_dag.comp_matrix[task_idx, hw_idx]
                start_time = max(earliest_start, self.current_time)
                finish_time = start_time + exec_time

                # 记录任务时间
                self.task_start_times[task_id] = start_time
                self.task_finish_times[task_id] = finish_time

                # 更新硬件时间线
                self.hw_timelines[hw_idx]['tasks'].append({
                    'task_id': task_id,
                    'start_time': start_time,
                    'finish_time': finish_time
                })
                self.hw_timelines[hw_idx]['next_available'] = finish_time

                # 修复：正确计算能耗
                energy = self.current_dag.energy_matrix[task_idx, hw_idx]
                self.total_energy += energy  # 累加总能耗

                scheduled_count += 1

        # 更新完成的任务
        for task_id in topological_order:
            if (task_id in self.task_finish_times and
                    task_id not in self.completed_tasks and
                    self.task_finish_times[task_id] <= self.current_time):
                self.completed_tasks.add(task_id)

        # 计算新的负载状态
        for hw_idx in range(HARDWARE_NUM):
            hw_total_time = 0.0
            for task_info in self.hw_timelines[hw_idx]['tasks']:
                if task_info['finish_time'] <= self.current_time:
                    hw_total_time += (task_info['finish_time'] - task_info['start_time'])

            if self.current_time > 0:
                current_load_states[hw_idx] = hw_total_time / self.current_time
            else:
                current_load_states[hw_idx] = 0.0

        # 检查是否完成
        done = len(self.completed_tasks) == len(task_ids)
        if done:
            makespan = max(self.task_finish_times.values()) if self.task_finish_times else 0.0
        else:
            # 推进时间
            self.current_time += 1.0
            makespan = self.current_time

        # 计算奖励
        reward = self.calculate_reward(makespan, current_load_states, self.total_energy, scheduled_count)

        # 获取下一个状态
        next_state = self.get_state(current_load_states)

        return next_state, reward, done, {
            'makespan': makespan,
            'energy': self.total_energy,  # 使用累加的能耗
            'scheduled_count': scheduled_count
        }

    def calculate_reward(self, makespan, load_states, energy, scheduled_count):
        """计算奖励"""
        # 简化奖励函数
        task_num = len(self.current_dag.task_nodes)

        # 进度奖励
        progress_reward = scheduled_count / task_num if task_num > 0 else 0

        # 时间奖励
        time_penalty = -makespan / 1000.0

        # 负载均衡奖励
        load_balance_reward = 1.0 - np.std(load_states)

        # 能耗惩罚
        energy_penalty = -energy / 1000.0

        # 综合奖励
        total_reward = (
                progress_reward * 0.3 +
                time_penalty * 0.3 +
                load_balance_reward * 0.3 +
                energy_penalty * 0.1
        )

        return total_reward * REWARD_SCALE


def train_ppo_scheduler():
    """训练PPO调度器"""
    print("===== 生成嵌入式任务数据集 =====")
    dataset = EmbeddedDatasetGenerator.generate_dataset()

    # 状态维度计算
    sample_dag = dataset["split"]["工业控制"][0]
    sample_state_dim = len(PPOEmbeddedScheduler(0, 0).get_state_representation(sample_dag, np.zeros(HARDWARE_NUM)))
    action_dim = HARDWARE_NUM

    print(f"状态维度: {sample_state_dim}, 动作维度: {action_dim}")

    # 初始化PPO调度器和环境
    ppo_scheduler = PPOEmbeddedScheduler(sample_state_dim, action_dim)
    env = PPOSchedulingEnvironment(dataset)
    env.ppo_scheduler = ppo_scheduler  # 正确设置PPO调度器引用

    # 训练参数
    total_episodes = 100
    max_steps_per_episode = 200
    epsilon = 0.2
    epsilon_decay = 0.995
    min_epsilon = 0.01

    # 记录指标
    metrics = {
        'episode_rewards': [],
        'episode_makespans': [],
        'episode_energies': [],
        'actor_losses': [],
        'critic_losses': [],
        'epsilon': []
    }

    print("===== 开始PPO训练 =====")
    progress_bar = tqdm(range(total_episodes), desc="训练进度")

    for episode in progress_bar:
        # 重置环境
        state, dag = env.reset()
        current_load_states = np.zeros(HARDWARE_NUM)

        episode_reward = 0.0
        episode_makespan = 0.0
        episode_energy = 0.0
        step_count = 0

        # 为整个DAG选择动作
        task_ids = sorted(dag.task_nodes.keys())
        actions = []
        action_probs = []

        for task_idx, task_id in enumerate(task_ids):
            task_comp_costs = dag.comp_matrix[task_idx]
            action, action_prob = ppo_scheduler.select_action(
                state, task_comp_costs, epsilon
            )
            actions.append(action)
            action_probs.append(action_prob[action])

        # 执行episode
        for step in range(max_steps_per_episode):
            # 执行一步
            next_state, reward, done, info = env.step(actions, current_load_states)

            # 存储经验
            for task_idx in range(len(task_ids)):
                ppo_scheduler.store_experience(
                    state, actions[task_idx], action_probs[task_idx],
                    reward, next_state, done
                )

            state = next_state
            episode_reward += reward
            episode_makespan = info['makespan']
            episode_energy = info['energy']
            step_count += 1

            if done:
                break

        # 更新PPO网络
        actor_loss, critic_loss = ppo_scheduler.update()

        # 衰减探索率
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # 记录指标
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_makespans'].append(episode_makespan)
        metrics['episode_energies'].append(episode_energy)
        metrics['actor_losses'].append(actor_loss)
        metrics['critic_losses'].append(critic_loss)
        metrics['epsilon'].append(epsilon)

        # 更新进度条
        progress_bar.set_postfix({
            'Reward': f'{episode_reward:.2f}',
            'Makespan': f'{episode_makespan:.2f}',
            'Epsilon': f'{epsilon:.3f}',
            'Actor Loss': f'{actor_loss:.4f}' if actor_loss != 0 else '0.0000'
        })

        # 定期保存模型
        if episode % 50 == 0 and episode > 0:
            ppo_scheduler.save_model(f'ppo_scheduler_ep{episode}.pth')

    # 保存最终模型
    ppo_scheduler.save_model('ppo_scheduler_final.pth')

    # 绘制训练曲线
    plot_training_metrics(metrics)

    return ppo_scheduler, metrics


def plot_training_metrics(metrics):
    """绘制训练指标"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 奖励曲线
    ax1.plot(metrics['episode_rewards'])
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)

    # Makespan曲线
    ax2.plot(metrics['episode_makespans'])
    ax2.set_title('Episode Makespans')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Makespan (ms)')
    ax2.grid(True)

    # 损失曲线
    ax3.plot(metrics['actor_losses'], label='Actor Loss')
    ax3.plot(metrics['critic_losses'], label='Critic Loss')
    ax3.set_title('Training Losses')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)

    # 探索率曲线
    ax4.plot(metrics['epsilon'])
    ax4.set_title('Exploration Rate (Epsilon)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('ppo_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_ppo_scheduler(ppo_scheduler, dataset):
    """评估PPO调度器"""
    print("===== 评估PPO调度器 =====")

    env = PPOSchedulingEnvironment(dataset)
    env.ppo_scheduler = ppo_scheduler

    results = {
        'makespans': [],
        'energies': [],
        'deadline_rates': [],
        'load_balances': []
    }

    for task_type in TASK_TYPES:
        test_dags = dataset["split"][task_type][:20]  # 每个类型测试20个DAG

        for dag in tqdm(test_dags, desc=f"评估{task_type}"):
            state, _ = env.reset()
            env.current_dag = dag  # 手动设置DAG
            current_load_states = np.zeros(HARDWARE_NUM)

            # 选择动作
            task_ids = sorted(dag.task_nodes.keys())
            actions = []

            for task_idx, task_id in enumerate(task_ids):
                task_comp_costs = dag.comp_matrix[task_idx]
                action, _ = ppo_scheduler.select_action(state, task_comp_costs, epsilon=0.0)
                actions.append(action)

            # 执行完整调度
            done = False
            while not done:
                state, reward, done, info = env.step(actions, current_load_states)

            # 计算指标
            makespan = info['makespan']
            energy = info['energy']

            # 调试输出：打印能耗信息
            print(f"DAG {dag.dag_id} 能耗计算:")
            print(f"  - 任务数: {len(task_ids)}")
            print(f"  - 总能耗: {energy:.2f} J")
            print(f"  - 平均每任务能耗: {energy/len(task_ids) if task_ids else 0:.2f} J")

            # 计算截止时间满足率
            deadline_met = 0
            for task_id, finish_time in env.task_finish_times.items():
                task = dag.task_nodes[task_id]
                if finish_time <= task.deadline:
                    deadline_met += 1
            deadline_rate = deadline_met / len(task_ids) if task_ids else 0

            # 计算负载均衡
            hw_times = [env.hw_timelines[i]['next_available'] for i in range(HARDWARE_NUM)]
            load_balance = 1.0 - np.std(hw_times) / (np.mean(hw_times) + 1e-6)

            results['makespans'].append(makespan)
            results['energies'].append(energy)
            results['deadline_rates'].append(deadline_rate)
            results['load_balances'].append(load_balance)

    # 计算平均指标
    avg_results = {
        'makespan': np.mean(results['makespans']),
        'energy': np.mean(results['energies']),
        'deadline_rate': np.mean(results['deadline_rates']),
        'load_balance': np.mean(results['load_balances'])
    }

    print("\nPPO调度器评估结果:")
    print(f"平均Makespan: {avg_results['makespan']:.2f} ms")
    print(f"平均能耗: {avg_results['energy']:.2f} J")
    print(f"截止时间满足率: {avg_results['deadline_rate']:.2%}")
    print(f"负载均衡: {avg_results['load_balance']:.4f}")

    return avg_results


def compare_with_baselines(ppo_results, dataset):
    """与基线算法对比"""
    print("===== 与基线算法对比 =====")

    baseline_results = {}

    for algo in BASELINE_ALGORITHMS:
        print(f"评估基线算法: {algo}")
        makespans = []
        energies = []

        for task_type in TASK_TYPES:
            test_dags = dataset["split"][task_type][:10]

            for dag in test_dags:
                if algo == "HEFT":
                    makespan, load_balance = BaselineScheduler.heft_schedule(dag)
                elif algo == "RM":
                    makespan, load_balance = BaselineScheduler.rm_schedule(dag)
                elif algo == "EDF":
                    makespan, load_balance = BaselineScheduler.edf_schedule(dag)
                else:
                    makespan, load_balance = 0.0, 0.0

                # 估算能耗
                energy = np.sum(dag.energy_matrix)

                makespans.append(makespan)
                energies.append(energy)

        baseline_results[algo] = {
            'makespan': np.mean(makespans),
            'energy': np.mean(energies)
        }

    # 可视化对比
    algorithms = list(baseline_results.keys()) + ['PPO']
    makespans = [baseline_results[algo]['makespan'] for algo in baseline_results] + [ppo_results['makespan']]
    energies = [baseline_results[algo]['energy'] for algo in baseline_results] + [ppo_results['energy']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Makespan对比
    bars1 = ax1.bar(algorithms, makespans, color=['lightblue', 'lightgreen', 'lightcoral', 'orange'])
    ax1.set_title('Makespan Comparison')
    ax1.set_ylabel('Makespan (ms)')
    ax1.set_xticklabels(algorithms, rotation=45)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}', ha='center', va='bottom')

    # 能耗对比
    bars2 = ax2.bar(algorithms, energies, color=['lightblue', 'lightgreen', 'lightcoral', 'orange'])
    ax2.set_title('Energy Consumption Comparison')
    ax2.set_ylabel('Energy (J)')
    ax2.set_xticklabels(algorithms, rotation=45)

    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('ppo_vs_baselines.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 输出详细对比
    print("\n算法对比结果:")
    for algo in baseline_results:
        print(f"{algo}: Makespan={baseline_results[algo]['makespan']:.2f}, "
              f"Energy={baseline_results[algo]['energy']:.2f}")

    print(f"PPO: Makespan={ppo_results['makespan']:.2f}, "
          f"Energy={ppo_results['energy']:.2f}, "
          f"Deadline Rate={ppo_results['deadline_rate']:.2%}")


def demo_ppo_scheduling(ppo_scheduler, dataset):
    """演示PPO调度过程"""
    print("===== PPO调度演示 =====")

    visualizer = DetailedSchedulerVisualizer()

    # 选择一个测试DAG
    task_type = "工业控制"
    dag = dataset["split"][task_type][0]

    print(f"演示DAG: {dag.dag_id} ({task_type}), 任务数: {len(dag.task_nodes)}")

    # 使用PPO进行调度
    env = PPOSchedulingEnvironment(dataset)
    env.ppo_scheduler = ppo_scheduler
    state, _ = env.reset()
    env.current_dag = dag

    current_load_states = np.zeros(HARDWARE_NUM)
    task_ids = sorted(dag.task_nodes.keys())
    actions = []

    # 选择动作
    for task_idx, task_id in enumerate(task_ids):
        task_comp_costs = dag.comp_matrix[task_idx]
        action, _ = ppo_scheduler.select_action(state, task_comp_costs, epsilon=0.0)
        actions.append(action)

    # 执行调度
    done = False
    while not done:
        state, reward, done, info = env.step(actions, current_load_states)

    # 构建调度结果
    schedule_result = {
        "makespan": info['makespan'],
        "total_energy": info['energy'],
        "deadline_satisfaction_rate": 0.0,  # 需要单独计算
        "load_balance": 0.0,  # 需要单独计算
        "task_schedule": {},
        "hardware_usage": {i: {"total_time": env.hw_timelines[i]['next_available']}
                           for i in range(HARDWARE_NUM)}
    }

    # 计算截止时间满足率
    deadline_met = 0
    for task_id, finish_time in env.task_finish_times.items():
        task = dag.task_nodes[task_id]
        task_idx = task_ids.index(task_id)
        hw_idx = actions[task_idx]

        schedule_result["task_schedule"][task_id] = (hw_idx,
                                                     env.task_start_times[task_id],
                                                     finish_time)

        if finish_time <= task.deadline:
            deadline_met += 1

    schedule_result["deadline_satisfaction_rate"] = deadline_met / len(task_ids)

    # 计算负载均衡
    hw_times = [env.hw_timelines[i]['next_available'] for i in range(HARDWARE_NUM)]
    schedule_result["load_balance"] = np.var(hw_times)

    # 可视化调度结果
    visualizer.visualize_single_dag_schedule(dag, schedule_result, "PPO")

    print(f"调度完成: Makespan={info['makespan']:.2f}ms, "
          f"Energy={info['energy']:.2f}J, "
          f"Deadline Rate={schedule_result['deadline_satisfaction_rate']:.2%}")


if __name__ == "__main__":
    # 设置随机种子
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"使用设备: {DEVICE}")

    # 训练PPO调度器
    ppo_scheduler, training_metrics = train_ppo_scheduler()
    # 生成测试数据集
    test_dataset = EmbeddedDatasetGenerator.generate_dataset()
    # 评估PPO调度器
    ppo_results = evaluate_ppo_scheduler(ppo_scheduler, test_dataset)
    # 与基线算法对比
    compare_with_baselines(ppo_results, test_dataset)
    # 演示调度过程
    demo_ppo_scheduling(ppo_scheduler, test_dataset)

    print("PPO嵌入式任务调度系统运行完成!")