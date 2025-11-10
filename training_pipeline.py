import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import networkx as nx
from tqdm import tqdm
from typing import Tuple, List, Dict

# PyG (PyTorch Geometric)
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

# 从您的项目中导入
from mcs_env import MCSSchedulingEnv
from model_hgat import HGATActorCritic
from agent_ppo import PPOAgent, RolloutBuffer
from dag_utils import (
    Task, Processor, create_dag_from_spec,
    STATUS_UNREADY, STATUS_READY, STATUS_COMPLETED,
    PROC_CPU, PROC_NPU
)
# 导入 HEFT 的核心逻辑
from baseline_heft import compute_upward_ranks, get_avg_exec_time, get_avg_comm_cost

"""
实现 Sec 4.1: 两阶段训练策略
"""


def generate_expert_trajectories(
        dag_spec: dict,
        proc_spec: list,
        data_path: str = "data/",
        num_trajectories: int = 1):
    """
    阶段1: 行为克隆 (BC) - 专家轨迹生成

    [已实现]

    运行"状态感知"的HEFT算法来生成 (s, a) 专家对。
    [cite: 115, 116]
    """
    print(f"--- [阶段 1 BC] 正在生成 {num_trajectories} 条专家轨迹... ---")

    all_states: List[HeteroData] = []
    all_actions: List[int] = []

    # 注意：我们目前没有随机DAG生成器，
    # 所以我们只在同一个DAG上生成一次轨迹。
    # （在实际应用中，num_trajectories 会与随机DAG生成器一起使用）

    # 1. 初始化HEFT所需的组件
    dag, task_map = create_dag_from_spec(**dag_spec)
    processors = [Processor(**p) for p in proc_spec]
    num_tasks = len(task_map)
    num_procs = len(processors)

    # 2. 计算向上排名 (HEFT 阶段 1)
    ranks = compute_upward_ranks(dag, processors)
    task_queue = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    # 3. 模拟HEFT调度 (HEFT 阶段 2)

    # 跟踪HEFT的内部状态
    task_finish_times: Dict[int, float] = {}
    proc_schedules: Dict[int, List[Tuple[float, float, int]]] = {p.id: [] for p in processors}

    # 跟踪用于构建HGAT图的"模拟"状态
    task_status: Dict[int, int] = {t_id: STATUS_UNREADY for t_id in task_map}
    proc_becomes_free_at: Dict[int, float] = {p.id: 0.0 for p in processors}

    # 初始化就绪任务 (那些没有前驱的任务)
    ready_tasks_for_status = []
    for n_id in dag.nodes:
        if dag.in_degree(n_id) == 0:
            task_status[n_id] = STATUS_READY
            ready_tasks_for_status.append(n_id)

    print(f"HEFT 任务队列 (按 rank_u 排序): {[t[0] for t in task_queue]}")

    for task_id, rank in task_queue:
        task = task_map[task_id]

        # -----------------------------------------------------------------
        # (A) 捕获当前状态 (s_t) [cite: 117]
        # 我们必须在HEFT决策 *之前* 手动构建HGAT状态图
        # 这模拟了 mcs_env.py 中的 _get_obs()
        # -----------------------------------------------------------------

        # 1. 构建节点特征
        task_features = []
        for t_id in range(num_tasks):
            t = task_map[t_id]
            # HEFT是静态的，current_time=0，因此 laxity 也是静态的
            laxity = t.deadline - t.exec_times[PROC_NPU] if t.deadline != float('inf') else 1e6
            clipped_laxity = max(laxity, -1000.0)  # 应用稳定性修复
            deadline_feat = t.deadline if t.deadline != float('inf') else 1e6

            task_features.append([
                t.exec_times[PROC_CPU],
                t.exec_times[PROC_NPU],
                t.criticality,
                deadline_feat,
                clipped_laxity,
                task_status[t_id],  # 使用我们模拟的状态
            ])

        proc_features = []
        for p in processors:
            # HEFT 没有 "load" 或 "crit"，但它有 "becomes_free_at"
            # 我们用 1.0 (如果繁忙) 或 0.0 (如果空闲) 作为 "load" 代理
            load = 1.0 if proc_becomes_free_at[p.id] > 0 else 0.0
            proc_features.append([
                p.type,
                load,
                0,  # queue_len (HEFT中为0)
                0,  # curr_crit (HEFT中为0)
            ])

        s_t = HeteroData()
        s_t['task'].x = torch.tensor(task_features, dtype=torch.float)
        s_t['proc'].x = torch.tensor(proc_features, dtype=torch.float)

        # 2. 构建边 [cite: 46]
        task_task_edges = []
        for u, v in dag.edges():
            task_task_edges.append((u, v))
        if task_task_edges:
            s_t['task', 'depends_on', 'task'].edge_index = torch.tensor(task_task_edges,
                                                                        dtype=torch.long).t().contiguous()
        else:
            s_t['task', 'depends_on', 'task'].edge_index = torch.empty((2, 0), dtype=torch.long)

        task_proc_edges = []
        for t_id in range(num_tasks):
            for p_id in range(num_procs):
                task_proc_edges.append((t_id, p_id))
        s_t['task', 'can_run_on', 'proc'].edge_index = torch.tensor(task_proc_edges, dtype=torch.long).t().contiguous()
        s_t['proc', 'rev_can_run_on', 'task'].edge_index = s_t['task', 'can_run_on', 'proc'].edge_index[[1, 0]]

        all_states.append(s_t)

        # -----------------------------------------------------------------
        # (B) 计算专家动作 (a_t) [cite: 118]
        # (这是 baseline_heft.py 中 schedule_heft 的内部逻辑)
        # -----------------------------------------------------------------

        best_eft = float('inf')
        best_proc_id = -1
        best_start_time = 0

        # 计算任务的“就绪时间”(Ready Time)
        ready_time = 0.0
        for pred_id in dag.predecessors(task_id):
            # 注意：HEFT 假设 0 通信成本，我们在此复制该行为
            comm_cost = 0
            ready_time = max(ready_time, task_finish_times[pred_id] + comm_cost)

        # 寻找能最早完成此任务的处理器
        for proc in processors:
            exec_time = task.exec_times[proc.type]

            # HEFT的EFT计算
            proc_ready_time = proc_schedules[proc.id][-1][1] if proc_schedules[proc.id] else 0.0

            start_time = max(ready_time, proc_ready_time)
            finish_time = start_time + exec_time

            if finish_time < best_eft:
                best_eft = finish_time
                best_proc_id = proc.id
                best_start_time = start_time

        # 将扁平化的动作索引存盘
        a_t = task_id * num_procs + best_proc_id
        all_actions.append(a_t)

        # -----------------------------------------------------------------
        # (C) 更新模拟状态，以便下一次 (A) 捕获
        # -----------------------------------------------------------------
        proc_schedules[best_proc_id].append((best_start_time, best_eft, task_id))
        task_finish_times[task_id] = best_eft

        # 更新我们的"模拟"状态
        task_status[task_id] = STATUS_COMPLETED
        proc_becomes_free_at[best_proc_id] = best_eft

        # 更新后继任务的状态
        for succ_id in dag.successors(task_id):
            ready = True
            for pred_id in dag.predecessors(succ_id):
                if task_status[pred_id] != STATUS_COMPLETED:
                    ready = False
                    break
            if ready:
                task_status[succ_id] = STATUS_READY

    # 结束循环 - 保存轨迹
    os.makedirs(data_path, exist_ok=True)
    save_path = os.path.join(data_path, "expert_trajectories.pth")
    torch.save((all_states, all_actions), save_path)

    print(f"--- [阶段 1 BC] 成功生成并保存了 {len(all_states)} 个 (s, a) 对到 {save_path} ---")
    return True


# -----------------------------------------------------------------------------
# PyG 数据集辅助类
# -----------------------------------------------------------------------------
class ExpertDataset(torch.utils.data.Dataset):
    """
    一个简单的 PyTorch Dataset，用于包装 (state, action) 对。
    它将 (HeteroData, int) 转换为 (HeteroData)
    其中 action 被附加为 `data.y` 属性。
    """

    def __init__(self, states: List[HeteroData], actions: List[int]):
        self.states = states
        self.actions = torch.tensor(actions, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # PyG DataLoader 需要返回一个 Data/HeteroData 对象
        # 我们将 action (label) 附加为 'y' 属性
        data = self.states[idx].clone()
        data.y = self.actions[idx]
        return data


def train_behavioral_cloning(
        model: HGATActorCritic,
        data_path: str = "data/",
        epochs: int = 50,
        batch_size: int = 4,  # 批处理大小必须很小，因为 (s,a) 对很少
        lr: float = 1e-3) -> HGATActorCritic:
    """
    阶段1: 行为克隆 (BC) - 监督学习

    [已实现]

    使用专家轨迹 (s, a) 对来预训练策略网络。
    [cite: 119]
    """
    print("--- [阶段 1 BC] 开始监督学习预训练... ---")

    # 1. 加载数据
    load_path = os.path.join(data_path, "expert_trajectories.pth")
    if not os.path.exists(load_path):
        print(f"错误：找不到专家轨迹 '{load_path}'。")
        print("请先运行 generate_expert_trajectories。")
        return model

    # [修复] 明确设置 weights_only=False 以加载包含 PyG Data 对象的 pickle 文件
    all_states, all_actions = torch.load(load_path, weights_only=False)
    print(f"加载了 {len(all_states)} 个 (s, a) 样本。")

    # 2. 创建 PyG DataLoader
    dataset = ExpertDataset(all_states, all_actions)

    # PyG 的 DataLoader 会自动处理图的批处理
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. 监督学习训练循环 [cite: 119]
    model.train()  # 切换到训练模式

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            # batch 是一个 PyG Batch 对象
            # expert_actions (batch.y) 形状是 [batch_size]

            # 4a. 前向传播
            # action_logits 形状是 [batch_size, num_total_actions]
            # (注意: 在我们的案例中, batch_size=N_steps, num_total_actions=N_edges)
            # 在我们的实现中，模型为 (task,proc) 边输出 logits
            # 而 HEFT 的动作是 (task, proc) 索引

            # ----
            # 关键修正: 我们的模型(model_hgat.py)输出 (N_edges) 个 logits
            # 我们的专家(HEFT)输出 (N_tasks * N_procs) 个索引
            # 幸运的是，'can_run_on' 边 (N_edges)
            # 恰好是 (N_tasks * N_procs)
            # 它们的扁平化索引是匹配的！
            # ----

            action_logits, _ = model(batch)  # [N_edges_in_batch, 1] -> [N_edges_in_batch]

            # 4b. 计算损失
            # action_logits: [TotalEdges], batch.y: [BatchSize]

            # [修复]
            # action_logits 是一个扁平化的张量 (batch_size * num_actions_per_step)
            # 形状是 [72]
            # batch.y 是 (batch_size)
            # 形状是 [6]
            # F.cross_entropy 期望 (N, C) 和 (N), 其中 N=batch_size, C=num_classes

            batch_size = batch.y.shape[0]  # N = 6

            # 检查是否能整除
            if action_logits.shape[0] % batch_size != 0:
                raise RuntimeError(f"Logits shape {action_logits.shape} 不能被 batch size {batch_size} 整除")

            num_classes = action_logits.shape[0] // batch_size  # C = 72 / 6 = 12

            # 将 logits 重塑为 (N, C)
            reshaped_logits = action_logits.view(batch_size, num_classes)  # 形状 [6, 12]

            # 现在形状匹配了: (N, C) 和 (N)
            loss = F.cross_entropy(reshaped_logits, batch.y)

            # 4c. 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"[BC] Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}")

    print("--- [阶段 1 BC] 预训练完成。 ---")
    # torch.save(model.state_dict(), "model_bc_pretrained.pth")
    return model  # 返回预训练好的模型


def train_ppo_finetuning(model: HGATActorCritic,
                         env: MCSSchedulingEnv,
                         agent: PPOAgent,
                         num_episodes: int = 2000,
                         max_steps_per_ep: int = 500) -> HGATActorCritic:
    """
    阶段2: PPO 微调 (热启动或冷启动)

    [此函数来自您之前的代码，未修改]
    ... (代码不变) ...
    """
    print(f"--- [阶段 2 PPO] 开始 PPO 微调 (共 {num_episodes} 回合)... ---")

    # 初始化 Rollout 缓冲区
    buffer = RolloutBuffer(agent.gamma, agent.gae_lambda)

    all_episode_rewards = []
    all_episode_makespans = []

    # 主训练循环
    for episode in tqdm(range(num_episodes)):

        # 1. 重置环境并获取初始状态
        state, info = env.reset()
        done = False
        truncated = False
        current_step = 0
        total_ep_reward = 0

        # 2. 收集轨迹 (Rollout)
        while not done and not truncated:

            # 2a. 获取动作屏蔽 [Sec 3.2]
            action_mask = env.get_action_mask()

            if not np.any(action_mask):
                truncated = True
                continue

                # 2b. 从智能体获取动作和价值 [Sec 3.1 HGAT]
            action_idx, log_prob, value = agent.get_action_and_value(state, action_mask)

            # 2c. 与环境交互 [Sec 3.3, 3.4]
            next_state, reward, done, truncated, info = env.step(action_idx)

            total_ep_reward += reward

            # 2d. 存储轨迹
            buffer.add(state, action_idx, log_prob, reward, value, done, action_mask)

            state = next_state
            current_step += 1
            if current_step >= max_steps_per_ep:
                truncated = True

        # 3. 回合结束 - 计算 GAE 和更新

        # 3a. 获取最后一步的价值
        if truncated:
            # 检查掩码是否全为False
            last_mask = env.get_action_mask()
            if not np.any(last_mask):
                last_value = 0.0  # 终端状态
            else:
                _, _, last_value = agent.get_action_and_value(state, last_mask)
            last_done = False
        else:
            last_value = 0.0
            last_done = True

        # 3b. 计算 GAE 和 Returns
        advantages, returns = buffer.compute_gae(last_value, last_done)
        buffer.advantages = advantages
        buffer.returns = returns

        # 3c. 更新 PPO 智能体
        agent.update(buffer)

        # 3d. 清空缓冲区
        buffer.clear()

        # 4. 记录
        all_episode_rewards.append(total_ep_reward)
        all_episode_makespans.append(info.get('current_time', 0))

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(all_episode_rewards[-100:])
            avg_makespan = np.mean(all_episode_makespans[-100:])
            print(f"\n[PPO] 回合 {episode + 1}/{num_episodes} | "
                  f"平均奖励: {avg_reward:.2f} | "
                  f"平均完工时间: {avg_makespan:.2f}s")

    print(f"--- [阶段 2 PPO] 训练完成。 ---")
    return model