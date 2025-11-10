import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch, HeteroData
from typing import Optional, Tuple

from model_hgat import HGATActorCritic  # 依赖 model_hgat.py

"""
PPO 智能体，包含完整的更新逻辑和轨迹缓冲区
"""


class RolloutBuffer:
    """ 一个用于存储 PPO 轨迹的缓冲区 """

    def __init__(self, gamma: float, gae_lambda: float):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # 缓冲区列表
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = []  # 存储每一步的动作掩码

    def add(self, state: HeteroData, action: int, log_prob: float, reward: float, value: float, done: bool,
            mask: np.ndarray):
        """向缓冲区添加一个时间步的数据"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.masks.append(torch.tensor(mask, dtype=torch.bool))  # 将掩码转换为张量

    def compute_gae(self, last_value: float, last_done: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计 (GAE) 和回报 (Returns)
        """
        # 将列表转换为张量
        self.values = torch.tensor(self.values, dtype=torch.float).squeeze()
        self.rewards = torch.tensor(self.rewards, dtype=torch.float)
        self.dones = torch.tensor(self.dones, dtype=torch.float)

        advantages = torch.zeros_like(self.rewards)
        last_advantage = 0

        # 从后向前计算 GAE
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # GAE Delta: [r_t + gamma * V(s_{t+1}) * (1-d)] - V(s_t)
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]

            # GAE: delta_t + gamma * lambda * (1-d) * GAE_{t+1}
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            advantages[t] = last_advantage

        # Returns = Advantages + Values
        returns = advantages + self.values
        return advantages, returns

    def get_batches(self) -> Tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将整个缓冲区的数据作为一批 (Batch) 返回
        (在 GNN 训练中，一次性处理整个图序列通常更简单)
        """
        # 数据已在 compute_gae 中转换为张量
        actions_tensor = torch.tensor(self.actions, dtype=torch.long)
        log_probs_tensor = torch.tensor(self.log_probs, dtype=torch.float)
        masks_tensor = torch.stack(self.masks)  # (N_steps, N_actions)

        # PyG 图需要使用 Batch.from_data_list 进行批处理
        states_batch = Batch.from_data_list(self.states)

        return states_batch, actions_tensor, log_probs_tensor, self.advantages, self.returns, masks_tensor

    def clear(self):
        """清空缓冲区以便下一次 rollout"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = []


class PPOAgent:

    def __init__(self,
                 model: HGATActorCritic,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 ppo_epochs: int = 10,
                 entropy_coeff: float = 0.01,
                 value_loss_coeff: float = 0.5):

        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def get_action_and_value(self, state: HeteroData, action_mask: Optional[np.ndarray] = None) -> Tuple[
        int, float, float]:
        """
        从模型获取动作、对数值和价值
        state: 当前的 PyG 异构图状态
        action_mask: 来自环境的布尔掩码
        """

        # 确保模型处于评估模式 (用于采样)
        self.model.eval()
        with torch.no_grad():
            action_logits, value = self.model(state)

        if action_mask is not None:
            # 关键：应用动作屏蔽 [Sec 3.2]
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool)

            # 确保掩码和 logits 形状匹配
            if mask_tensor.shape[0] != action_logits.shape[0]:
                print(f"警告: 掩码形状 ({mask_tensor.shape}) 与 Logits形状 ({action_logits.shape}) 不匹配")
                # 这种情况下，我们假设没有有效动作
                mask_tensor = torch.zeros_like(action_logits, dtype=torch.bool)

            action_logits[~mask_tensor] = -torch.inf  # 将无效动作的概率设为负无穷

        # 从 logits 创建概率分布
        probs = Categorical(logits=action_logits)
        action = probs.sample()  # 采样动作
        log_prob = probs.log_prob(action)  # 获取该动作的 log_prob

        return action.item(), log_prob.item(), value.item()

    def update(self, buffer: RolloutBuffer):
        """
        PPO 核心更新逻辑。
        使用收集到的轨迹数据，更新 Actor 和 Critic 网络。
        """
        # 确保模型处于训练模式
        self.model.train()

        # 1. 计算 GAE 和 Returns
        # (假设 buffer 已经在外部被告知 last_value 和 last_done)
        advantages, returns = buffer.advantages, buffer.returns

        # 标准化优势 (可选，但强烈推荐)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. 获取批处理数据
        states_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch, masks_batch = buffer.get_batches()

        # 3. PPO 优化循环
        for _ in range(self.ppo_epochs):
            # 4. 重新评估旧数据
            # (在 PyG Batch 对象上评估)
            new_logits, new_values = self.model(states_batch)
            new_values = new_values.squeeze()

            # 应用掩码 [关键]
            new_logits[~masks_batch.flatten()] = -torch.inf

            new_dist = Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(actions_batch)
            entropy = new_dist.entropy().mean()  # 计算熵以鼓励探索

            # 5. 计算 Actor (Policy) 损失 (Clipped Objective)
            ratio = (new_log_probs - old_log_probs_batch).exp()

            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch

            policy_loss = -torch.min(surr1, surr2).mean()

            # 6. 计算 Critic (Value) 损失 (MSE)
            value_loss = F.mse_loss(new_values, returns_batch)

            # 7. 计算总损失
            loss = (policy_loss +
                    self.value_loss_coeff * value_loss -
                    self.entropy_coeff * entropy)

            # 8. 优化
            self.optimizer.zero_grad()
            loss.backward()
            # (可选) 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()