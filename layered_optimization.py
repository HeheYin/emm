import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
from lightweight_modules import SpatioTemporalEmbedder, LightweightSetTransformer


class ResourcePredictor(nn.Module):
    """硬件资源状态预测子网络（LSTM）"""

    def __init__(self, input_dim=4, hidden_dim=32, pred_steps=50):
        super().__init__()
        self.input_dim = input_dim  # 添加这一行
        self.pred_steps = pred_steps  # 预测未来50ms的资源状态
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1
        )
        self.predictor = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, history_data):
        """
        history_data: 历史资源数据 (batch_size, seq_len, input_dim)
        return: 预测资源状态 (batch_size, pred_steps, input_dim)
        """
        batch_size = history_data.shape[0]

        # LSTM特征提取
        lstm_out, (hidden, cell) = self.lstm(history_data)
        lstm_out = self.layer_norm(lstm_out)

        # 使用最后一个时间步的隐藏状态
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)

        pred_outs = []
        current_hidden = last_hidden.unsqueeze(0)  # (1, batch_size, hidden_dim)
        current_cell = torch.zeros_like(current_hidden)

        # 初始输入是零
        current_input = torch.zeros(batch_size, 1, self.input_dim,
                                    device=history_data.device)

        for _ in range(self.pred_steps):
            # LSTM前向传播
            _, (current_hidden, current_cell) = self.lstm(
                current_input, (current_hidden, current_cell)
            )

            # 从隐藏状态预测输出
            pred = self.predictor(current_hidden.squeeze(0))  # (batch_size, input_dim)
            pred_outs.append(pred)

            # 使用预测结果作为下一个输入（自回归）
            current_input = pred.unsqueeze(1)  # (batch_size, 1, input_dim)

        # 堆叠预测结果
        pred_tensor = torch.stack(pred_outs, dim=1)  # (batch_size, pred_steps, input_dim)
        return pred_tensor


class ImprovedMultiObjectiveReward:
    """改进的多目标奖励函数"""

    @staticmethod
    def calculate_reward(
            current_makespan,
            load_states,
            hardware_load_threshold,
            energy_consumption,
            task_priority,
            task_deadline,
            actual_finish_time,
            task_type="传感器融合"
    ):
        """
        修复：使用config.py中的动态权重
        """
        # 1. Makespan奖励
        makespan_reward = 1.0 / (1.0 + current_makespan / 100.0)

        # 2. 截止时间奖励
        deadline_met = 1.0 if actual_finish_time <= task_deadline else -2.0
        deadline_reward = deadline_met

        # 3. 负载均衡奖励
        load_balance_reward = 1.0 - np.std(load_states)

        # 4. 能耗奖励
        energy_reward = 1.0 / (1.0 + energy_consumption / 500.0)

        # 5. 可靠性惩罚
        reliability_penalty = 0.0
        for i, (load, threshold) in enumerate(zip(load_states, hardware_load_threshold)):
            if load > threshold:
                overload_ratio = (load - threshold) / threshold
                penalty = 1.0 / (1.0 + np.exp(-overload_ratio * 3))
                reliability_penalty += penalty

        reliability_reward = 1.0 - reliability_penalty / len(load_states)

        # 6. 修复：使用config中的动态权重，修正键名
        weights = WEIGHTS[CURRENT_MODE]

        # 7. 优先级加权
        priority_weight = 0.5 + task_priority / 20.0

        # 8. 计算综合奖励 - 修正键名
        total_reward = (
                               makespan_reward * weights["makespan"] +
                               deadline_reward * weights["reliability"] +  # 使用reliability权重
                               load_balance_reward * weights["load"] +
                               energy_reward * weights["energy"] +
                               reliability_reward * weights["reliability"]  # 可靠性奖励也使用reliability权重
                       ) * priority_weight

        # 9. 奖励缩放和裁剪
        total_reward = total_reward * REWARD_SCALE
        total_reward = np.clip(total_reward, -1.0, 1.0)

        return float(total_reward)

class TaskMigrationStrategy:
    """任务迁移子策略（资源紧急调度）"""

    @staticmethod
    def calculate_migration_cost(from_hw, to_hw, task_data_size, current_load_states):
        # 1. 数据传输成本：数据量 / 带宽（假设带宽=100MB/s → 0.1MB/ms）
        bandwidth = 0.1  # MB/ms
        transfer_cost = task_data_size / bandwidth

        # 2. 通信延迟成本：硬件间通信延迟
        comm_delay = EMBEDDED_HARDWARES[HARDWARE_TYPES[from_hw]]["通信延迟"][HARDWARE_TYPES[to_hw]]

        # 3. 目标硬件负载成本：负载越高成本越高
        target_load = current_load_states[to_hw]
        load_cost = target_load * 10  # 负载0.9 → 9ms成本

        total_cost = transfer_cost + comm_delay + load_cost
        return total_cost

    @staticmethod
    def select_target_hardware(from_hw, task_data_size, current_load_states, hardware_thresholds):
        """选择最优目标硬件"""
        min_cost = float("inf")
        best_hw = from_hw  # 默认不迁移

        for to_hw in range(HARDWARE_NUM):
            if to_hw == from_hw:
                continue
            # 目标硬件负载必须低于阈值
            if current_load_states[to_hw] >= hardware_thresholds[to_hw]:
                continue
            # 计算迁移成本
            cost = TaskMigrationStrategy.calculate_migration_cost(
                from_hw, to_hw, task_data_size, current_load_states
            )
            if cost < min_cost:
                min_cost = cost
                best_hw = to_hw

        return best_hw, min_cost


class StableMODRLScheduler(nn.Module):
    def __init__(self, task_feat_dim, adj_dim, hardware_feat_dim):
        super().__init__()

        # 时空嵌入层
        self.st_embedder = SpatioTemporalEmbedder(task_feat_dim, adj_dim)

        # 硬件状态嵌入层
        self.hardware_embedder = LightweightSetTransformer(hardware_feat_dim)

        # 资源预测子网络
        self.resource_predictor = ResourcePredictor()

        # 修正输入维度
        input_dim = EMBED_DIM * 3 + 4

        # 更深的Q网络，添加Dropout防止过拟合
        self.q_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, HARDWARE_NUM)
        )

        # 目标网络
        self.target_q_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, HARDWARE_NUM)
        )

        # 初始化目标网络
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def prepare_state_features(self, task_feat, adj, seq_order, hardware_feat, load_states, history_resource):
        """准备状态特征"""
        batch_size, task_num = task_feat.shape[:2]

        # 1. 时空嵌入
        task_embeds, global_dag_embed = self.st_embedder(task_feat, adj, seq_order)

        # 2. 硬件状态嵌入
        hardware_embed = self.hardware_embedder(hardware_feat, load_states)

        # 3. 资源预测
        pred_resource = self.resource_predictor(history_resource)
        pred_current_resource = pred_resource[:, 0, :]

        # 4. 构建状态特征
        global_embed_expand = global_dag_embed.unsqueeze(1).repeat(1, task_num, 1)
        hardware_embed_expand = hardware_embed.unsqueeze(1).repeat(1, task_num, 1)
        pred_resource_expand = pred_current_resource.unsqueeze(1).repeat(1, task_num, 1)

        state_feat = torch.cat([
            task_embeds,
            global_embed_expand,
            hardware_embed_expand,
            pred_resource_expand
        ], dim=-1)

        return state_feat

    def forward(self, task_feat, adj, seq_order, hardware_feat, load_states, history_resource_data):
        state_feat = self.prepare_state_features(
            task_feat, adj, seq_order, hardware_feat, load_states, history_resource_data
        )
        q_values = self.q_network(state_feat)
        return q_values

    def soft_update_target(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    # 保持向后兼容


MODRLScheduler = StableMODRLScheduler
MultiObjectiveReward = ImprovedMultiObjectiveReward
