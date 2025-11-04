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
        self.pred_steps = pred_steps  # 预测未来50ms的资源状态
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1
        )
        self.predictor = nn.Linear(hidden_dim, input_dim)

    def forward(self, history_data):
        """
        history_data: 历史资源数据 (batch_size, seq_len, input_dim)
                     input_dim: [CPU负载, GPU负载, FPGA负载, MCU负载]
        return: 预测资源状态 (batch_size, pred_steps, input_dim)
        """
        # LSTM特征提取
        lstm_out, (hidden, _) = self.lstm(history_data)
        # 预测未来状态（用最后一个时间步的隐藏状态预测所有未来步）
        hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        pred_outs = []

        # 初始化隐藏状态和细胞状态
        hx = hidden.unsqueeze(0)  # (1, batch_size, hidden_dim)
        cx = torch.zeros_like(hx)  # (1, batch_size, hidden_dim)

        for _ in range(self.pred_steps):
            # 预测当前时间步
            pred = self.predictor(hidden)  # (batch_size, input_dim)
            pred_outs.append(pred)
            # 用预测结果更新隐藏状态（自回归）
            pred_expand = pred.unsqueeze(1)  # (batch_size, 1, input_dim)
            _, (hx, cx) = self.lstm(pred_expand, (hx, cx))
            hidden = hx.squeeze(0)  # 更新hidden用于下一次预测

        return torch.stack(pred_outs, dim=1)  # (batch_size, pred_steps, input_dim)



# layered_optimization.py 中的 MultiObjectiveReward 类修改
class MultiObjectiveReward:
    """多目标奖励函数（Makespan+负载+能耗+可靠性）"""

    @staticmethod
    def get_adaptive_weights(task_type, current_load_states, hardware_types):
        """
        根据任务类型和当前硬件状态动态调整权重
        task_type: 任务类型
        current_load_states: 当前硬件负载状态
        hardware_types: 硬件类型列表
        return: 动态权重字典
        """
        # 基础权重配置
        base_weights = {
            "工业控制": {"makespan": 0.5, "load": 0.1, "energy": 0.1, "reliability": 0.3},
            "边缘AI": {"makespan": 0.2, "load": 0.2, "energy": 0.4, "reliability": 0.2},
            "传感器融合": {"makespan": 0.3, "load": 0.2, "energy": 0.3, "reliability": 0.2}
        }

        # 根据任务类型获取基础权重
        weights = base_weights.get(task_type, base_weights["传感器融合"]).copy()

        # 根据硬件负载状态调整权重
        avg_load = np.mean(current_load_states)
        if avg_load > 0.8:  # 高负载时更关注可靠性
            weights["reliability"] += 0.1
            weights["makespan"] -= 0.05
            weights["energy"] -= 0.05

        # 根据硬件类型调整能耗权重
        # DSP通常用于低功耗场景，NPU用于高性能场景
        dsp_load = current_load_states[hardware_types.index("DSP")] if "DSP" in hardware_types else 0
        npu_load = current_load_states[hardware_types.index("NPU")] if "NPU" in hardware_types else 0

        if dsp_load > 0.7:  # DSP高负载时更关注能耗
            weights["energy"] += 0.1
            weights["makespan"] -= 0.05

        if npu_load > 0.8:  # NPU高负载时更关注可靠性
            weights["reliability"] += 0.1
            weights["energy"] -= 0.05

        # 归一化权重
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total

        return weights

    @staticmethod
    def calculate_reward(
            current_makespan, last_makespan,
            load_states, hardware_load_threshold,
            energy_consumption, last_energy,
            task_priority, task_type="传感器融合"
    ):
        """
        current_makespan: 当前总执行时间
        last_makespan: 上一轮总执行时间
        load_states: 各硬件负载 (hardware_num,)
        energy_consumption: 当前总能耗
        last_energy: 上一轮总能耗
        task_priority: 当前任务优先级
        task_type: 任务类型，用于动态调整权重
        return: 综合奖励值
        """
        # 获取动态权重
        weights = MultiObjectiveReward.get_adaptive_weights(task_type, load_states, HARDWARE_TYPES)

        # 1. Makespan奖励：使用相对改进而不是绝对差值
        if last_makespan > 1e-6:  # 避免除零错误
            makespan_improvement = (last_makespan - current_makespan) / last_makespan
        else:
            # 第一次计算时，使用绝对值的负数作为惩罚
            makespan_improvement = -current_makespan / 100.0  # 标准化

        makespan_reward = makespan_improvement

        # 2. 负载均衡奖励：负载方差越小奖励越高
        load_var = np.var(load_states)
        load_reward = -load_var

        # 3. 能耗奖励：使用相对改进而不是绝对差值
        if last_energy > 1e-6:  # 避免除零错误
            energy_improvement = (last_energy - energy_consumption) / last_energy
        else:
            # 第一次计算时，使用绝对值的负数作为惩罚
            energy_improvement = -energy_consumption / 100.0  # 标准化

        energy_reward = energy_improvement

        # 4. 可靠性奖励：超过负载阈值惩罚，结合硬件特性
        reliability_reward = 0.0
        for i, (load, threshold) in enumerate(zip(load_states, hardware_load_threshold)):
            hw_type = HARDWARE_TYPES[i]
            if load > threshold:
                # 根据硬件类型调整惩罚系数
                penalty_factor = 1.0
                if hw_type == "DSP":  # DSP对过载更敏感
                    penalty_factor = 1.5
                elif hw_type == "NPU":  # NPU对过载容忍度较高
                    penalty_factor = 0.8
                reliability_reward -= (load - threshold) * penalty_factor

        # 5. 任务优先级加权
        priority_weight = (task_priority / 10.0)  # 1-10 → 0.1-1.0

        # 综合奖励
        total_reward = (
                               makespan_reward * weights["makespan"] +
                               load_reward * weights["load"] +
                               energy_reward * weights["energy"] +
                               reliability_reward * weights["reliability"]
                       ) * priority_weight

        return total_reward



# class MultiObjectiveReward:
#     """多目标奖励函数（Makespan+负载+能耗+可靠性）"""
#
#     @staticmethod
#     def calculate_reward(
#             current_makespan, last_makespan,
#             load_states, hardware_load_threshold,
#             energy_consumption, last_energy,
#             task_priority
#     ):
#         """
#         current_makespan: 当前总执行时间
#         last_makespan: 上一轮总执行时间
#         load_states: 各硬件负载 (hardware_num,)
#         energy_consumption: 当前总能耗
#         last_energy: 上一轮总能耗
#         task_priority: 当前任务优先级
#         return: 综合奖励值
#         """
#         weights = WEIGHTS[CURRENT_MODE]
#
#         # 1. Makespan奖励：减少量越大奖励越高
#         makespan_reward = -(current_makespan - last_makespan) / (last_makespan + 1e-6)
#
#         # 2. 负载均衡奖励：负载方差越小奖励越高
#         load_var = np.var(load_states)
#         load_reward = -load_var
#
#         # 3. 能耗奖励：能耗增量越小奖励越高
#         energy_increment = energy_consumption - last_energy
#         energy_reward = -energy_increment / (last_energy + 1e-6)
#
#         # 4. 可靠性奖励：超过负载阈值惩罚
#         reliability_reward = 0.0
#         for load, threshold in zip(load_states, hardware_load_threshold):
#             if load > threshold:
#                 reliability_reward -= (load - threshold)
#
#         # 5. 任务优先级加权
#         priority_weight = (task_priority / 10.0)  # 1-10 → 0.1-1.0
#
#         # 综合奖励
#         total_reward = (
#                                makespan_reward * weights["makespan"] +
#                                load_reward * weights["load"] +
#                                energy_reward * weights["energy"] +
#                                reliability_reward * weights["reliability"]
#                        ) * priority_weight
#
#         return total_reward
#

class TaskMigrationStrategy:
    """任务迁移子策略（资源紧急调度）"""

    @staticmethod
    def calculate_migration_cost(from_hw, to_hw, task_data_size, current_load_states):
        """
        from_hw: 源硬件索引
        to_hw: 目标硬件索引
        task_data_size: 任务数据量（MB）
        current_load_states: 当前硬件负载状态
        return: 迁移成本（ms）
        """
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


class MODRLScheduler(nn.Module):
    """嵌入式轻量化MODRL调度器"""

    # 修改 MODRLScheduler.__init__ 方法中的 q_network 和 target_q_network 定义
    def __init__(self, task_feat_dim, adj_dim, hardware_feat_dim):
        super().__init__()
        # 时空嵌入层
        self.st_embedder = SpatioTemporalEmbedder(task_feat_dim, adj_dim)
        # 硬件状态嵌入层
        self.hardware_embedder = LightweightSetTransformer(hardware_feat_dim)
        # 资源预测子网络
        self.resource_predictor = ResourcePredictor()
        # 调度决策网络（D3QN）
        # 修正输入维度：task_embed(EMBED_DIM) + global_dag_embed(EMBED_DIM) +
        # hardware_embed(EMBED_DIM) + pred_resource(4) = 64 + 64 + 64 + 4 = 196
        self.q_network = nn.Sequential(
            nn.Linear(EMBED_DIM * 3 + 4, 128),  # 修改此处
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, HARDWARE_NUM)  # 输出每个硬件的Q值
        )
        # 目标网络
        self.target_q_network = nn.Sequential(
            nn.Linear(EMBED_DIM * 3 + 4, 128),  # 修改此处
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, HARDWARE_NUM)
        )
        # 初始化目标网络参数
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

    def forward(self, task_feat, adj, seq_order, hardware_feat, load_states, history_resource_data):
        """
        task_feat: 任务特征 (batch_size, task_num, task_feat_dim)
        adj: 邻接矩阵 (batch_size, task_num, task_num)
        seq_order: 任务优先级顺序 (batch_size, task_num)
        hardware_feat: 硬件特征 (batch_size, hardware_num, hardware_feat_dim)
        load_states: 硬件负载状态 (batch_size, hardware_num)
        history_resource_data: 历史资源数据 (batch_size, seq_len, 4)
        return: Q值 (batch_size, task_num, hardware_num)
        """
        batch_size, task_num = task_feat.shape[:2]

        # 1. 时空嵌入（任务特征）
        task_embeds, global_dag_embed = self.st_embedder(task_feat, adj, seq_order)

        # 2. 硬件状态嵌入
        hardware_embed = self.hardware_embedder(hardware_feat, load_states)

        # 3. 资源预测
        pred_resource = self.resource_predictor(history_resource_data)
        # 取预测的第一个时间步作为当前资源状态补充
        pred_current_resource = pred_resource[:, 0, :]  # (batch_size, 4)

        # 4. 构建状态特征（任务嵌入 + 全局DAG嵌入 + 硬件嵌入 + 资源预测）
        global_embed_expand = global_dag_embed.unsqueeze(1).repeat(1, task_num, 1)  # (batch_size, task_num, EMBED_DIM)
        hardware_embed_expand = hardware_embed.unsqueeze(1).repeat(1, task_num, 1)  # (batch_size, task_num, EMBED_DIM)
        pred_resource_expand = pred_current_resource.unsqueeze(1).repeat(1, task_num, 1)  # (batch_size, task_num, 4)

        state_feat = torch.cat([
            task_embeds,
            global_embed_expand,
            hardware_embed_expand,
            pred_resource_expand
        ], dim=-1)  # (batch_size, task_num, EMBED_DIM*2 + EMBED_DIM +4)

        # 5. 计算Q值
        q_values = self.q_network(state_feat)
        return q_values

    def soft_update_target(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)