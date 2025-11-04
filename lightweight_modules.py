import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *


class LightweightSGC(nn.Module):
    """轻量级简单图卷积（替换原论文GAT）"""

    def __init__(self, in_dim, out_dim, k=SGC_K):
        super().__init__()
        self.k = k
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.weight.weight)

    def forward(self, x, adj):
        """
        x: 节点特征 (batch_size, task_num, in_dim)
        adj: 邻接矩阵 (batch_size, task_num, task_num)
        return: 卷积后特征 (batch_size, task_num, out_dim)
        """
        # SGC：A^k * X * W
        batch_size = x.shape[0]
        # 归一化邻接矩阵
        adj = self.normalize_adj(adj)
        # 计算A^k
        adj_pow = adj
        for _ in range(self.k - 1):
            adj_pow = torch.bmm(adj_pow, adj)
        # 特征传播
        x = torch.bmm(adj_pow, x)
        # 线性变换
        x = self.weight(x)
        return x

    @staticmethod
    def normalize_adj(adj):
        """邻接矩阵归一化"""
        batch_size = adj.shape[0]
        for i in range(batch_size):
            # 添加自环
            adj[i] += torch.eye(adj.shape[1], device=adj.device)
            # 行归一化
            row_sum = adj[i].sum(dim=1, keepdim=True)
            adj[i] = adj[i] / (row_sum + 1e-6)
        return adj


class TemporalGRU(nn.Module):
    """时序GRU（替换原论文LSTM）"""

    def __init__(self, in_dim, hidden_dim=GRU_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, seq_order):
        """
        x: 节点特征 (batch_size, task_num, in_dim)
        seq_order: 任务优先级顺序 (batch_size, task_num) → 每个元素是原任务索引
        return: 时序嵌入特征 (batch_size, task_num, hidden_dim)
        """
        batch_size, task_num = seq_order.shape
        # 按优先级顺序重排特征
        ordered_x = torch.zeros_like(x, device=x.device)
        for b in range(batch_size):
            ordered_x[b] = x[b, seq_order[b]]
        # GRU前向传播
        out, _ = self.gru(ordered_x)
        # 层归一化
        out = self.layer_norm(out)
        # 恢复原顺序
        reversed_x = torch.zeros_like(out, device=x.device)
        for b in range(batch_size):
            reversed_x[b, seq_order[b]] = out[b]
        return reversed_x


class LightweightSetTransformer(nn.Module):
    """轻量化集合转换器（注意力剪枝+简化结构）"""

    def __init__(self, in_dim, out_dim=EMBED_DIM):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=2,  # 减少注意力头数
            batch_first=True,
            dropout=0.1
        )
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x, load_states):
        """
        x: 硬件特征 (batch_size, hardware_num, in_dim)
        load_states: 硬件负载状态 (batch_size, hardware_num) → 0-1
        return: 聚合后的集群特征 (batch_size, out_dim)
        """
        batch_size = x.shape[0]
        # 注意力剪枝：仅保留负载<阈值的硬件
        mask = (load_states >= LOAD_THRESHOLD).unsqueeze(1)  # (batch_size, 1, hardware_num)

        # 自注意力聚合
        attn_out, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask
        )

        # 全局池化（max pooling）
        global_feat = torch.max(attn_out, dim=1)[0]  # (batch_size, in_dim)
        # 线性变换
        out = self.linear(global_feat)
        return out


class SpatioTemporalEmbedder(nn.Module):
    """时空嵌入层（SGC+GRU）"""

    def __init__(self, task_feat_dim, adj_dim):
        super().__init__()
        self.sgc = LightweightSGC(task_feat_dim, EMBED_DIM)
        self.gru = TemporalGRU(EMBED_DIM)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, task_feat, adj, seq_order):
        """
        task_feat: 任务特征 (batch_size, task_num, task_feat_dim)
        adj: 邻接矩阵 (batch_size, task_num, task_num)
        seq_order: 任务优先级顺序 (batch_size, task_num)
        return: 任务嵌入 (batch_size, task_num, EMBED_DIM), 全局DAG嵌入 (batch_size, EMBED_DIM)
        """
        # 空间嵌入（SGC）
        spatial_embed = self.sgc(task_feat, adj)
        # 时序嵌入（GRU）
        spatio_temporal_embed = self.gru(spatial_embed, seq_order)
        # 全局DAG嵌入
        global_embed = self.global_pool(spatio_temporal_embed.transpose(1, 2)).squeeze(-1)
        return spatio_temporal_embed, global_embed