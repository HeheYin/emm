import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *


class GAT(nn.Module):
    """图注意力网络（替换LightweightSGC）"""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        # 多头注意力机制
        self.attention_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim // num_heads, bias=False) for _ in range(num_heads)
        ])

        # 注意力系数计算
        self.attention_coeff = nn.ModuleList([
            nn.Linear(2 * (out_dim // num_heads), 1, bias=False) for _ in range(num_heads)
        ])

        # 输出层
        self.output_layer = nn.Linear(out_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adj):
        """
        x: 节点特征 (batch_size, task_num, in_dim)
        adj: 邻接矩阵 (batch_size, task_num, task_num)
        return: 注意力聚合后特征 (batch_size, task_num, out_dim)
        """
        batch_size, task_num, in_dim = x.shape
        out_dim = self.attention_layers[0].out_features * self.num_heads

        # 存储所有头的输出
        head_outputs = []

        for head in range(self.num_heads):
            # 线性变换节点特征
            Wh = self.attention_layers[head](x)  # (batch_size, task_num, out_dim//num_heads)

            # 计算注意力系数
            attention_scores = self._compute_attention_scores(Wh, adj)

            # 应用softmax获取注意力权重
            attention_weights = self.softmax(attention_scores)
            attention_weights = F.dropout(attention_weights, self.dropout, training=self.training)

            # 聚合邻居节点信息
            head_output = torch.bmm(attention_weights, Wh)  # (batch_size, task_num, out_dim//num_heads)
            head_outputs.append(head_output)

        # 拼接多头输出
        multi_head_output = torch.cat(head_outputs, dim=-1)  # (batch_size, task_num, out_dim)

        # 最终线性变换
        output = self.output_layer(multi_head_output)
        return output

    def _compute_attention_scores(self, Wh, adj):
        """
        计算注意力系数
        Wh: 变换后的节点特征 (batch_size, task_num, out_dim//num_heads)
        adj: 邻接矩阵 (batch_size, task_num, task_num)
        return: 注意力系数 (batch_size, task_num, task_num)
        """
        batch_size, task_num, feat_dim = Wh.shape

        # 计算每对节点间的注意力分数
        # 使用广播机制计算 [Wh_i || Wh_j] 矩阵
        Wh_expanded_i = Wh.unsqueeze(2).repeat(1, 1, task_num, 1)  # (batch_size, task_num, task_num, feat_dim)
        Wh_expanded_j = Wh.unsqueeze(1).repeat(1, task_num, 1, 1)  # (batch_size, task_num, task_num, feat_dim)

        # 拼接特征
        concat_features = torch.cat([Wh_expanded_i, Wh_expanded_j],
                                    dim=-1)  # (batch_size, task_num, task_num, 2*feat_dim)

        # 计算注意力系数
        head_idx = 0  # 假设使用第一个注意力头的系数计算层
        attention_scores = self.attention_coeff[head_idx](concat_features).squeeze(
            -1)  # (batch_size, task_num, task_num)

        # 应用LeakyReLU
        attention_scores = self.leaky_relu(attention_scores)

        # 根据邻接矩阵屏蔽不相连的节点
        mask = (adj == 0)  # 邻接矩阵为0的位置需要屏蔽
        attention_scores.masked_fill_(mask, float('-inf'))

        return attention_scores


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
        # 检查seq_order中的索引是否超出范围，并进行修正
        seq_order = torch.clamp(seq_order, 0, task_num - 1)

        # 按优先级顺序重排特征
        ordered_x = torch.zeros_like(x, device=x.device)
        for b in range(batch_size):
            # 使用高级索引正确重排
            ordered_x[b] = x[b][seq_order[b]]

        # GRU前向传播
        out, _ = self.gru(ordered_x)
        # 层归一化
        out = self.layer_norm(out)
        # 恢复原顺序
        reversed_x = torch.zeros_like(out, device=x.device)
        for b in range(batch_size):
            # 使用 inverse indexing 恢复原顺序
            reversed_x[b][seq_order[b]] = out[b]

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
        # 注意力剪枝：仅保留负载<阈值的硬件
        mask = (load_states >= LOAD_THRESHOLD)  # (batch_size, hardware_num) - 移除了unsqueeze(1)

        # 自注意力聚合
        attn_out, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask  # 直接使用2D mask
        )

        # 全局池化（max pooling）
        global_feat = torch.max(attn_out, dim=1)[0]  # (batch_size, in_dim)
        # 线性变换
        out = self.linear(global_feat)
        return out


class SpatioTemporalEmbedder(nn.Module):
    """时空嵌入层（GAT+GRU）"""

    def __init__(self, task_feat_dim, adj_dim):
        super().__init__()
        # 使用GAT替换SGC
        self.gat = GAT(task_feat_dim, EMBED_DIM)
        self.gru = TemporalGRU(EMBED_DIM)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, task_feat, adj, seq_order):
        """
        task_feat: 任务特征 (batch_size, task_num, task_feat_dim)
        adj: 邻接矩阵 (batch_size, task_num, task_num)
        seq_order: 任务优先级顺序 (batch_size, task_num)
        return: 任务嵌入 (batch_size, task_num, EMBED_DIM), 全局DAG嵌入 (batch_size, EMBED_DIM)
        """
        # 空间嵌入（GAT）
        spatial_embed = self.gat(task_feat, adj)
        # 时序嵌入（GRU）
        spatio_temporal_embed = self.gru(spatial_embed, seq_order)
        # 全局DAG嵌入
        global_embed = self.global_pool(spatio_temporal_embed.transpose(1, 2)).squeeze(-1)
        return spatio_temporal_embed, global_embed
