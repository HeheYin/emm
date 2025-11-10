from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F

"""
智能体的大脑：实现基于HGAT的编码器
以及 Actor (策略) 和 Critic (价值) 网络
"""


class HGATActorCritic(nn.Module):

    def __init__(self, hidden_dim: int, metadata: dict):
        super().__init__()

        # 1. 为每种节点类型定义输入线性层 (特征嵌入) [基于 表3.1]
        self.task_embed = Linear(6, hidden_dim)  # 6个任务特征: exec_cpu, exec_npu, crit, deadline, laxity, status
        self.proc_embed = Linear(4, hidden_dim)  # 4个处理器特征: type, load, queue_len, curr_crit

        # 2. 定义异构图卷积层 (HGTConv)
        # HGTConv 是一种强大的异构图神经网络层
        self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata, heads=4)
        self.conv2 = HGTConv(hidden_dim, hidden_dim, metadata, heads=4)

        # 3. 定义策略网络 (Actor)
        # 动作是 (Task_ID, Processor_ID) 的扁平化索引
        # 我们需要一种方法将 'task' 嵌入和 'proc' 嵌入结合起来
        # 我们将使用 'can_run_on' 边上的嵌入来计算 logits
        self.actor_mlp = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # 输入 (task_embed + proc_embed)
            nn.ReLU(),
            Linear(hidden_dim, 1)  # 输出该 (task, proc) 对的 "logit" 分数
        )

        # 4. 定义价值网络 (Critic)
        # 价值函数V(s)是关于整个状态(图)的
        # 我们使用所有任务节点的平均嵌入作为图的表示
        self.critic_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1)  # 输出 V(s)
        )

    def forward(self, hetero_data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        """

        # 1. 特征嵌入
        x_dict = {
            'task': self.task_embed(hetero_data['task'].x),
            'proc': self.proc_embed(hetero_data['proc'].x)
        }

        # 2. HGT 消息传递
        # 这使得代理能够 "看到" DAG拓扑和异构性
        x_dict = self.conv1(x_dict, hetero_data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, hetero_data.edge_index_dict)

        # 3. 计算 Actor Logits
        # 我们需要为每个 (task, proc) 对计算一个分数

        # 获取所有可能的 (task, proc) 边 [来自 'can_run_on' 边类型]
        task_idx, proc_idx = hetero_data['task', 'can_run_on', 'proc'].edge_index

        # 从 HGT 的输出中获取这些节点的嵌入
        task_embeds = x_dict['task'][task_idx]
        proc_embeds = x_dict['proc'][proc_idx]

        # 将任务和处理器嵌入拼接
        combined_embeds = torch.cat([task_embeds, proc_embeds], dim=1)

        # [N_edges, 1] -> N_edges 是 (num_tasks * num_procs)
        # actor_mlp 计算每个 (task, proc) 对的分数
        action_logits = self.actor_mlp(combined_embeds).squeeze(-1)

        # 4. 计算 Critic Value
        # 使用所有任务节点的平均嵌入作为图的表示
        mean_task_embed = x_dict['task'].mean(dim=0)
        value = self.critic_mlp(mean_task_embed)

        return action_logits, value