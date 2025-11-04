import torch
import numpy as np
from collections import deque
from config import *
from lightweight_modules import LightweightSetTransformer
from task_model import EmbeddedDAG, EmbeddedTaskNode


class DynamicTaskBuffer:
    """动态任务缓存队列（支持POMDP模型）"""

    # def __init__(self, buffer_size=100):
    #     self.buffer = deque(maxlen=buffer_size)
    #     self.arrival_rate_window = deque(maxlen=100)  # 滑动窗口统计到达率
    #     self.last_arrival_time = 0

    # def add_task(self, task_node, current_time):
    #     """添加动态任务"""
    #     # 计算到达间隔
    #     if self.last_arrival_time > 0:
    #         inter_arrival = current_time - self.last_arrival_time
    #         self.arrival_rate_window.append(1 / inter_arrival if inter_arrival > 0 else 0)
    #     self.last_arrival_time = current_time
    #     # 按优先级插入队列（高优先级在前）
    #     inserted = False
    #     for i in range(len(self.buffer)):
    #         if task_node.priority > self.buffer[i].priority:
    #             self.buffer.insert(i, task_node)
    #             inserted = True
    #             break
    #     if not inserted:
    #         self.buffer.append(task_node)
    def __init__(self, capacity=100):
        """初始化动态任务缓冲区"""
        self.buffer = deque(maxlen=capacity)  # 设置deque的最大长度
        self.capacity = capacity  # 保存容量属性

    def add_task(self, task_node, current_time):
        """添加动态任务到缓冲区"""
        task_node.arrival_time = current_time
        if len(self.buffer) >= self.capacity:
            # 如果缓冲区已满，移除最旧的任务（最左边的元素）
            self.buffer.popleft()
        # 添加新任务到缓冲区末尾
        self.buffer.append(task_node)

    def is_empty(self):
        """判断缓冲区是否为空"""
        return len(self.buffer) == 0

    def predict_arrival_rate(self):
        """预测任务到达率（滑动窗口平均）"""
        if len(self.arrival_rate_window) == 0:
            return DYNAMIC_ARRIVAL_RATE["平峰"]
        return np.mean(self.arrival_rate_window)

    def get_next_task(self):
        """获取下一个任务（非空则返回队首）"""
        return self.buffer.popleft() if self.buffer else None

    def get_buffer_state(self):
        """获取缓存队列状态（用于POMDP状态空间）"""
        # 队列长度、最高优先级、平均优先级
        if len(self.buffer) == 0:
            return np.array([0, 0, 0])
        priorities = [t.priority for t in self.buffer]
        return np.array([
            len(self.buffer) / self.buffer.maxlen,
            max(priorities) / 10.0,
            np.mean(priorities) / 10.0
        ])


class PartitionedReplayBuffer:
    """分区经验回放池（静态任务+动态任务）"""

    def __init__(self, static_capacity=5000, dynamic_capacity=5000):
        self.static_buffer = []
        self.dynamic_buffer = []
        self.static_capacity = static_capacity
        self.dynamic_capacity = dynamic_capacity
        self.priorities = {"static": [], "dynamic": []}

    def add_experience(self, experience, task_type="static"):
        """添加经验（static/dynamic）"""
        if task_type == "static":
            if len(self.static_buffer) >= self.static_capacity:
                self.static_buffer.pop(0)
                self.priorities["static"].pop(0)
            self.static_buffer.append(experience)
            self.priorities["static"].append(1.0)  # 初始优先级为1.0
        else:
            if len(self.dynamic_buffer) >= self.dynamic_capacity:
                self.dynamic_buffer.pop(0)
                self.priorities["dynamic"].pop(0)
            self.dynamic_buffer.append(experience)
            self.priorities["dynamic"].append(1.0)

    def sample_batch(self, batch_size=BATCH_SIZE, task_type="all"):
        """采样批次经验"""
        if task_type == "static":
            buffer = self.static_buffer
            priorities = self.priorities["static"]
        elif task_type == "dynamic":
            buffer = self.dynamic_buffer
            priorities = self.priorities["dynamic"]
        else:
            # 混合采样（各占一半）
            static_batch_size = batch_size // 2
            dynamic_batch_size = batch_size - static_batch_size
            static_samples = self._sample_from_buffer(self.static_buffer, self.priorities["static"], static_batch_size)
            dynamic_samples = self._sample_from_buffer(self.dynamic_buffer, self.priorities["dynamic"],
                                                       dynamic_batch_size)
            return static_samples + dynamic_samples

        return self._sample_from_buffer(buffer, priorities, batch_size)

    def _sample_from_buffer(self, buffer, priorities, batch_size):
        """从单个缓冲区采样"""
        if len(buffer) < batch_size:
            return buffer  # 不足时返回全部
        # 优先采样高优先级经验
        priorities = np.array(priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(buffer), size=batch_size, p=probs, replace=False)
        samples = [buffer[i] for i in indices]
        return samples

    def update_priorities(self, indices, priorities, task_type="static"):
        """更新经验优先级"""
        for i, idx in enumerate(indices):
            if task_type == "static" and idx < len(self.priorities["static"]):
                self.priorities["static"][idx] = priorities[i]
            elif task_type == "dynamic" and idx < len(self.priorities["dynamic"]):
                self.priorities["dynamic"][idx] = priorities[i]


class MultiSoftwareScheduler:
    """多软件并发调度器（软件优先级感知）"""

    def __init__(self, software_priorities):
        """
        software_priorities: 软件优先级字典 → {软件名称: 优先级权重（0-1）}
        """
        self.software_priorities = software_priorities
        self.software_dags = {}  # 软件DAG字典 → {软件名称: [DAG列表]}

    def register_software_dag(self, software_name, dag):
        """注册软件的DAG任务"""
        if software_name not in self.software_priorities:
            raise ValueError(f"软件 {software_name} 未配置优先级")
        if software_name not in self.software_dags:
            self.software_dags[software_name] = []
        self.software_dags[software_name].append(dag)

    def layered_set_embedding(self, hardware_feat, load_states):
        """分层集合嵌入（软件级→全局级）"""
        software_embeds = []
        hardware_embedder = LightweightSetTransformer(hardware_feat.shape[-1])

        # 1. 软件级嵌入：每个软件的任务集单独嵌入
        for software_name, dags in self.software_dags.items():
            # 聚合该软件的所有任务特征
            software_task_feats = []
            for dag in dags:
                # 简化：使用DAG的全局特征（实际应聚合所有任务特征）
                dag_feat = np.mean(dag.comp_matrix, axis=0)  # 任务平均计算开销
                software_task_feats.append(dag_feat)
            if not software_task_feats:
                continue
            software_task_feat = np.mean(software_task_feats, axis=0)  # 软件级任务特征
            # 硬件特征 + 软件优先级权重
            software_hardware_feat = np.hstack([
                hardware_feat,
                np.array([self.software_priorities[software_name]]).reshape(1, -1)
            ])
            # 软件级嵌入
            software_embed = hardware_embedder(
                torch.tensor(software_hardware_feat[None], dtype=torch.float32),
                torch.tensor(load_states[None], dtype=torch.float32)
            ).detach().numpy()
            software_embeds.append(software_embed)

        # 2. 全局级嵌入：聚合所有软件的嵌入
        if not software_embeds:
            return np.zeros((1, EMBED_DIM))
        global_embed = np.mean(software_embeds, axis=0)
        return global_embed

    def schedule_next_task(self, current_load_states):
        """选择下一个要调度的任务（软件优先级感知）"""
        best_task = None
        best_software = None
        best_hw = None
        min_cost = float("inf")

        # 遍历所有软件的任务
        for software_name, dags in self.software_dags.items():
            software_priority = self.software_priorities[software_name]
            for dag in dags:
                # 找到DAG中就绪的任务（所有前驱任务已完成）
                ready_tasks = [
                    task for task in dag.task_nodes.values()
                    if all(dag.graph.in_degree(task.task_id) == 0 or
                           all(pre_task in dag.task_nodes and dag.task_nodes[pre_task].is_completed
                               for pre_task in dag.graph.predecessors(task.task_id)))
                ]
                for task in ready_tasks:
                    # 计算该任务在各硬件上的调度成本
                    for hw_idx, hw in enumerate(HARDWARE_TYPES):
                        if task.comp_cost.get(hw, float("inf")) == float("inf"):
                            continue
                        # 调度成本 = 执行时间 + 负载成本 - 优先级奖励
                        exec_time = task.comp_cost[hw]
                        load_cost = current_load_states[hw_idx] * 10
                        priority_reward = software_priority * task.priority
                        total_cost = exec_time + load_cost - priority_reward

                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_task = task
                            best_software = software_name
                            best_hw = hw_idx

        return best_software, best_task, best_hw