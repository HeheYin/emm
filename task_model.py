import networkx as nx
import numpy as np
import random
from config import *


class EmbeddedTaskNode:
    """嵌入式任务节点（扩展DAG节点属性）"""

    def __init__(self, task_id, task_type, comp_cost, hardware_tags, deadline, period=0):
        self.task_id = task_id  # 任务ID
        self.task_type = task_type  # 任务类型（工业控制/边缘AI/传感器融合）
        self.comp_cost = comp_cost  # 计算开销（不同硬件上的执行时间，字典格式）
        self.hardware_tags = hardware_tags  # 硬件适配标签（如["CPU", "GPU"]）
        self.deadline = deadline  # 截止时间（ms）
        self.period = period  # 周期（0表示非周期任务）
        self.priority = self._calc_priority()  # 任务优先级（基于截止时间和类型）
        self.is_preemptive = "工业控制" in task_type  # 工业控制任务支持抢占

    def _calc_priority(self):
        """计算任务优先级（1-10，数值越大优先级越高）"""
        if self.period > 0:
            # 周期任务：周期越短优先级越高
            period_score = 10 - (self.period - PERIOD_RANGE[0]) / (PERIOD_RANGE[1] - PERIOD_RANGE[0]) * 4
        else:
            # 非周期任务：截止时间越紧优先级越高
            period_score = 5
        # 任务类型权重
        type_score = {"工业控制": 10, "边缘AI": 7, "传感器融合": 5}[self.task_type]
        return (period_score * 0.4 + type_score * 0.6) / 2


class EmbeddedDAG:
    """嵌入式DAG任务模型（扩展原论文DAG）"""

    def __init__(self, dag_id, task_type):
        self.dag_id = dag_id
        self.task_type = task_type
        self.graph = nx.DiGraph()  # 有向无环图
        self.task_nodes = {}  # 任务节点字典：task_id -> EmbeddedTaskNode
        self.comp_matrix = None  # 计算开销矩阵 M_c (task_num × hardware_num)
        self.energy_matrix = None  # 能耗矩阵 M_e (task_num × hardware_num)
        self.comm_matrix = None  # 通信延迟矩阵 (task_num × task_num × hardware_num × hardware_num)
        self.task_num = 0

    def add_task(self, task_node):
        """添加任务节点"""
        self.task_nodes[task_node.task_id] = task_node
        self.graph.add_node(task_node.task_id, priority=task_node.priority)
        self.task_num += 1

    def add_dependency(self, from_task_id, to_task_id):
        """添加任务依赖（边）"""
        if from_task_id not in self.task_nodes or to_task_id not in self.task_nodes:
            raise ValueError("任务ID不存在")
        # 边权重：默认通信延迟（后续会根据硬件组合更新）
        self.graph.add_edge(from_task_id, to_task_id, weight=0)

    def build_matrices(self):
        """构建计算开销矩阵、能耗矩阵、通信延迟矩阵"""
        task_ids = sorted(self.task_nodes.keys())
        hardware_indices = {hw: i for i, hw in enumerate(HARDWARE_TYPES)}

        # 1. 计算开销矩阵 M_c (task_num × hardware_num)
        self.comp_matrix = np.zeros((self.task_num, HARDWARE_NUM))
        for ti, task_id in enumerate(task_ids):
            task = self.task_nodes[task_id]
            for hw, idx in hardware_indices.items():
                self.comp_matrix[ti, idx] = task.comp_cost.get(hw, float("inf"))

        # 2. 能耗矩阵 M_e (task_num × hardware_num)：单位时间能耗 × 执行时间
        self.energy_matrix = np.zeros((self.task_num, HARDWARE_NUM))
        for ti, task_id in enumerate(task_ids):
            task = self.task_nodes[task_id]
            for hw, idx in hardware_indices.items():
                if hw in task.comp_cost:
                    energy_per_ms = EMBEDDED_HARDWARES[hw]["能耗系数"]
                    self.energy_matrix[ti, idx] = energy_per_ms * task.comp_cost[hw]

        # 3. 通信延迟矩阵 (task_num × task_num × hardware_num × hardware_num)
        self.comm_matrix = np.zeros((self.task_num, self.task_num, HARDWARE_NUM, HARDWARE_NUM))
        for i, from_id in enumerate(task_ids):
            for j, to_id in enumerate(task_ids):
                if self.graph.has_edge(from_id, to_id):
                    for h1, idx1 in hardware_indices.items():
                        for h2, idx2 in hardware_indices.items():
                            self.comm_matrix[i, j, idx1, idx2] = EMBEDDED_HARDWARES[h1]["通信延迟"][h2]


class TaskSplitter:
    """软件层任务拆分预优化模块"""

    @staticmethod
    def calc_dependency_strength(dag):
        """计算任务间依赖强度（0-1，越大依赖越强）"""
        task_ids = sorted(dag.task_nodes.keys())
        dep_strength = np.zeros((dag.task_num, dag.task_num))

        for i, from_id in enumerate(task_ids):
            for j, to_id in enumerate(task_ids):
                if dag.graph.has_edge(from_id, to_id):
                    # 基于任务类型和计算开销计算依赖强度
                    from_task = dag.task_nodes[from_id]
                    to_task = dag.task_nodes[to_id]

                    # 类型一致性权重（同类型任务依赖更强）
                    type_weight = 1.0 if from_task.task_type == to_task.task_type else 0.5

                    # 计算开销比例权重（开销越接近依赖越强）
                    from_cost = np.mean([v for v in from_task.comp_cost.values() if v != float("inf")])
                    to_cost = np.mean([v for v in to_task.comp_cost.values() if v != float("inf")])
                    cost_weight = 1 - abs(from_cost - to_cost) / (from_cost + to_cost + 1e-6)

                    # 综合依赖强度
                    dep_strength[i, j] = (type_weight * 0.6 + cost_weight * 0.4)

        return dep_strength

    @staticmethod
    def split_tasks(dag, dep_threshold=0.7):
        """基于依赖强度拆分任务（修复循环依赖问题）"""
        task_ids = sorted(dag.task_nodes.keys())
        dep_strength = TaskSplitter.calc_dependency_strength(dag)

        # 步骤1：找到强依赖任务组（连通分量）
        task_groups = []
        visited = set()
        for i in range(dag.task_num):
            if i not in visited:
                group = [i]
                visited.add(i)
                # 广度优先搜索强依赖任务
                for j in range(dag.task_num):
                    if j not in visited and (
                            dep_strength[i, j] >= dep_threshold or dep_strength[j, i] >= dep_threshold):
                        group.append(j)
                        visited.add(j)
                task_groups.append(group)

        # 步骤2：构建拆分后的新DAG
        new_dag = EmbeddedDAG(dag.dag_id, dag.task_type)
        task_id_map = {}  # 原任务索引 -> 新任务ID
        new_task_id = 0

        for group in task_groups:
            # 合并强依赖任务为一个任务组节点
            group_tasks = [dag.task_nodes[task_ids[ti]] for ti in group]
            # 计算任务组的综合属性
            comp_cost = {}
            for hw in HARDWARE_TYPES:
                group_costs = [t.comp_cost.get(hw, float("inf")) for t in group_tasks]
                if any(c == float("inf") for c in group_costs):
                    comp_cost[hw] = float("inf")
                else:
                    comp_cost[hw] = max(group_costs)

            hardware_tags = list(set.intersection(*[set(t.hardware_tags) for t in group_tasks]))
            deadline = min(t.deadline for t in group_tasks)
            period = max(t.period for t in group_tasks)

            # 创建任务组节点
            group_node = EmbeddedTaskNode(
                task_id=new_task_id,
                task_type=dag.task_type,
                comp_cost=comp_cost,
                hardware_tags=hardware_tags,
                deadline=deadline,
                period=period
            )
            new_dag.add_task(group_node)
            for ti in group:
                task_id_map[ti] = new_task_id
            new_task_id += 1

        # 步骤3：安全地添加任务组间的依赖（避免循环）
        edge_added = set()  # 记录已添加的边，避免重复

        for i in range(dag.task_num):
            for j in range(dag.task_num):
                if dag.graph.has_edge(task_ids[i], task_ids[j]):
                    new_from_id = task_id_map[i]
                    new_to_id = task_id_map[j]

                    # 避免自环和重复边
                    if new_from_id == new_to_id:
                        continue  # 跳过自环

                    edge_key = (new_from_id, new_to_id)
                    if edge_key in edge_added:
                        continue  # 跳过重复边

                    # 检查添加这条边是否会产生环
                    new_dag.graph.add_edge(new_from_id, new_to_id)
                    try:
                        # 尝试拓扑排序，如果有环会抛出异常
                        list(nx.topological_sort(new_dag.graph))
                        edge_added.add(edge_key)  # 无环，保留这条边
                    except nx.NetworkXUnfeasible:
                        # 有环，移除这条边
                        new_dag.graph.remove_edge(new_from_id, new_to_id)
                        print(f"警告：跳过可能产生循环依赖的边 ({new_from_id} -> {new_to_id})")

        # 步骤4：验证新DAG的无环性
        try:
            list(nx.topological_sort(new_dag.graph))
            print(f"DAG {dag.dag_id} 拆分成功：原任务数 {dag.task_num} → 新任务数 {new_dag.task_num}")
        except nx.NetworkXUnfeasible:
            print(f"严重警告：DAG {dag.dag_id} 拆分后仍有循环依赖，创建链式结构")
            # 回退方案：创建简单的链式结构
            new_dag.graph.clear_edges()
            new_task_ids = sorted(new_dag.task_nodes.keys())
            for i in range(len(new_task_ids) - 1):
                new_dag.graph.add_edge(new_task_ids[i], new_task_ids[i + 1])

        # 步骤5：构建新DAG的矩阵和计算拆分增益
        new_dag.build_matrices()
        split_gain = TaskSplitter.evaluate_split_gain(dag, new_dag)

        return new_dag, split_gain

    @staticmethod
    def evaluate_split_gain(original_dag, split_dag):
        """评估拆分增益（>0表示拆分有效）"""
        # 1. 计算并行加速比提升
        original_max_cost = np.max(original_dag.comp_matrix[original_dag.comp_matrix != float("inf")])
        split_max_cost = np.max(split_dag.comp_matrix[split_dag.comp_matrix != float("inf")])
        speedup_gain = (original_max_cost - split_max_cost) / original_max_cost

        # 2. 计算通信开销变化
        original_comm_cost = np.sum(original_dag.comm_matrix[original_dag.comm_matrix > 0])
        split_comm_cost = np.sum(split_dag.comm_matrix[split_dag.comm_matrix > 0])
        comm_cost_change = (original_comm_cost - split_comm_cost) / original_comm_cost

        # 综合增益（权重：加速比0.7，通信开销0.3）
        total_gain = speedup_gain * 0.7 + comm_cost_change * 0.3
        return max(total_gain, 0)  # 增益不低于0