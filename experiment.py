import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from config import *
from task_model import EmbeddedDAG, EmbeddedTaskNode, TaskSplitter
from layered_optimization import MODRLScheduler, MultiObjectiveReward
from special_scenarios import DynamicTaskBuffer, MultiSoftwareScheduler


class DetailedSchedulerVisualizer:
    """详细调度可视化器"""

    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, HARDWARE_NUM))
        self.hardware_names = HARDWARE_TYPES

    def visualize_single_dag_schedule(self, dag, schedule_result, algorithm_name):
        """可视化单个DAG的详细调度过程"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # 1. DAG结构图
        self._plot_dag_structure(ax1, dag, "DAG任务依赖结构")

        # 2. 调度甘特图
        self._plot_schedule_gantt(ax2, schedule_result, algorithm_name)

        # 3. 硬件负载图
        self._plot_hardware_load(ax3, schedule_result)

        plt.tight_layout()
        plt.savefig(f"detailed_schedule_{algorithm_name}_dag{dag.dag_id}.png", dpi=300, bbox_inches='tight')
        plt.show()

    # In the DetailedSchedulerVisualizer class methods:

    def _plot_dag_structure(self, ax, dag, title):
        """Plot DAG structure with English labels"""
        pos = nx.spring_layout(dag.graph, seed=42)
        nx.draw(dag.graph, pos, ax=ax, with_labels=True,
                node_color='lightblue', node_size=500,
                font_size=8, font_weight='bold',
                arrows=True, arrowsize=20)

        # Add task information labels (using English)
        labels = {}
        for task_id, task in dag.task_nodes.items():
            labels[task_id] = f"T{task_id}\nP:{task.priority:.1f}\nD:{task.deadline:.0f}"

        nx.draw_networkx_labels(dag.graph, pos, labels, ax=ax, font_size=6)
        ax.set_title(title)  # Keep title as is since it's passed as parameter

    def _plot_schedule_gantt(self, ax, schedule_result, algorithm_name):
        """Plot scheduling Gantt chart with English labels"""
        task_schedule = schedule_result["task_schedule"]
        hardware_usage = schedule_result["hardware_usage"]

        # Create timeline for each hardware
        for hw_idx, hw_name in enumerate(self.hardware_names):
            hw_tasks = [(task_id, start, finish)
                        for task_id, (hw, start, finish) in task_schedule.items()
                        if hw == hw_idx]

            for i, (task_id, start, finish) in enumerate(hw_tasks):
                ax.barh(hw_idx, finish - start, left=start,
                        color=self.colors[task_id % len(self.colors)],
                        alpha=0.7, edgecolor='black')
                ax.text(start + (finish - start) / 2, hw_idx,
                        f"T{task_id}", ha='center', va='center', fontsize=8)

        ax.set_yticks(range(len(self.hardware_names)))
        ax.set_yticklabels(self.hardware_names)
        ax.set_xlabel("Time (ms)")  # Changed from Chinese to English
        ax.set_title(f"{algorithm_name} - Task Scheduling Gantt Chart")  # English title
        ax.grid(True, alpha=0.3)

    def _plot_hardware_load(self, ax, schedule_result):
        """Plot hardware load distribution with English labels"""
        hardware_usage = schedule_result["hardware_usage"]
        hardware_thresholds = [EMBEDDED_HARDWARES[hw]["负载阈值"] for hw in HARDWARE_TYPES]

        x = range(len(self.hardware_names))
        usage_times = [usage["total_time"] for usage in hardware_usage.values()]

        bars = ax.bar(x, usage_times, color=self.colors, alpha=0.7)
        ax.axhline(y=schedule_result["makespan"], color='red', linestyle='--',
                   label=f'Makespan: {schedule_result["makespan"]:.2f}ms')

        # Add threshold lines
        for i, threshold in enumerate(hardware_thresholds):
            ax.axhline(y=threshold * schedule_result["makespan"],
                       color='red', alpha=0.3, linestyle=':')

        ax.set_xticks(x)
        ax.set_xticklabels(self.hardware_names, rotation=45)
        ax.set_ylabel("Execution Time (ms)")  # English label
        ax.set_title("Hardware Load Distribution")  # English title
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    def compare_algorithms_on_dag(self, dag, algorithm_results):
        """比较不同算法在同一个DAG上的表现"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        algorithms = list(algorithm_results.keys())

        # 1. Makespan对比
        makespans = [result["makespan"] for result in algorithm_results.values()]
        axes[0].bar(algorithms, makespans, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0].set_title("各算法Makespan对比")
        axes[0].set_ylabel("Makespan (ms)")

        # 2. 能耗对比
        energies = [result["total_energy"] for result in algorithm_results.values()]
        axes[1].bar(algorithms, energies, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[1].set_title("各算法能耗对比")
        axes[1].set_ylabel("能耗 (J)")

        # 3. 截止时间满足率
        deadline_rates = [result["deadline_satisfaction_rate"] for result in algorithm_results.values()]
        axes[2].bar(algorithms, deadline_rates, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[2].set_title("截止时间满足率")
        axes[2].set_ylabel("满足率")
        axes[2].set_ylim(0, 1)

        # 4. 负载均衡对比
        load_balances = [result["load_balance"] for result in algorithm_results.values()]
        axes[3].bar(algorithms, load_balances, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[3].set_title("负载均衡（方差）")
        axes[3].set_ylabel("方差")

        plt.tight_layout()
        plt.savefig(f"algorithm_comparison_dag{dag.dag_id}.png", dpi=300, bbox_inches='tight')
        plt.show()

        # 输出详细数据
        print(f"\n=== DAG {dag.dag_id} ({dag.task_type}) 算法对比 ===")
        for algo, result in algorithm_results.items():
            print(f"{algo}:")
            print(f"  - Makespan: {result['makespan']:.2f}ms")
            print(f"  - 总能耗: {result['total_energy']:.2f}J")
            print(f"  - 截止时间满足率: {result['deadline_satisfaction_rate']:.2%}")
            print(f"  - 负载均衡方差: {result['load_balance']:.4f}")


class EmbeddedDatasetGenerator:
    """嵌入式任务数据集生成器"""

    @staticmethod
    def generate_task_comp_cost(task_type, hw):
        """生成任务在指定硬件上的计算开销"""
        base_cost = {
            "工业控制": {"Cortex-A78": 20, "Mali-G78": 15, "NPU": 25, "DSP": 18},
            "边缘AI": {"Cortex-A78": 50, "Mali-G78": 30, "NPU": 20, "DSP": 40},
            "传感器融合": {"Cortex-A78": 30, "Mali-G78": 25, "NPU": 35, "DSP": 22}
        }[task_type][hw]

        # 添加随机扰动
        return base_cost * random.uniform(0.8, 1.2)

    @staticmethod
    def generate_hardware_tags(task_type):
        """生成任务的硬件适配标签"""
        tags = {
            "工业控制": ["CPU", "FPGA", "MCU"],
            "边缘AI": ["CPU", "GPU", "FPGA"],
            "传感器融合": ["CPU", "GPU", "FPGA", "MCU"]
        }[task_type]
        # 随机剔除1个非必需硬件（保证至少1个标签）
        if len(tags) > 1:
            tags.pop(random.randint(0, len(tags) - 1))
        return tags

    # 在 experiment.py 文件中，找到 generate_dag 方法，替换为以下代码：
    @staticmethod
    def generate_dag(task_type, dag_id):
        """生成单个DAG任务（改进版，确保清晰的层次结构）"""
        dag = EmbeddedDAG(dag_id, task_type)
        task_num = random.randint(8, MAX_TASK_NUM)  # 增加最小任务数，确保有足够层次

        # 生成任务节点
        for task_id in range(task_num):
            # 计算开销（各硬件）
            comp_cost = {}
            for hw in HARDWARE_TYPES:
                comp_cost[hw] = EmbeddedDatasetGenerator.generate_task_comp_cost(task_type, hw)
            # 硬件标签
            hardware_tags = EmbeddedDatasetGenerator.generate_hardware_tags(task_type)
            # 截止时间（基于总计算量的1.5倍）
            total_comp_cost = np.sum([v for v in comp_cost.values() if v != float("inf")])
            deadline = total_comp_cost * DEADLINE_FACTOR
            # 周期（仅30%的任务是周期任务）
            period = random.randint(*PERIOD_RANGE) if random.random() < 0.3 else 0
            # 创建任务节点
            task_node = EmbeddedTaskNode(
                task_id=task_id,
                task_type=task_type,
                comp_cost=comp_cost,
                hardware_tags=hardware_tags,
                deadline=deadline,
                period=period
            )
            dag.add_task(task_node)

        # === 改进的层级结构生成逻辑 ===
        task_ids = list(range(task_num))

        # 确定层数（3-6层，确保有清晰的层次）
        num_layers = min(random.randint(3, min(6, task_num // 3 + 2)), task_num)

        # 将任务分配到各层 - 确保第一层只有1个起始节点
        layers = [[] for _ in range(num_layers)]

        # 第一层：只有1个起始节点
        layers[0] = [task_ids[0]]
        remaining_tasks = task_ids[1:]
        random.shuffle(remaining_tasks)

        # 中间层分配（确保每层都有合理的任务数）
        for layer_idx in range(1, num_layers - 1):
            if not remaining_tasks:
                break

            # 计算该层应有的任务数（逐渐增加）
            base_tasks = max(1, len(remaining_tasks) // (num_layers - layer_idx))
            # 添加一些随机性，但避免层间任务数差异过大
            tasks_in_layer = random.randint(
                max(1, base_tasks - 1),
                min(base_tasks + 2, len(remaining_tasks))
            )

            layers[layer_idx] = remaining_tasks[:tasks_in_layer]
            remaining_tasks = remaining_tasks[tasks_in_layer:]

        # 最后一层：分配所有剩余任务
        if remaining_tasks:
            layers[-1] = remaining_tasks

        # 移除空层
        layers = [layer for layer in layers if layer]

        # 构建层间连接 - 确保清晰的层次结构
        for i in range(len(layers) - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]

            # 确保每个当前层任务至少连接到下一层的1-3个任务
            for task_id in current_layer:
                # 确定连接数量（1-3个，但不超过下一层任务数）
                max_connections = min(3, len(next_layer))
                num_connections = random.randint(1, max_connections)

                # 随机选择下一层的任务进行连接
                connected_tasks = random.sample(next_layer, num_connections)
                for next_task in connected_tasks:
                    dag.graph.add_edge(task_id, next_task)

            # 确保下一层每个任务至少有一个前驱（避免孤立）
            for next_task in next_layer:
                if dag.graph.in_degree(next_task) == 0:
                    # 如果没有前驱，随机连接一个当前层任务
                    random_pred = random.choice(current_layer)
                    dag.graph.add_edge(random_pred, next_task)

        # 添加少量跨层连接（最多20%的概率），但确保不形成环
        for i in range(len(layers) - 2):  # 跳过相邻层
            current_layer = layers[i]
            skip_layer = layers[i + 2]  # 跳过下一层，连接到下下层

            for task_id in current_layer:
                if random.random() < 0.2 and skip_layer:  # 20%概率添加跨层边
                    target_task = random.choice(skip_layer)
                    # 检查添加这条边是否会产生环
                    dag.graph.add_edge(task_id, target_task)
                    try:
                        # 验证无环性
                        list(nx.topological_sort(dag.graph))
                    except nx.NetworkXUnfeasible:
                        # 如果产生环，移除这条边
                        dag.graph.remove_edge(task_id, target_task)

        # 最终验证和修复
        try:
            topological_order = list(nx.topological_sort(dag.graph))

            # 检查是否有孤立节点（没有前驱也没有后继）
            isolated = [node for node in dag.graph.nodes()
                        if dag.graph.in_degree(node) == 0 and dag.graph.out_degree(node) == 0]

            if isolated:
                # 将孤立节点连接到随机的前驱或后继
                for isolated_node in isolated:
                    if topological_order.index(isolated_node) > 0:
                        # 连接到前一个节点
                        pred_idx = topological_order.index(isolated_node) - 1
                        dag.graph.add_edge(topological_order[pred_idx], isolated_node)
                    elif len(topological_order) > 1:
                        # 连接到后一个节点
                        succ_idx = topological_order.index(isolated_node) + 1
                        dag.graph.add_edge(isolated_node, topological_order[succ_idx])

            print(f"DAG {dag_id} 生成成功，层数: {len(layers)}, 拓扑顺序验证通过")

        except nx.NetworkXUnfeasible:
            print(f"警告：DAG {dag_id} 仍有循环依赖，创建链式结构")
            # 回退方案：创建简单的链式结构
            dag.graph.clear_edges()
            for i in range(task_num - 1):
                dag.graph.add_edge(i, i + 1)

        # 构建矩阵
        dag.build_matrices()

        # 输出DAG结构信息
        entry_nodes = [node for node in dag.graph.nodes() if dag.graph.in_degree(node) == 0]
        exit_nodes = [node for node in dag.graph.nodes() if dag.graph.out_degree(node) == 0]
        print(f"DAG {dag_id} 结构: 入口节点{entry_nodes}, 出口节点{exit_nodes}, 总边数: {dag.graph.number_of_edges()}")

        return dag

    @staticmethod
    def generate_dataset():
        """生成完整数据集（使用修复后的拆分）"""
        dataset = {"original": {}, "split": {}}
        for task_type, size in DATASET_SIZE.items():
            original_dags = []
            split_dags = []
            for dag_id in range(size):
                # 生成原始DAG
                original_dag = EmbeddedDatasetGenerator.generate_dag(task_type, dag_id)
                original_dags.append(original_dag)

                # 使用修复后的任务拆分
                try:
                    split_dag, split_gain = TaskSplitter.split_tasks(original_dag)
                    split_dags.append(split_dag)
                except Exception as e:
                    print(f"DAG {dag_id} 拆分失败: {e}，使用原始DAG")
                    split_dags.append(original_dag)  # 回退到原始DAG

            dataset["original"][task_type] = original_dags
            dataset["split"][task_type] = split_dags

        # 生成动态任务数据集
        dynamic_dataset = []
        for _ in range(200):  # 200个动态任务
            task_type = random.choice(TASK_TYPES)
            comp_cost = {}
            for hw in HARDWARE_TYPES:
                comp_cost[hw] = EmbeddedDatasetGenerator.generate_task_comp_cost(task_type, hw)
            hardware_tags = EmbeddedDatasetGenerator.generate_hardware_tags(task_type)
            deadline = np.sum([v for v in comp_cost.values() if v != float("inf")]) * DEADLINE_FACTOR
            task_node = EmbeddedTaskNode(
                task_id=len(dynamic_dataset),
                task_type=task_type,
                comp_cost=comp_cost,
                hardware_tags=hardware_tags,
                deadline=deadline,
                period=0
            )
            dynamic_dataset.append(task_node)

        dataset["dynamic"] = dynamic_dataset
        print("数据集生成完成：")
        for task_type in TASK_TYPES:
            print(
                f"- {task_type}：原始DAG {len(dataset['original'][task_type])} 个，拆分后 {len(dataset['split'][task_type])} 个")
        print(f"- 动态任务：{len(dataset['dynamic'])} 个")
        return dataset


class BaselineScheduler:
    """基线算法实现（嵌入式常用调度算法）"""

    @staticmethod
    def heft_schedule(dag):
        """HEFT算法（异质最早完成时间）"""
        task_ids = sorted(dag.task_nodes.keys())
        hardware_indices = {hw: i for i, hw in enumerate(HARDWARE_TYPES)}
        scheduled_tasks = set()
        makespan = 0.0
        task_schedule = {}  # task_id → (hw_idx, start_time, finish_time)

        # 计算任务优先级（Rank_up）
        rank_up = {}
        for task_id in reversed(task_ids):
            task_idx = task_ids.index(task_id)
            # 平均计算开销
            avg_comp = np.mean(dag.comp_matrix[task_idx][dag.comp_matrix[task_idx] != float("inf")])
            # 平均通信开销
            avg_comm = 0.0
            succ_tasks = list(dag.graph.successors(task_id))
            if succ_tasks:
                comm_costs = []
                for succ_id in succ_tasks:
                    succ_idx = task_ids.index(succ_id)
                    comm_cost = np.mean(dag.comm_matrix[task_idx, succ_idx][dag.comm_matrix[task_idx, succ_idx] > 0])
                    comm_costs.append(comm_cost)
                avg_comm = np.mean(comm_costs) if comm_costs else 0.0
            # 后续任务最大Rank_up
            max_succ_rank = max([rank_up.get(succ_id, 0) for succ_id in succ_tasks], default=0)
            rank_up[task_id] = avg_comp + avg_comm + max_succ_rank

        # 按优先级排序（降序）
        sorted_tasks = sorted(task_ids, key=lambda x: rank_up[x], reverse=True)

        # 调度每个任务
        for task_id in sorted_tasks:
            task_idx = task_ids.index(task_id)
            best_hw = None
            best_finish = float("inf")
            best_start = 0.0

            # 遍历所有可用硬件
            for hw in HARDWARE_TYPES:
                hw_idx = hardware_indices[hw]
                if dag.comp_matrix[task_idx, hw_idx] == float("inf"):
                    continue

                # 计算最早开始时间（所有前驱任务完成时间 + 通信延迟）
                pred_finish_times = []
                for pred_id in dag.graph.predecessors(task_id):
                    if pred_id not in task_schedule:
                        continue
                    pred_hw_idx = task_schedule[pred_id][0]
                    pred_finish = task_schedule[pred_id][2]
                    comm_delay = dag.comm_matrix[task_ids.index(pred_id), task_idx, pred_hw_idx, hw_idx]
                    pred_finish_times.append(pred_finish + comm_delay)
                est = max(pred_finish_times) if pred_finish_times else 0.0

                # 计算最早完成时间
                eft = est + dag.comp_matrix[task_idx, hw_idx]

                # 更新最优硬件
                if eft < best_finish:
                    best_finish = eft
                    best_start = est
                    best_hw = hw_idx

            # 记录调度结果
            task_schedule[task_id] = (best_hw, best_start, best_finish)
            scheduled_tasks.add(task_id)
            makespan = max(makespan, best_finish)

        # 计算负载均衡（各硬件总执行时间方差）
        hw_total_time = np.zeros(HARDWARE_NUM)
        for task_id, (hw_idx, start, finish) in task_schedule.items():
            hw_total_time[hw_idx] += (finish - start)
        load_balance = np.var(hw_total_time)

        return makespan, load_balance

    @staticmethod
    def rm_schedule(dag):
        """RM调度（速率单调，适用于周期任务）"""
        # 筛选周期任务
        periodic_tasks = [t for t in dag.task_nodes.values() if t.period > 0]
        if not periodic_tasks:
            # 无周期任务时退化为HEFT
            return BaselineScheduler.heft_schedule(dag)

        # 按周期排序（周期越短优先级越高）
        periodic_tasks.sort(key=lambda x: x.period)
        makespan = 0.0
        hw_total_time = np.zeros(HARDWARE_NUM)

        # 调度每个周期任务
        for task in periodic_tasks:
            task_idx = list(dag.task_nodes.keys()).index(task.task_id)
            # 选择执行时间最短的硬件
            hw_idx = np.argmin(dag.comp_matrix[task_idx])
            exec_time = dag.comp_matrix[task_idx, hw_idx]
            # 最早开始时间 = 硬件当前总时间
            start_time = hw_total_time[hw_idx]
            finish_time = start_time + exec_time
            # 更新硬件总时间
            hw_total_time[hw_idx] = finish_time
            # 更新总Makespan
            makespan = max(makespan, finish_time)

        load_balance = np.var(hw_total_time)
        return makespan, load_balance

    @staticmethod
    def edf_schedule(dag):
        """EDF调度（截止时间最早优先）"""
        # 按截止时间排序
        tasks = sorted(dag.task_nodes.values(), key=lambda x: x.deadline)
        makespan = 0.0
        hw_total_time = np.zeros(HARDWARE_NUM)

        for task in tasks:
            task_idx = list(dag.task_nodes.keys()).index(task.task_id)
            # 选择执行时间最短的硬件
            hw_idx = np.argmin(dag.comp_matrix[task_idx])
            exec_time = dag.comp_matrix[task_idx, hw_idx]
            start_time = hw_total_time[hw_idx]
            finish_time = start_time + exec_time
            hw_total_time[hw_idx] = finish_time
            makespan = max(makespan, finish_time)

        load_balance = np.var(hw_total_time)
        return makespan, load_balance


class ExperimentEvaluator:
    """实验评估器（指标计算+结果可视化）"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.results = {"baseline": {}, "modrl": {}}

    def evaluate_baseline(self):
        """评估基线算法"""
        for algo in BASELINE_ALGORITHMS[:3]:  # 先评估HEFT、CPOP、ADTS，后续可扩展
            algo_results = {"makespan": [], "load_balance": [], "energy": []}
            for task_type in TASK_TYPES:
                dags = self.dataset["split"][task_type]
                for dag in dags:
                    if algo == "HEFT":
                        makespan, load_balance = BaselineScheduler.heft_schedule(dag)
                    elif algo == "RM":
                        makespan, load_balance = BaselineScheduler.rm_schedule(dag)
                    elif algo == "EDF":
                        makespan, load_balance = BaselineScheduler.edf_schedule(dag)
                    else:
                        # 其他基线算法可在此扩展
                        makespan, load_balance = BaselineScheduler.heft_schedule(dag)

                    # 计算能耗（各硬件执行时间 × 能耗系数）
                    task_ids = sorted(dag.task_nodes.keys())
                    energy = 0.0
                    for ti, task_id in enumerate(task_ids):
                        for hi in range(HARDWARE_NUM):
                            exec_time = dag.comp_matrix[ti, hi]
                            if exec_time != float("inf"):
                                energy += dag.energy_matrix[ti, hi]

                    algo_results["makespan"].append(makespan)
                    algo_results["load_balance"].append(load_balance)
                    algo_results["energy"].append(energy)

            # 计算平均值
            self.results["baseline"][algo] = {
                "makespan": np.mean(algo_results["makespan"]),
                "load_balance": np.mean(algo_results["load_balance"]),
                "energy": np.mean(algo_results["energy"])
            }
            print(f"基线算法 {algo} 评估完成：")
            print(f"  - 平均Makespan：{self.results['baseline'][algo]['makespan']:.2f} ms")
            print(f"  - 平均负载均衡（方差）：{self.results['baseline'][algo]['load_balance']:.2f}")
            print(f"  - 平均能耗：{self.results['baseline'][algo]['energy']:.2f} J")

    def evaluate_modrl(self, modrl_scheduler):
        """评估MODRL调度器"""
        modrl_results = {"makespan": [], "load_balance": [], "energy": [], "deadline_satisfaction": []}

        for task_type in TASK_TYPES:
            dags = self.dataset["split"][task_type]
            for dag in dags:
                # 模拟调度过程
                task_ids = sorted(dag.task_nodes.keys())
                hardware_indices = {hw: i for i, hw in enumerate(HARDWARE_TYPES)}
                task_schedule = {}
                hw_total_time = np.zeros(HARDWARE_NUM)
                total_energy = 0.0
                deadline_satisfied = 0

                # 任务优先级顺序
                seq_order = np.argsort([-dag.task_nodes[t].priority for t in task_ids])

                # 构建模型输入
                task_feat = np.hstack([
                    dag.comp_matrix,
                    dag.energy_matrix,
                    np.array([[t.priority for t in dag.task_nodes.values()]]).T
                ])  # (task_num, comp_dim + energy_dim + 1)
                adj = nx.to_numpy_array(dag.graph)  # (task_num, task_num)
                hardware_feat = np.array([
                    [EMBEDDED_HARDWARES[hw]["算力"], EMBEDDED_HARDWARES[hw]["内存"]]
                    for hw in HARDWARE_TYPES
                ])  # (hardware_num, 2)
                load_states = np.zeros(HARDWARE_NUM)  # 初始负载为0
                history_resource_data = np.random.rand(1, 20, 4)  # 模拟历史资源数据

                # 转换为Tensor
                task_feat_tensor = torch.tensor(task_feat[None], dtype=torch.float32).to(DEVICE)
                adj_tensor = torch.tensor(adj[None], dtype=torch.float32).to(DEVICE)
                seq_order_tensor = torch.tensor(seq_order[None], dtype=torch.long).to(DEVICE)
                hardware_feat_tensor = torch.tensor(hardware_feat[None], dtype=torch.float32).to(DEVICE)
                load_states_tensor = torch.tensor(load_states[None], dtype=torch.float32).to(DEVICE)
                history_resource_tensor = torch.tensor(history_resource_data, dtype=torch.float32).to(DEVICE)

                # 调度每个任务
                for ti, task_id in enumerate(task_ids):
                    # 获取Q值
                    q_values = modrl_scheduler(
                        task_feat_tensor,
                        adj_tensor,
                        seq_order_tensor,
                        hardware_feat_tensor,
                        load_states_tensor,
                        history_resource_tensor
                    )
                    # 选择Q值最大的硬件
                    hw_idx = torch.argmax(q_values[0, ti]).item()
                    if dag.comp_matrix[ti, hw_idx] == float("inf"):
                        # 若该硬件不支持，选择次优
                        q_values[0, ti, hw_idx] = -float("inf")
                        hw_idx = torch.argmax(q_values[0, ti]).item()

                    # 计算开始和结束时间
                    start_time = hw_total_time[hw_idx]
                    exec_time = dag.comp_matrix[ti, hw_idx]
                    finish_time = start_time + exec_time

                    # 更新硬件总时间和负载
                    hw_total_time[hw_idx] = finish_time
                    load_states = hw_total_time / finish_time  # 负载 = 硬件总执行时间 / 当前总时间

                    # 计算能耗
                    energy = dag.energy_matrix[ti, hw_idx]
                    total_energy += energy

                    # 检查截止时间
                    task = dag.task_nodes[task_id]
                    if finish_time <= task.deadline:
                        deadline_satisfied += 1

                    # 记录调度结果
                    task_schedule[task_id] = (hw_idx, start_time, finish_time)

                # 计算指标
                makespan = max(hw_total_time)
                load_balance = np.var(hw_total_time)
                deadline_satisfaction_rate = deadline_satisfied / len(task_ids)

                modrl_results["makespan"].append(makespan)
                modrl_results["load_balance"].append(load_balance)
                modrl_results["energy"].append(total_energy)
                modrl_results["deadline_satisfaction"].append(deadline_satisfaction_rate)

        # 计算平均值
        self.results["modrl"] = {
            "makespan": np.mean(modrl_results["makespan"]),
            "load_balance": np.mean(modrl_results["load_balance"]),
            "energy": np.mean(modrl_results["energy"]),
            "deadline_satisfaction": np.mean(modrl_results["deadline_satisfaction"])
        }
        print("\nMODRL调度器评估完成：")
        print(f"  - 平均Makespan：{self.results['modrl']['makespan']:.2f} ms")
        print(f"  - 平均负载均衡（方差）：{self.results['modrl']['load_balance']:.2f}")
        print(f"  - 平均能耗：{self.results['modrl']['energy']:.2f} J")
        print(f"  - 截止时间满足率：{self.results['modrl']['deadline_satisfaction']:.2%}")

    def visualize_results(self):
        """可视化实验结果"""
        # 1. Makespan对比
        plt.figure(figsize=(12, 8))

        # 子图1：Makespan
        plt.subplot(2, 2, 1)
        algorithms = list(self.results["baseline"].keys()) + ["MODRL"]
        makespans = [self.results["baseline"][algo]["makespan"] for algo in algorithms[:-1]] + [
            self.results["modrl"]["makespan"]]
        plt.bar(algorithms, makespans, color=["gray"] * (len(algorithms) - 1) + ["red"])
        plt.title("各算法平均Makespan对比")
        plt.ylabel("Makespan (ms)")
        plt.xticks(rotation=45)

        # 子图2：负载均衡
        plt.subplot(2, 2, 2)
        load_balances = [self.results["baseline"][algo]["load_balance"] for algo in algorithms[:-1]] + [
            self.results["modrl"]["load_balance"]]
        plt.bar(algorithms, load_balances, color=["gray"] * (len(algorithms) - 1) + ["red"])
        plt.title("各算法平均负载均衡（方差越小越好）")
        plt.ylabel("负载方差")
        plt.xticks(rotation=45)

        # 子图3：能耗
        plt.subplot(2, 2, 3)
        energies = [self.results["baseline"][algo]["energy"] for algo in algorithms[:-1]] + [
            self.results["modrl"]["energy"]]
        plt.bar(algorithms, energies, color=["gray"] * (len(algorithms) - 1) + ["red"])
        plt.title("各算法平均能耗对比")
        plt.ylabel("能耗 (J)")
        plt.xticks(rotation=45)

        # 子图4：截止时间满足率
        plt.subplot(2, 2, 4)
        deadline_rates = [0.0] * len(algorithms[:-1]) + [self.results["modrl"]["deadline_satisfaction"]]
        plt.bar(algorithms, deadline_rates, color=["gray"] * (len(algorithms) - 1) + ["red"])
        plt.title("截止时间满足率对比")
        plt.ylabel("满足率")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig("experiment_results.png")
        plt.show()

        # 输出量化对比
        print("\n量化对比结果（相对于HEFT算法）：")
        heft_results = self.results["baseline"]["HEFT"]
        modrl_results = self.results["modrl"]
        print(
            f"  - Makespan提升：{(heft_results['makespan'] - modrl_results['makespan']) / heft_results['makespan']:.2%}")
        print(
            f"  - 负载均衡提升：{(heft_results['load_balance'] - modrl_results['load_balance']) / heft_results['load_balance']:.2%}")
        print(f"  - 能耗降低：{(heft_results['energy'] - modrl_results['energy']) / heft_results['energy']:.2%}")
        print(f"  - 截止时间满足率：{modrl_results['deadline_satisfaction']:.2%}")