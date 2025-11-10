import networkx as nx
import numpy as np

"""
实现静态 HEFT 算法 (Heterogeneous Earliest Finish Time)
[cite: 7, 21, 114]
这是我们的基准，也是阶段1行为克隆的“专家”。
"""


def get_avg_exec_time(task, processors):
    """ 计算任务在所有处理器上的平均执行时间 """
    return np.mean([task.exec_times[p.type] for p in processors])


def get_avg_comm_cost(u, v, dag, processors):
    """ 计算平均通信成本 """
    data_size = dag.edges[u.id, v.id].get('data_size', 0)
    # TODO: 需要一个简化的带宽模型
    avg_bandwidth = 1.0  # 占位符
    return data_size / avg_bandwidth


def compute_upward_ranks(dag, processors):
    """
    阶段1: 计算所有任务的向上排名 (Upward Rank)
    rank_u = w_i + max_{j \in succ(i)} (c_ij + rank_u(j))
    """
    ranks = {}
    for node_id in reversed(list(nx.topological_sort(dag))):
        task = dag.nodes[node_id]['task']
        avg_exec = get_avg_exec_time(task, processors)

        max_succ_rank = 0.0
        for succ_id in dag.successors(node_id):
            succ_task = dag.nodes[succ_id]['task']
            avg_comm = get_avg_comm_cost(task, succ_task, dag, processors)
            succ_rank = ranks.get(succ_id, 0.0)
            max_succ_rank = max(max_succ_rank, avg_comm + succ_rank)

        ranks[node_id] = avg_exec + max_succ_rank

    return ranks


def schedule_heft(dag: nx.DiGraph, processors: list):
    """
    阶段2: 任务调度
    按 rank_u 递减顺序，将任务调度到提供最早完成时间(EFT)的处理器上。
    """
    ranks = compute_upward_ranks(dag, processors)

    # 按rank_u递减排序
    task_queue = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    # 调度时间表 (proc_id -> list of (start, finish, task_id))
    schedule = {p.id: [] for p in processors}
    task_finish_times = {}  # 存储每个任务的实际完成时间

    expert_decisions = []  # 用于BC训练 [cite: 116]

    for task_id, rank in task_queue:
        task = dag.nodes[task_id]['task']

        best_eft = float('inf')
        best_proc_id = -1
        best_start_time = 0

        # 计算任务的“就绪时间”(Ready Time)
        # max(pred_finish_time + comm_cost)
        ready_time = 0.0
        for pred_id in dag.predecessors(task_id):
            pred_finish = task_finish_times[pred_id]
            # TODO: 实现更真实的通信成本 [cite: 8]
            comm_cost = 0  # HEFT 的一个关键缺陷：倾向于0通信成本 [cite: 8]
            ready_time = max(ready_time, pred_finish + comm_cost)

        # 寻找能最早完成此任务的处理器
        for proc in processors:
            exec_time = task.exec_times[proc.type]

            # 寻找处理器上的可用时间槽
            # (简化的EFT计算, 真实HEFT会查找间隙)
            proc_ready_time = schedule[proc.id][-1][1] if schedule[proc.id] else 0.0

            start_time = max(ready_time, proc_ready_time)
            finish_time = start_time + exec_time

            if finish_time < best_eft:
                best_eft = finish_time
                best_proc_id = proc.id
                best_start_time = start_time

        # 记录调度决策
        schedule[best_proc_id].append((best_start_time, best_eft, task_id))
        task_finish_times[task_id] = best_eft

        # TODO: 在此决策点捕获'状态' (s_t)
        current_state_representation = {}  # 占位符

        # 存储 (s, a) 对用于BC [cite: 116]
        expert_decisions.append({
            "state": current_state_representation,
            "action": (task_id, best_proc_id)
        })

    makespan = max(task_finish_times.values())
    print(f"[HEFT] 基准测试完成. 完工时间 (Makespan): {makespan:.2f}s")

    # 返回详细的调度表以用于甘特图
    task_map = {node_id: dag.nodes[node_id]['task'] for node_id in dag.nodes}
    return makespan, schedule, task_map  # <--- 新的代码行