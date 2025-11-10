import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 定义任务生命周期状态 [基于 表3.1, "状态" $status$]
STATUS_UNREADY = 0
STATUS_READY = 1
STATUS_RUNNING = 2
STATUS_COMPLETED = 3

# 定义处理器类型 [基于 表3.1, "处理器类型" $type$]
PROC_CPU = 0
PROC_NPU = 1

# 定义关键性级别 [基于 表3.1, "关键性级别" $crit$]
CRIT_LO = 0
CRIT_HI = 1


@dataclass
class Task:
    """ 定义DAG中的单个任务节点 (基于 表3.1) """
    id: int
    name: str

    # 异构执行时间
    exec_times: Dict[int, float]  # {PROC_CPU: 10.0, PROC_NPU: 2.0}

    # MCS 混合关键性属性
    criticality: int = CRIT_LO
    deadline: float = float('inf')

    # 内部状态
    status: int = STATUS_UNREADY
    start_time: float = -1.0
    finish_time: float = -1.0
    assigned_proc: int = -1

    # 新增：用于抢占和动态计算
    remaining_time: float = -1.0  # 任务剩余执行时间

    def __post_init__(self):
        # 初始化 remaining_time 为一个基准值
        self.remaining_time = self.exec_times.get(PROC_CPU, self.exec_times.get(PROC_NPU, 0.0))


@dataclass
class Processor:
    """ 定义系统中的一个处理器 (基于 表3.1) """
    id: int
    type: int  # PROC_CPU 或 PROC_NPU

    # 动态状态
    running_task_id: Optional[int] = None
    becomes_free_at: float = 0.0  # 此处理器变为空闲的模拟时间

    # 关联的任务对象 (用于快速查找)
    _task_map: Dict[int, 'Task'] = field(default_factory=dict, repr=False)

    def set_task_map(self, task_map: Dict[int, 'Task']):
        self._task_map = task_map

    @property
    def load(self) -> float:
        # 负载定义为是否正在运行任务
        return 1.0 if self.running_task_id is not None else 0.0

    @property
    def queue_len(self) -> int:
        # 在我们的事件驱动模型中，"就绪"任务在环境的 "ready_task_pool" 中
        # 处理器本身没有队列
        return 0

    @property
    def curr_crit(self) -> int:
        """ [实现 表3.1]：获取当前运行任务的关键性 """
        if self.running_task_id is not None and self._task_map:
            task = self._task_map.get(self.running_task_id)
            if task:
                return task.criticality
        return CRIT_LO  # 默认空闲时为低关键性


def create_dag_from_spec(task_specs, dependencies) -> Tuple[nx.DiGraph, Dict[int, Task]]:
    """
    一个辅助函数，用于创建 NetworkX DiGraph 并填充 Task 对象。
    返回: (dag, task_map)
    """
    dag = nx.DiGraph()
    task_map = {}  # 新增: ID到Task对象的映射

    # 1. 添加任务节点 (Task)
    for spec in task_specs:
        task_obj = Task(**spec)
        dag.add_node(task_obj.id, task=task_obj)
        task_map[task_obj.id] = task_obj

    # 2. 添加依赖边 (Task-Task)
    for (u_id, v_id, attrs) in dependencies:
        dag.add_edge(u_id, v_id, **attrs)  # attrs 包含 'data_size'

    return dag, task_map