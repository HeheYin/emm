import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Tuple, Dict, Any, List, Optional

from dag_utils import *

"""
核心环境：实现蓝图第二部分定义的全新MDP架构
(基于上一轮响应中已补完的事件驱动模拟逻辑)
"""


class MCSSchedulingEnv(gym.Env):

    def __init__(self, dag_spec, proc_spec, energy_constants=None, reward_weights=None):
        super().__init__()

        self.dag_spec = dag_spec
        self.proc_spec = proc_spec

        # 能量模型常数 (占位符)
        self.energy_constants = energy_constants or {
            "P_CPU_STATIC": 0.5, "P_NPU_STATIC": 1.0,
            "ALPHA_CPU": 0.2, "ALPHA_NPU": 0.8,  # P_dynamic = alpha * V^2 * f
            "P_DMA": 0.5, "BANDWIDTH_DMA": 1e6  # 1MB/s
        }

        # 奖励权重 (占位符) [基于 Sec 3.3]
        self.reward_weights = reward_weights or {
            "w_balance": 0.0, "w_deadline": 0.5,
            "w_comm": 0.1, "w_energy": 0.2, "w_final": 10.0
        }

        self.dag, self.task_map = self._setup_environment()
        self.num_tasks = self.dag.number_of_nodes()
        self.num_procs = len(self.processors)

        # 动作空间: 扁平化的 (Task_ID * Processor_ID) 索引
        self.action_space = spaces.Discrete(self.num_tasks * self.num_procs)

        # 状态空间: 由 _get_obs() 中的异构图定义
        self.observation_space = spaces.Dict()  # Gymnasium 推荐使用 Dict 空间

    def _setup_environment(self) -> Tuple[nx.DiGraph, Dict[int, Task]]:
        """重置/初始化DAG和处理器的辅助函数"""
        dag, task_map = create_dag_from_spec(**self.dag_spec)
        self.processors = [Processor(**p) for p in self.proc_spec]
        for p in self.processors:
            p.set_task_map(task_map)

        self.current_time = 0.0
        self.ready_task_pool: List[int] = []
        self.completed_task_count = 0

        # 初始化: 找到所有没有前驱的任务
        for n_id in dag.nodes:
            if dag.in_degree(n_id) == 0:
                task_map[n_id].status = STATUS_READY
                self.ready_task_pool.append(n_id)

        return dag, task_map

    def reset(self, seed=None, options=None) -> Tuple[HeteroData, Dict]:
        super().reset(seed=seed)
        self.dag, self.task_map = self._setup_environment()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action_idx: int) -> Tuple[HeteroData, float, bool, bool, Dict]:
        """
        环境的核心推进逻辑
        action_idx: 一个扁平化的索引 (task_id * num_procs + proc_id)
        """

        # 1. 解码动作
        task_id = action_idx // self.num_procs
        proc_id = action_idx % self.num_procs

        task = self.task_map[task_id]
        proc = self.processors[proc_id]

        # 2. 验证动作 (是否在掩码允许的范围内)
        if not self._is_action_valid(task, proc):
            # 惩罚无效动作 (代理试图调度一个未就绪或被阻塞的任务)
            reward = -10.0
            terminated = self._check_terminated()
            return self._get_obs(), reward, terminated, False, self._get_info({"invalid_action": True})

        # 3. 执行动作 (调度任务)
        # [实现 Sec 3.2 抢占逻辑]
        if proc.running_task_id is not None:
            running_task = self.task_map[proc.running_task_id]
            # print(f"--- 抢占! HI任务 {task.id} 抢占了 LO任务 {running_task.id} on P{proc.id} ---")

            # 更新被抢占任务的剩余时间
            time_executed = self.current_time - running_task.start_time
            running_task.remaining_time -= time_executed

            # 将被抢占的任务放回就绪队列
            running_task.status = STATUS_READY
            self.ready_task_pool.append(running_task.id)

        # 4. 分配新任务
        self.ready_task_pool.remove(task_id)  # 从就绪池中移除
        task.status = STATUS_RUNNING
        task.assigned_proc = proc.id
        task.start_time = self.current_time

        # 如果任务之前被抢占过，它会继续执行剩余时间
        if task.remaining_time <= 0 or task.remaining_time == task.exec_times.get(PROC_CPU,
                                                                                  task.exec_times.get(PROC_NPU, 0.0)):
            exec_time = task.exec_times[proc.type]
            task.remaining_time = exec_time
        else:
            exec_time = task.remaining_time  # 从中断处恢复

        task.finish_time = self.current_time + exec_time

        proc.running_task_id = task.id
        proc.becomes_free_at = task.finish_time

        # 5. 计算即时奖励 (r_t) [Sec 3.3]
        reward, reward_info = self._calculate_dense_reward(task, proc)

        # 6. 推进模拟到下一个事件点
        self._advance_simulation()

        # 7. 检查终止条件
        terminated = self._check_terminated()

        if terminated:
            # 添加最终稀疏奖励 (-makespan)
            final_reward = -self.current_time
            reward += self.reward_weights["w_final"] * final_reward
            reward_info["r_final"] = final_reward

        truncated = False  # (由 max_steps_per_ep 在 pipeline 中处理)

        obs = self._get_obs()
        info = self._get_info(reward_info)

        return obs, reward, terminated, truncated, info

    def _is_action_valid(self, task: Task, proc: Processor) -> bool:
        """检查动作 (task, proc) 是否有效"""

        # 规则1: 任务必须处于就绪状态
        if task.status != STATUS_READY:
            return False

        # 规则2: 检查处理器是否可用
        if proc.running_task_id is None:
            return True  # 处理器空闲，始终有效

        # 规则3: 处理器忙，检查抢占 [Sec 3.2]
        running_task = self.task_map[proc.running_task_id]

        # 仅当新任务是HI，且正在运行的任务是LO时，才允许抢占
        if task.criticality == CRIT_HI and running_task.criticality == CRIT_LO:
            return True  # 合法的抢占

        # 所有其他情况 (LO抢LO, HI抢HI, LO抢HI) 均无效
        return False

    def _advance_simulation(self):
        """
        事件驱动的模拟推进器。
        找到下一个最早完成的任务，并将时间推进到该点。
        """
        running_tasks_finish_times = [
            p.becomes_free_at
            for p in self.processors if p.running_task_id is not None
        ]

        if not running_tasks_finish_times:
            # 没有任务在运行，不推进时间 (等待下一个决策)
            return

        # 找到最早的完成时间
        min_finish_time = min(running_tasks_finish_times)

        if min_finish_time <= self.current_time:
            # 避免时间停滞
            min_finish_time = self.current_time + 1e-5

        self.current_time = min_finish_time

        # 完成所有在该时间点结束的任务
        for proc in self.processors:
            if proc.running_task_id is not None and proc.becomes_free_at == self.current_time:
                task_id = proc.running_task_id
                task = self.task_map[task_id]

                task.status = STATUS_COMPLETED
                task.remaining_time = 0
                proc.running_task_id = None
                proc.becomes_free_at = 0.0
                self.completed_task_count += 1

                # 任务完成，更新其后继任务的状态
                self._update_ready_tasks(task_id)

    def _update_ready_tasks(self, completed_task_id: int):
        """检查已完成任务的后继，看它们是否已就绪"""
        for succ_id in self.dag.successors(completed_task_id):
            task = self.task_map[succ_id]
            if task.status == STATUS_UNREADY:
                # 检查所有前驱是否都已完成
                ready = True
                for pred_id in self.dag.predecessors(succ_id):
                    if self.task_map[pred_id].status != STATUS_COMPLETED:
                        ready = False
                        break
                if ready:
                    task.status = STATUS_READY
                    if succ_id not in self.ready_task_pool:
                        self.ready_task_pool.append(succ_id)

    def _check_terminated(self) -> bool:
        """检查是否所有任务都已完成"""
        return self.completed_task_count == self.num_tasks

    def _get_obs(self) -> HeteroData:
        """
        构建异构图注意力网络(HGAT)的状态表示 [Sec 3.1]
        """
        data = HeteroData()

        # 1. 节点特征: 任务(Task) [基于 表3.1]
        task_features = []
        task_nodes = sorted(self.task_map.keys())
        for n_id in task_nodes:
            task = self.task_map[n_id]

            # 计算动态 $laxity$ (松弛度)
            # 计算动态 $laxity$ (松弛度) 和处理 $deadline$
            if task.deadline == float('inf'):
                laxity = 1e6  # 大的正数
                deadline_feat = 1e6  # [修复] 替换 inf
            else:
                deadline_feat = task.deadline  # 使用实际的 deadline
                rem_time_est = task.remaining_time if task.remaining_time > 0 else task.exec_times[
                    PROC_NPU]  # 默认用NPU时间估算
                laxity = task.deadline - (self.current_time + rem_time_est)

                # [新修复] 裁剪 laxity 特征以防止数值不稳定
                # 允许大的正松弛度，但限制大的负松弛度 (错过ddl)
            clipped_laxity = max(laxity, -1000.0)

            task_features.append([
                task.exec_times[PROC_CPU],  # $exec_cpu$
                task.exec_times[PROC_NPU],  # $exec_npu$
                task.criticality,  # $crit$
                deadline_feat,  # $deadline$ [已修复]
                clipped_laxity,  # $laxity$ [新修复]
                task.status,  # $status$
            ])
        data['task'].x = torch.tensor(task_features, dtype=torch.float)

        # 2. 节点特征: 处理器(Proc) [基于 表3.1]
        proc_features = []
        for p in self.processors:
            proc_features.append([p.type, p.load, p.queue_len, p.curr_crit])
        data['proc'].x = torch.tensor(proc_features, dtype=torch.float)

        # 3. 边: Task-Task (依赖) [基于 表3.2]
        task_task_edges = []
        task_task_attrs = []
        for u, v, attrs in self.dag.edges(data=True):
            task_task_edges.append((u, v))
            task_task_attrs.append([attrs.get('data_size', 0)])  # $data_size$

        if task_task_edges:
            data['task', 'depends_on', 'task'].edge_index = torch.tensor(task_task_edges,
                                                                         dtype=torch.long).t().contiguous()
            data['task', 'depends_on', 'task'].edge_attr = torch.tensor(task_task_attrs, dtype=torch.float)
        else:
            data['task', 'depends_on', 'task'].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 4. 边: Task-Proc (分配可能性) [基于 表3.2]
        task_proc_edges = []
        task_proc_attrs = []
        for t_id in task_nodes:
            for p_id in range(self.num_procs):
                task_proc_edges.append((t_id, p_id))
                comm_cost = self._estimate_comm_cost(self.task_map[t_id], self.processors[p_id])
                task_proc_attrs.append([comm_cost])  # $comm_cost$

        data['task', 'can_run_on', 'proc'].edge_index = torch.tensor(task_proc_edges, dtype=torch.long).t().contiguous()
        data['task', 'can_run_on', 'proc'].edge_attr = torch.tensor(task_proc_attrs, dtype=torch.float)

        # 5. 添加反向边 (HGTConv 需要)
        data['proc', 'rev_can_run_on', 'task'].edge_index = data['task', 'can_run_on', 'proc'].edge_index[[1, 0]]
        data['proc', 'rev_can_run_on', 'task'].edge_attr = data['task', 'can_run_on', 'proc'].edge_attr

        return data

    def _get_info(self, reward_info: Optional[Dict] = None) -> Dict:
        return {
            "current_time": self.current_time,
            "ready_tasks": self.ready_task_pool.copy(),
            "completed_tasks": self.completed_task_count,
            "reward_components": reward_info
        }

    def _estimate_comm_cost(self, task: Task, proc: Processor) -> float:
        """ 估算 $comm_cost$ (延迟) [Sec 3.2, 状态特征] """
        total_cost_time = 0.0
        for pred_id in self.dag.predecessors(task.id):
            pred_task = self.task_map[pred_id]
            if pred_task.assigned_proc != -1 and pred_task.assigned_proc != proc.id:
                data_size = self.dag.edges[pred_id, task.id].get('data_size', 0)
                total_cost_time += data_size / (self.energy_constants["BANDWIDTH_DMA"] + 1e-6)
        return total_cost_time

    def _calculate_dense_reward(self, task: Task, proc: Processor) -> Tuple[float, Dict]:
        """ 实现 表3.3: 多目标密集奖励函数 """

        # 1. r_balance (解决 RCA 1)
        loads = [p.load for p in self.processors]
        loads[proc.id] = 1.0  # 假设此动作为真
        r_balance = -np.var(loads)

        # 2. r_deadline (解决 RCA 2)
        exec_time_est = task.remaining_time if task.remaining_time > 0 else task.exec_times[proc.type]
        laxity = task.deadline - (self.current_time + exec_time_est)
        r_missed_deadline = -100.0 if laxity < 0 else 0.0
        r_laxity = (task.criticality + 0.1) * (1.0 / (max(laxity, 0) + 0.1))
        r_deadline = r_laxity + r_missed_deadline

        # 3. r_comm (解决 "通信近视")
        r_comm_penalty = self._estimate_comm_cost(task, proc)

        # 4. r_energy (解决能耗)
        r_energy_penalty, _ = self._calculate_dynamic_energy(task, proc, exec_time_est)

        # 组合奖励
        w = self.reward_weights
        total_reward = (w["w_balance"] * r_balance +
                        w["w_deadline"] * r_deadline -
                        w["w_comm"] * r_comm_penalty -
                        w["w_energy"] * r_energy_penalty)

        return total_reward, {"r_balance": r_balance, "r_deadline": r_deadline, "r_comm_penalty": r_comm_penalty,
                              "r_energy_penalty": r_energy_penalty}

    def _calculate_dynamic_energy(self, task: Task, proc: Processor, exec_time: float) -> Tuple[float, Dict]:
        """ 实现 表3.4: 综合动态能耗模型 """
        freq = 1.0  # 假设没有 DVFS

        # 1. E_compute
        if proc.type == PROC_CPU:
            p_static = self.energy_constants["P_CPU_STATIC"]
            p_dynamic = self.energy_constants["ALPHA_CPU"] * freq
        else:
            p_static = self.energy_constants["P_NPU_STATIC"]
            p_dynamic = self.energy_constants["ALPHA_NPU"] * freq
        e_compute = (p_static + p_dynamic) * exec_time

        # 2. E_migrate
        e_migrate = 0.0
        p_dma = self.energy_constants["P_DMA"]
        bandwidth_dma = self.energy_constants["BANDWIDTH_DMA"] + 1e-6
        for pred_id in self.dag.predecessors(task.id):
            pred_task = self.task_map[pred_id]
            if pred_task.assigned_proc != -1 and pred_task.assigned_proc != proc.id:
                data_size = self.dag.edges[pred_id, task.id].get('data_size', 0)
                migration_time = data_size / bandwidth_dma
                e_migrate += p_dma * migration_time  # E = P * t

        # 3. E_idle
        e_idle = 0.0
        for p in self.processors:
            if p.id != proc.id:
                p_idle = self.energy_constants["P_CPU_STATIC"] if p.type == PROC_CPU else self.energy_constants[
                    "P_NPU_STATIC"]
                e_idle += p_idle * exec_time  # 假设空闲时间 = 任务执行时间

        total_energy = e_compute + e_migrate + e_idle
        return total_energy, {"e_compute": e_compute, "e_migrate": e_migrate, "e_idle": e_idle}

    def get_action_mask(self) -> np.ndarray:
        """
        实现动作屏蔽 [Sec 3.2]
        返回一个扁平化的布尔掩码，长度为 (num_tasks * num_procs)
        """
        mask = np.zeros(self.num_tasks * self.num_procs, dtype=bool)

        for t_id in self.ready_task_pool:
            for p_id in range(self.num_procs):
                task = self.task_map[t_id]
                proc = self.processors[p_id]

                # 使用与 step 中相同的验证逻辑
                if self._is_action_valid(task, proc):
                    # TODO: 在此实现硬约束检查 (例如, 检查此动作是否 *必然* 导致HI任务错过deadline)
                    # if self._predict_deadline_miss(task, proc):
                    #    pass # 保持 mask[idx] = False
                    # else:
                    #    mask[idx] = True

                    idx = t_id * self.num_procs + p_id
                    mask[idx] = True  # 简化版：如果MCS规则有效，就取消屏蔽

        return mask