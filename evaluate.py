import torch
import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt  # <--- 导入 Matplotlib
import matplotlib.font_manager as fm
from typing import List, Tuple, Dict

# 导入您的项目文件
try:
    from mcs_env import MCSSchedulingEnv
    from model_hgat import HGATActorCritic
    from agent_ppo import PPOAgent
    from baseline_heft import schedule_heft
    from dag_utils import create_dag_from_spec, Processor, Task
    from dag_utils import CRIT_LO, CRIT_HI, PROC_CPU, PROC_NPU
except ImportError as e:
    print(f"导入错误: {e}")
    print(
        "请确保此脚本与 mcs_env.py, model_hgat.py, agent_ppo.py, baseline_heft.py, dag_utils.py 在同一目录或Python路径中。")
    sys.exit(1)

# 配置中文字体
try:
    # 尝试使用系统中的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except Exception as e:
    print(f"中文字体设置失败: {e}。图表标题可能显示为方框。")


def get_problem_specs():
    """
    加载与 main.py 中用于训练完全相同的 DAG 和 处理器规格

    """
    dag_spec = {
        "task_specs": [
            {"id": 0, "name": "T1", "exec_times": {0: 5, 1: 2}, "criticality": CRIT_LO},
            {"id": 1, "name": "T2", "exec_times": {0: 2, 1: 6}, "criticality": CRIT_LO},
            {"id": 2, "name": "T3", "exec_times": {0: 4, 1: 1}, "criticality": CRIT_LO},
            {"id": 3, "name": "T4", "exec_times": {0: 3, 1: 1}, "criticality": CRIT_HI, "deadline": 10.0},
            {"id": 4, "name": "T5", "exec_times": {0: 2, 1: 2}, "criticality": CRIT_LO},
            {"id": 5, "name": "T6", "exec_times": {0: 8, 1: 3}, "criticality": CRIT_LO},
        ],
        "dependencies": [
            (0, 2, {"data_size": 10}), (1, 2, {"data_size": 20}), (2, 3, {"data_size": 5}),
            (2, 4, {"data_size": 15}), (3, 5, {"data_size": 10}), (4, 5, {"data_size": 5}),
        ]
    }
    # 在 proc_spec 中添加 'name' 字段用于绘图
    proc_spec = [
        {"id": 0, "type": PROC_CPU, "name": "CPU (P0)"},
        {"id": 1, "type": PROC_NPU, "name": "NPU (P1)"},
    ]
    return dag_spec, proc_spec


def plot_gantt_chart(
        schedule_data: List[Tuple[str, int, float, float]],
        processors: List[Dict],
        title: str,
        filename: str
):
    """
    生成甘特图以可视化调度结果。
    """
    print(f"--- 正在生成甘特图: {filename} ---")

    proc_names = {p['id']: p['name'] for p in processors}

    fig, ax = plt.subplots(figsize=(15, 5))

    # 为任务名称创建一致的颜色映射
    task_names = sorted(list(set([d[0] for d in schedule_data])))
    colors = plt.cm.get_cmap('tab20', len(task_names))
    task_colors = {name: colors(i) for i, name in enumerate(task_names)}

    for (task_name, proc_id, start, finish) in schedule_data:
        duration = finish - start
        proc_name = proc_names.get(proc_id, f"Proc {proc_id}")

        # 绘制条形图
        ax.barh(
            proc_name,  # y (处理器)
            duration,  # width (持续时间)
            left=start,  # x (开始时间)
            edgecolor='black',
            color=task_colors.get(task_name, 'gray'),
            align='center'
        )

        # 在条形图上添加任务名称
        ax.text(
            start + duration / 2,
            proc_name,
            task_name,
            ha='center',
            va='center',
            color='black',
            fontweight='bold'
        )

    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("处理器")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle=':', alpha=0.7)

    # 反转Y轴，使CPU在上方 (可选)
    ax.invert_yaxis()

    # 保存图表
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"甘特图已保存至: {filename}")


def evaluate_heft_baseline(dag_spec, proc_spec):
    """
    运行 HEFT 算法并生成其甘特图。
    """
    print("--- 1. 正在评估 HEFT 基准 ---")
    dag, _ = create_dag_from_spec(**dag_spec)
    # [修复] 明确传递 'id' 和 'type'，忽略 'name'
    processors_objs = [Processor(id=p['id'], type=p['type']) for p in proc_spec]

    # 运行 HEFT (修改版)
    heft_makespan, heft_schedule_dict, heft_task_map = schedule_heft(dag, processors_objs)

    print(f"[HEFT] 基准完工时间 (Makespan): {heft_makespan:.4f}s")

    # 提取 HEFT 调度数据用于绘图
    gantt_data = []
    for proc_id, tasks in heft_schedule_dict.items():
        for (start, finish, task_id) in tasks:
            task_name = heft_task_map[task_id].name
            gantt_data.append((task_name, proc_id, start, finish))

    # 计算 HEFT 的截止日期未达标情况
    deadline_misses = 0
    for task_id, task in heft_task_map.items():
        if task.deadline != float('inf'):
            # 从调度表中找到实际完成时间
            task_finish_time = 0
            for (t_name, p_id, s, f) in gantt_data:
                if t_name == task.name:
                    task_finish_time = f
                    break
            if task_finish_time > task.deadline:
                deadline_misses += 1

    print(f"[HEFT] 截止日期未达标: {deadline_misses}")

    plot_gantt_chart(gantt_data, proc_spec, "HEFT 调度 (基准)", "gantt_heft.png")

    return {"makespan": heft_makespan, "deadline_misses": deadline_misses, "total_energy": "N/A"}


def evaluate_drl_model(dag_spec, proc_spec, model_path="model_final_ppo.pth"):
    """
    加载、运行并评估 DRL 模型，生成其甘特图并报告多项指标。
    """
    print("\n--- 2. 正在评估 DRL (PPO) 模型 ---")

    # 2a. 初始化环境和模型
    env_proc_spec = [{"id": p['id'], "type": p['type']} for p in proc_spec]
    env = MCSSchedulingEnv(dag_spec, env_proc_spec)
    dummy_obs, _ = env.reset()
    metadata = dummy_obs.metadata()

    model = HGATActorCritic(hidden_dim=128, metadata=metadata)
    agent = PPOAgent(model)

    # 2b. 加载训练好的权重
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{model_path}'。")
        return None
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

    # 2d. 运行一个完整的评估回合
    state, info = env.reset()
    done = truncated = False

    total_energy = 0.0

    with torch.no_grad():
        while not done and not truncated:
            action_mask = env.get_action_mask()
            if not np.any(action_mask):
                break
            action_idx, _, _ = agent.get_action_and_value(state, action_mask)
            state, reward, done, truncated, info = env.step(action_idx)

            # 累积指标
            if info.get("reward_components"):
                total_energy += info["reward_components"].get("r_energy_penalty", 0.0)

    # 2e. 从环境中提取最终指标
    drl_makespan = info.get('current_time', 0)
    print(f"[DRL] 模型完工时间 (Makespan): {drl_makespan:.4f}s")

    # 提取 DRL 调度数据用于绘图
    gantt_data = []
    deadline_misses = 0
    final_task_map = env.task_map  #

    for task_id, task in final_task_map.items():
        #
        gantt_data.append((task.name, task.assigned_proc, task.start_time, task.finish_time))

        # 计算 DRL 的截止日期未达标情况
        if task.deadline != float('inf') and task.finish_time > task.deadline:
            deadline_misses += 1
            print(
                f"  [!] 截止日期未达标: {task.name} (HI) 在 {task.finish_time:.2f}s 完成 (截止日期: {task.deadline:.2f}s)")

    print(f"[DRL] 截止日期未达标: {deadline_misses}")
    print(f"[DRL] 总能耗 (代理值): {total_energy:.2f}")

    plot_gantt_chart(gantt_data, proc_spec, "DRL (PPO) 调度", "gantt_drl.png")

    return {"makespan": drl_makespan, "deadline_misses": deadline_misses, "total_energy": total_energy}


def main():
    dag_spec, proc_spec = get_problem_specs()

    # 评估基准
    heft_results = evaluate_heft_baseline(dag_spec, proc_spec)

    # 评估 DRL 模型
    drl_results = evaluate_drl_model(dag_spec, proc_spec)

    # 最终对比
    print("\n" + "=" * 40)
    print("--- 最终评估对比 ---")
    print(f"                       | HEFT (基准)       | DRL (PPO) 模型")
    print(f"-----------------------|-------------------|------------------")
    if drl_results:
        print(f"完工时间 (Makespan)    | {heft_results['makespan']:<17.4f} | {drl_results['makespan']:<16.4f}")
        print(f"截止日期未达标         | {heft_results['deadline_misses']:<17} | {drl_results['deadline_misses']:<16}")
        print(f"能耗 (代理值)          | {heft_results['total_energy']:<17} | {drl_results['total_energy']:<16.2f}")
    else:
        print("执行 DRL 评估时出错。")
    print("=" * 40)
    print("\n甘特图文件已生成: 'gantt_heft.png' 和 'gantt_drl.png'")


if __name__ == "__main__":
    main()