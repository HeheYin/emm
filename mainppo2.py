# ppo_evaluation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from mainppo import (
    PPOEmbeddedScheduler, PPOSchedulingEnvironment,
    BaselineScheduler, EmbeddedDatasetGenerator,
    simulate_baseline_schedule, plot_gantt_comparison
)
from experiment import DetailedSchedulerVisualizer
from config import *


class PPOEvaluator:
    """PPO模型评估器"""

    def __init__(self, model_path, num_test_dags=50):
        self.model_path = model_path
        self.num_test_dags = num_test_dags
        self.dataset = None
        self.ppo_scheduler = None
        self.env = None
        self.visualizer = DetailedSchedulerVisualizer()

    def load_model(self):
        """加载训练好的PPO模型"""
        print("===== 加载PPO模型 =====")

        # 生成测试数据集
        self.dataset = EmbeddedDatasetGenerator.generate_dataset()

        # 初始化PPO调度器
        sample_dag = self.dataset["split"]["工业控制"][0]
        sample_state_dim = len(PPOEmbeddedScheduler(0, 0).get_state_representation(
            sample_dag, np.zeros(HARDWARE_NUM)))
        action_dim = HARDWARE_NUM

        self.ppo_scheduler = PPOEmbeddedScheduler(sample_state_dim, action_dim)
        self.ppo_scheduler.load_model(self.model_path)

        # 初始化环境
        self.env = PPOSchedulingEnvironment(self.dataset)
        self.env.ppo_scheduler = self.ppo_scheduler

        print(f"模型加载成功: {self.model_path}")
        print(f"状态维度: {sample_state_dim}, 动作维度: {action_dim}")

    def evaluate_ppo_on_multiple_dags(self):
        """在多个DAG上评估PPO性能"""
        print(f"\n===== 在{self.num_test_dags}个DAG上评估PPO性能 =====")

        all_results = {
            'makespans': [],
            'energies': [],
            'deadline_rates': [],
            'load_balances': [],
            'utilizations': []
        }

        test_dags = []
        for task_type in TASK_TYPES:
            dags = self.dataset["split"][task_type]
            # 每个任务类型选择部分DAG
            num_per_type = max(1, self.num_test_dags // len(TASK_TYPES))
            test_dags.extend(dags[:num_per_type])

        # 如果DAG数量不足，补充随机选择
        if len(test_dags) < self.num_test_dags:
            remaining = self.num_test_dags - len(test_dags)
            all_dags = []
            for task_type in TASK_TYPES:
                all_dags.extend(self.dataset["split"][task_type])
            test_dags.extend(random.sample(all_dags, min(remaining, len(all_dags))))

        progress_bar = tqdm(test_dags, desc="评估PPO调度")

        for dag in progress_bar:
            result = self._evaluate_single_dag(dag)

            all_results['makespans'].append(result['makespan'])
            all_results['energies'].append(result['energy'])
            all_results['deadline_rates'].append(result['deadline_rate'])
            all_results['load_balances'].append(result['load_balance'])
            all_results['utilizations'].append(result['utilization'])

            progress_bar.set_postfix({
                'Makespan': f"{result['makespan']:.1f}",
                'Energy': f"{result['energy']:.1f}"
            })

        # 计算平均指标
        avg_results = {
            'makespan': np.mean(all_results['makespans']),
            'energy': np.mean(all_results['energies']),
            'deadline_rate': np.mean(all_results['deadline_rates']),
            'load_balance': np.mean(all_results['load_balances']),
            'utilization': np.mean(all_results['utilizations'])
        }

        print("\nPPO多DAG评估结果:")
        print(f"平均Makespan: {avg_results['makespan']:.2f} ms")
        print(f"平均能耗: {avg_results['energy']:.2f} J")
        print(f"截止时间满足率: {avg_results['deadline_rate']:.2%}")
        print(f"负载均衡度: {avg_results['load_balance']:.4f}")
        print(f"硬件利用率: {avg_results['utilization']:.3f}")

        return avg_results, all_results

    def _evaluate_single_dag(self, dag):
        """评估单个DAG"""
        state, _ = self.env.reset()
        self.env.current_dag = dag
        current_load_states = np.zeros(HARDWARE_NUM)

        # 选择动作
        task_ids = sorted(dag.task_nodes.keys())
        actions = []

        for task_idx, task_id in enumerate(task_ids):
            task_comp_costs = dag.comp_matrix[task_idx]
            action, _ = self.ppo_scheduler.select_action(state, task_comp_costs, epsilon=0.0)
            actions.append(action)

        # 执行完整调度
        done = False
        while not done:
            state, reward, done, info = self.env.step(actions, current_load_states)

        # 计算指标
        makespan = info['makespan']
        energy = info['energy']

        # 计算截止时间满足率
        deadline_met = 0
        for task_id, finish_time in self.env.task_finish_times.items():
            task = dag.task_nodes[task_id]
            if finish_time <= task.deadline:
                deadline_met += 1
        deadline_rate = deadline_met / len(task_ids) if task_ids else 0

        # 计算负载均衡
        hw_times = [self.env.hw_timelines[i]['next_available'] for i in range(HARDWARE_NUM)]
        if np.mean(hw_times) > 0:
            load_balance = 1.0 - np.std(hw_times) / np.mean(hw_times)
        else:
            load_balance = 0

        # 计算硬件利用率
        total_used_time = sum(hw_times)
        total_possible_time = makespan * HARDWARE_NUM
        utilization = total_used_time / total_possible_time if total_possible_time > 0 else 0

        return {
            'makespan': makespan,
            'energy': energy,
            'deadline_rate': deadline_rate,
            'load_balance': load_balance,
            'utilization': utilization
        }


class ComparativeAnalyzer:
    """对比分析器"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.results = {}

    def evaluate_baselines_on_multiple_dags(self, num_dags=50):
        """在多个DAG上评估基线算法"""
        print(f"\n===== 在{num_dags}个DAG上评估基线算法 =====")

        baseline_algorithms = ["HEFT", "RM", "EDF"]

        # 收集测试DAG
        test_dags = []
        for task_type in TASK_TYPES:
            dags = self.dataset["split"][task_type]
            num_per_type = max(1, num_dags // len(TASK_TYPES))
            test_dags.extend(dags[:num_per_type])

        if len(test_dags) < num_dags:
            remaining = num_dags - len(test_dags)
            all_dags = []
            for task_type in TASK_TYPES:
                all_dags.extend(self.dataset["split"][task_type])
            test_dags.extend(random.sample(all_dags, min(remaining, len(all_dags))))

        # 为每个算法初始化结果存储
        for algo in baseline_algorithms:
            self.results[algo] = {
                'makespans': [], 'energies': [],
                'deadline_rates': [], 'load_balances': [], 'utilizations': []
            }

        # 评估每个算法
        for algo in baseline_algorithms:
            print(f"评估{algo}算法...")
            algo_results = self.results[algo]

            for dag in tqdm(test_dags, desc=f"{algo}"):
                schedule_result = simulate_baseline_schedule(dag, algo)

                # 计算指标
                makespan = schedule_result["makespan"]
                energy = schedule_result["total_energy"]

                # 计算截止时间满足率
                deadline_met = 0
                task_ids = sorted(dag.task_nodes.keys())
                for task_id in task_ids:
                    task = dag.task_nodes[task_id]
                    if task_id in schedule_result["task_schedule"]:
                        finish_time = schedule_result["task_schedule"][task_id][2]
                        if finish_time <= task.deadline:
                            deadline_met += 1
                deadline_rate = deadline_met / len(task_ids) if task_ids else 0

                # 计算负载均衡
                hw_times = [schedule_result["hardware_usage"][i]["total_time"]
                            for i in range(HARDWARE_NUM)]
                if np.mean(hw_times) > 0:
                    load_balance = 1.0 - np.std(hw_times) / np.mean(hw_times)
                else:
                    load_balance = 0

                # 计算硬件利用率
                total_used_time = sum(hw_times)
                total_possible_time = makespan * HARDWARE_NUM
                utilization = total_used_time / total_possible_time if total_possible_time > 0 else 0

                algo_results['makespans'].append(makespan)
                algo_results['energies'].append(energy)
                algo_results['deadline_rates'].append(deadline_rate)
                algo_results['load_balances'].append(load_balance)
                algo_results['utilizations'].append(utilization)

            # 计算平均指标
            avg_results = {
                'makespan': np.mean(algo_results['makespans']),
                'energy': np.mean(algo_results['energies']),
                'deadline_rate': np.mean(algo_results['deadline_rates']),
                'load_balance': np.mean(algo_results['load_balances']),
                'utilization': np.mean(algo_results['utilizations'])
            }

            self.results[algo]['average'] = avg_results

            print(f"{algo}算法评估完成:")
            print(f"  - 平均Makespan: {avg_results['makespan']:.2f} ms")
            print(f"  - 平均能耗: {avg_results['energy']:.2f} J")
            print(f"  - 截止时间满足率: {avg_results['deadline_rate']:.2%}")
            print(f"  - 负载均衡度: {avg_results['load_balance']:.4f}")
            print(f"  - 硬件利用率: {avg_results['utilization']:.3f}")

    def add_ppo_results(self, ppo_avg_results, ppo_all_results):
        """添加PPO结果到对比分析"""
        self.results['PPO'] = {
            'average': ppo_avg_results,
            'all': ppo_all_results
        }

    def plot_comprehensive_comparison(self):
        """绘制综合对比图"""
        algorithms = [algo for algo in self.results.keys() if algo != 'PPO'] + ['PPO']

        # 准备数据
        metrics_data = {
            'Makespan (ms)': [self.results[algo]['average']['makespan'] for algo in algorithms],
            'Energy (J)': [self.results[algo]['average']['energy'] for algo in algorithms],
            'Deadline Rate': [self.results[algo]['average']['deadline_rate'] for algo in algorithms],
            'Load Balance': [self.results[algo]['average']['load_balance'] for algo in algorithms],
            'Utilization': [self.results[algo]['average']['utilization'] for algo in algorithms]
        }

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        colors = ['lightblue', 'lightgreen', 'lightcoral', 'orange']

        # 绘制每个指标的对比
        metrics = list(metrics_data.keys())
        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break

            ax = axes[idx]
            values = metrics_data[metric]

            bars = ax.bar(algorithms, values, color=colors[:len(algorithms)], alpha=0.8)
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}' if metric in ['Load Balance', 'Utilization'] else f'{height:.1f}',
                        ha='center', va='bottom', fontweight='bold')

        # 第六个子图：性能提升百分比（相对于HEFT）
        ax = axes[5]
        heft_makespan = self.results['HEFT']['average']['makespan']
        heft_energy = self.results['HEFT']['average']['energy']

        improvements = []
        for algo in algorithms:
            if algo == 'HEFT':
                improvements.append(0)
            else:
                # 综合提升：(Makespan提升 + 能耗降低) / 2
                makespan_improve = (heft_makespan - self.results[algo]['average']['makespan']) / heft_makespan
                energy_improve = (heft_energy - self.results[algo]['average']['energy']) / heft_energy
                total_improve = (makespan_improve + energy_improve) / 2 * 100
                improvements.append(total_improve)

        bars = ax.bar(algorithms, improvements, color=colors[:len(algorithms)], alpha=0.8)
        ax.set_title('Overall Improvement vs HEFT (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('comprehensive_algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_statistical_comparison(self):
        """绘制统计分布对比图（箱线图）"""
        algorithms = [algo for algo in self.results.keys() if algo != 'PPO'] + ['PPO']

        # 准备数据
        metrics = ['makespans', 'energies', 'load_balances']
        metric_names = ['Makespan (ms)', 'Energy (J)', 'Load Balance']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            data = []

            for algo in algorithms:
                if algo == 'PPO':
                    data.append(self.results[algo]['all'][metric])
                else:
                    data.append(self.results[algo][metric])

            box_plot = ax.boxplot(data, labels=algorithms, patch_artist=True)

            # 设置颜色
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'orange']
            for patch, color in zip(box_plot['boxes'], colors[:len(algorithms)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
            ax.set_ylabel(name)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('statistical_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_evaluation(model_path='ppo_scheduler_final.pth', num_dags=50):
    """运行综合评估"""
    print("===== 开始综合评估 =====")

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 1. 评估PPO模型
    ppo_evaluator = PPOEvaluator(model_path, num_dags)
    ppo_evaluator.load_model()
    ppo_avg_results, ppo_all_results = ppo_evaluator.evaluate_ppo_on_multiple_dags()

    # 2. 评估基线算法
    analyzer = ComparativeAnalyzer(ppo_evaluator.dataset)
    analyzer.evaluate_baselines_on_multiple_dags(num_dags)

    # 3. 添加PPO结果并对比
    analyzer.add_ppo_results(ppo_avg_results, ppo_all_results)

    # 4. 绘制综合对比图
    analyzer.plot_comprehensive_comparison()
    analyzer.plot_statistical_comparison()

    # 5. 选择代表性DAG进行详细对比
    print("\n===== 选择代表性DAG进行详细对比 =====")
    representative_dags = []
    for task_type in TASK_TYPES:
        dags = ppo_evaluator.dataset["split"][task_type]
        if dags:
            representative_dags.append(dags[0])  # 每个类型选第一个

    for i, dag in enumerate(representative_dags):
        print(f"\n对比DAG {i + 1}: {dag.dag_id} ({dag.task_type})")
        compare_schedules_on_representative_dag(dag, ppo_evaluator.ppo_scheduler, ppo_evaluator.dataset)

    return analyzer.results


def compare_schedules_on_representative_dag(dag, ppo_scheduler, dataset):
    """在代表性DAG上对比所有算法"""
    print(f"在DAG {dag.dag_id} ({dag.task_type}) 上进行详细算法对比")

    # 收集所有算法的调度结果
    all_schedules = {}

    # PPO调度
    env = PPOSchedulingEnvironment(dataset)
    env.ppo_scheduler = ppo_scheduler
    state, _ = env.reset()
    env.current_dag = dag

    current_load_states = np.zeros(HARDWARE_NUM)
    task_ids = sorted(dag.task_nodes.keys())
    actions = []

    # 选择动作
    for task_idx, task_id in enumerate(task_ids):
        task_comp_costs = dag.comp_matrix[task_idx]
        action, _ = ppo_scheduler.select_action(state, task_comp_costs, epsilon=0.0)
        actions.append(action)

    # 执行调度
    done = False
    while not done:
        state, reward, done, info = env.step(actions, current_load_states)

    # 构建PPO调度结果
    ppo_schedule = {
        "algorithm": "PPO",
        "makespan": info['makespan'],
        "total_energy": info['energy'],
        "task_schedule": {},
        "hardware_usage": {i: {"total_time": env.hw_timelines[i]['next_available'],
                               "tasks": env.hw_timelines[i]['tasks']}
                           for i in range(HARDWARE_NUM)}
    }

    for task_id, finish_time in env.task_finish_times.items():
        task_idx = task_ids.index(task_id)
        hw_idx = actions[task_idx]
        ppo_schedule["task_schedule"][task_id] = (
            hw_idx, env.task_start_times[task_id], finish_time
        )

    all_schedules["PPO"] = ppo_schedule

    # 基线算法调度
    baseline_algorithms = ["HEFT", "RM", "EDF"]

    for algo in baseline_algorithms:
        schedule_result = simulate_baseline_schedule(dag, algo)
        schedule_result["algorithm"] = algo
        all_schedules[algo] = schedule_result

    # 计算各项指标
    algorithms = list(all_schedules.keys())
    metrics = {
        'makespan': [],
        'energy': [],
        'utilization': [],
        'task_distribution': []
    }

    for algo in algorithms:
        schedule = all_schedules[algo]
        metrics['makespan'].append(schedule['makespan'])
        metrics['energy'].append(schedule['total_energy'])

        # 计算硬件利用率
        hw_times = [schedule['hardware_usage'][i]['total_time'] for i in range(HARDWARE_NUM)]
        total_used_time = sum(hw_times)
        total_possible_time = schedule['makespan'] * HARDWARE_NUM
        utilization = total_used_time / total_possible_time if total_possible_time > 0 else 0
        metrics['utilization'].append(utilization)

        # 计算任务分布均匀度
        task_counts = [len(schedule['hardware_usage'][i].get('tasks', [])) for i in range(HARDWARE_NUM)]
        if np.mean(task_counts) > 0:
            distribution_balance = 1 - np.std(task_counts) / np.mean(task_counts)
        else:
            distribution_balance = 0
        metrics['task_distribution'].append(distribution_balance)

    # 绘制对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Makespan对比
    bars1 = ax1.bar(algorithms, metrics['makespan'], color=['blue', 'green', 'red', 'orange'])
    ax1.set_title(f'Makespan Comparison - DAG {dag.dag_id} ({dag.task_type})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Makespan (ms)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}',
                 ha='center', va='bottom', fontweight='bold')

    # 能耗对比
    bars2 = ax2.bar(algorithms, metrics['energy'], color=['blue', 'green', 'red', 'orange'])
    ax2.set_title(f'Energy Consumption - DAG {dag.dag_id} ({dag.task_type})', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Energy (J)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}',
                 ha='center', va='bottom', fontweight='bold')

    # 硬件利用率对比
    bars3 = ax3.bar(algorithms, metrics['utilization'], color=['blue', 'green', 'red', 'orange'])
    ax3.set_title(f'Hardware Utilization - DAG {dag.dag_id} ({dag.task_type})', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Utilization Rate', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}',
                 ha='center', va='bottom', fontweight='bold')

    # 任务分布对比
    bars4 = ax4.bar(algorithms, metrics['task_distribution'], color=['blue', 'green', 'red', 'orange'])
    ax4.set_title(f'Task Distribution Balance - DAG {dag.dag_id} ({dag.task_type})', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Balance Score', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}',
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'detailed_comparison_dag{dag.dag_id}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 输出详细结果
    print(f"\nDAG {dag.dag_id} ({dag.task_type}) 详细对比结果:")
    print("=" * 60)
    for algo in algorithms:
        schedule = all_schedules[algo]
        idx = algorithms.index(algo)
        print(f"\n{algo}:")
        print(f"  - Makespan: {schedule['makespan']:.2f} ms")
        print(f"  - 总能耗: {schedule['total_energy']:.2f} J")
        print(f"  - 硬件利用率: {metrics['utilization'][idx]:.3f}")
        print(f"  - 任务分布均衡度: {metrics['task_distribution'][idx]:.3f}")

        # 显示任务分配情况
        hw_task_counts = [len(schedule['hardware_usage'][i].get('tasks', []))
                          for i in range(HARDWARE_NUM)]
        print(f"  - 硬件任务分配: {hw_task_counts}")

    return all_schedules


if __name__ == "__main__":
    # 运行综合评估
    results = run_comprehensive_evaluation(
        model_path='ppo_scheduler_final.pth',  # 替换为您的模型路径
        num_dags=200  # 测试DAG数量
    )

    print("\n===== 评估完成 =====")
    print("生成的图表:")
    print("1. comprehensive_algorithm_comparison.png - 综合算法对比")
    print("2. statistical_distribution_comparison.png - 统计分布对比")
    print("3. detailed_comparison_dagX.png - 各代表性DAG详细对比")