import os

import torch
from mcs_env import MCSSchedulingEnv
from model_hgat import HGATActorCritic
from agent_ppo import PPOAgent
from training_pipeline import (
    generate_expert_trajectories,  # <--- 导入
    train_behavioral_cloning,  # <--- 导入
    train_ppo_finetuning
)
from dag_utils import CRIT_LO, CRIT_HI, PROC_CPU, PROC_NPU

"""
主入口：协调 Sec 4.1 的两阶段训练策略
"""


def main():
    # 1. 定义环境和模型参数 (基于您的蓝图)

    # 示例 DAG 规格 (T4是HI-Mode) [基于 1.1节 的 RCA]
    dag_spec = {
        "task_specs": [
            # {id, name, exec_times{CPU, NPU}, crit, deadline}
            {"id": 0, "name": "T1", "exec_times": {0: 5, 1: 2}, "criticality": CRIT_LO},
            {"id": 1, "name": "T2", "exec_times": {0: 2, 1: 6}, "criticality": CRIT_LO},  # IO密集型 (CPU更快) [RCA 1]
            {"id": 2, "name": "T3", "exec_times": {0: 4, 1: 1}, "criticality": CRIT_LO},
            {"id": 3, "name": "T4", "exec_times": {0: 3, 1: 1}, "criticality": CRIT_HI, "deadline": 10.0},
            # HI-Mode [RCA 2, T4]
            {"id": 4, "name": "T5", "exec_times": {0: 2, 1: 2}, "criticality": CRIT_LO},
            {"id": 5, "name": "T6", "exec_times": {0: 8, 1: 3}, "criticality": CRIT_LO},  # LO-Mode [RCA 2, T6]
        ],
        "dependencies": [
            # (u, v, {data_size})
            (0, 2, {"data_size": 10}),
            (1, 2, {"data_size": 20}),  # T2(CPU) -> T3(NPU) 会产生高迁移
            (2, 3, {"data_size": 5}),
            (2, 4, {"data_size": 15}),
            (3, 5, {"data_size": 10}),  # T4(HI) -> T6(LO)
            (4, 5, {"data_size": 5}),
        ]
    }

    # 处理器规格
    proc_spec = [
        {"id": 0, "type": PROC_CPU},  # CPU
        {"id": 1, "type": PROC_NPU},  # NPU
    ]

    # 2. 初始化环境
    print("初始化 MCS 调度环境...")
    env = MCSSchedulingEnv(dag_spec, proc_spec)

    # 获取 PyG 的元数据 (用于 HGTConv)
    dummy_obs, _ = env.reset()
    metadata = dummy_obs.metadata()
    print("环境元数据:", metadata)

    # 3. 初始化模型和智能体
    print("初始化 HGAT 模型和 PPO 智能体...")
    model = HGATActorCritic(
        hidden_dim=128,
        metadata=metadata
    )

    agent = PPOAgent(
        model=model,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=10,
        entropy_coeff=0.01,
        value_loss_coeff=0.5
    )

    # --- 阶段 1: 行为克隆 (BC) ---
    #

    # 1a. 生成专家轨迹 (如果它们不存在)
    if not os.path.exists("data/expert_trajectories.pth"):
        generate_expert_trajectories(dag_spec, proc_spec, data_path="data/")
    else:
        print("--- [阶段 1 BC] 发现已存在的 'expert_trajectories.pth'。跳过生成。---")

    # 1b. 运行监督学习预训练 [cite: 119]
    model = train_behavioral_cloning(
        model=model,
        data_path="data/",
        epochs=500,  # BC 需要足够的 epochs 来收敛
        batch_size=len(dag_spec["task_specs"]),  # 批处理大小 = 轨迹长度
        lr=1e-3
    )

    print("已加载 (BC) 预训练权重。PPO 将进行热启动。")

    # --- 阶段 2: PPO 微调 ---
    # [cite: 121]
    # (PPO 现在从 BC 预训练的权重大大“热启动”)
    final_model = train_ppo_finetuning(
        model=model,
        env=env,
        agent=agent,
        num_episodes=5000,
        max_steps_per_ep=env.num_tasks * 2
    )

    print("\n--- 训练完成！ ---")
    torch.save(final_model.state_dict(), "model_final_ppo.pth")
    print("最终模型已保存到 model_final_ppo.pth")


if __name__ == "__main__":
    main()