import torch

# ===================== 嵌入式硬件配置 =====================
EMBEDDED_HARDWARES = {
    "CPU": {"算力": 100, "内存": 512, "能耗系数": 0.8, "负载阈值": 0.9, "通信延迟": {"GPU": 5, "FPGA": 10, "MCU": 8}},
    "GPU": {"算力": 800, "内存": 2048, "能耗系数": 2.5, "负载阈值": 0.85, "通信延迟": {"CPU": 5, "FPGA": 3, "MCU": 12}},
    "FPGA": {"算力": 500, "内存": 1024, "能耗系数": 1.2, "负载阈值": 0.9, "通信延迟": {"CPU": 10, "GPU": 3, "MCU": 5}},
    "MCU": {"算力": 50, "内存": 64, "能耗系数": 0.3, "负载阈值": 0.8, "通信延迟": {"CPU": 8, "GPU": 12, "FPGA": 5}}
}
HARDWARE_TYPES = list(EMBEDDED_HARDWARES.keys())
HARDWARE_NUM = len(HARDWARE_TYPES)

# ===================== 模型参数配置 =====================
# 轻量化嵌入层参数
EMBED_DIM = 64  # 嵌入维度
SGC_K = 2  # 简单图卷积层数
GRU_HIDDEN = 64  # GRU隐藏层维度
LOAD_THRESHOLD = 0.7  # 硬件负载剪枝阈值（仅保留负载<阈值的硬件）

# DRL参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
GAMMA = 0.99  # 折扣因子
TAU = 0.005  # 软更新系数
EPS_START = 1.0
EPS_DECAY = 5e-7
REPLAY_BUFFER_SIZE = 10000
PRIORITY_ALPHA = 0.6  # 优先经验回放参数

# ===================== 任务配置 =====================
TASK_TYPES = ["工业控制", "边缘AI", "传感器融合"]
MAX_TASK_NUM = 50  # 单DAG最大任务数
DEADLINE_FACTOR = 1.5  # 截止时间系数（基于任务总计算量）
PERIOD_RANGE = [10, 50]  # 周期任务周期范围（ms）

# ===================== 多目标优化权重 =====================
# 动态权重初始值（可根据场景模式调整）
WEIGHTS = {
    "实时优先": {"makespan": 0.4, "load": 0.2, "energy": 0.1, "reliability": 0.3},
    "能耗优先": {"makespan": 0.2, "load": 0.2, "energy": 0.4, "reliability": 0.2},
    "均衡模式": {"makespan": 0.3, "load": 0.25, "energy": 0.25, "reliability": 0.2}
}
CURRENT_MODE = "均衡模式"

# ===================== 实验配置 =====================
DATASET_SIZE = {
    "工业控制": 150,
    "边缘AI": 150,
    "传感器融合": 150
}
DYNAMIC_ARRIVAL_RATE = {"高峰": 0.1, "平峰": 0.02}  # 任务到达率（个/ms）
BASELINE_ALGORITHMS = ["HEFT", "CPOP", "ADTS", "RM", "EDF", "LLF"]

# ===================== 硬件实测配置 =====================
HARDWARE_PLATFORM = "JetsonNano+STM32+FPGA"
POWER_METER_PATH = "/dev/power_meter"  # 功率计设备路径（模拟）
TEMPERATURE_SENSOR_PATH = "/dev/temp_sensor"  # 温度传感器路径（模拟）

# ===================== 设备配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 嵌入式设备强制使用CPU时注释上一行，启用下一行
# DEVICE = torch.device("cpu")