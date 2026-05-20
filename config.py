"""
全局配置模块：集中管理数据集路径、训练超参数、模型设置、病害类别信息等。

该配置文件是整个系统的核心配置中心，所有其他模块通过导入此模块获取配置信息。
修改此处配置可统一调整系统行为，无需修改多处代码。
"""
import os

# ──────────────── 路径配置 ────────────────
# 项目根目录（自动检测，无需手动修改）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集根目录及训练/验证/测试子目录
# 数据集结构应为: data/rice_disease/train|val|test/类别名/图片.jpg
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# 模型检查点、日志和结果保存目录
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")  # 保存训练好的模型权重
LOG_DIR = os.path.join(BASE_DIR, "logs")                 # TensorBoard日志目录
RESULT_DIR = os.path.join(BASE_DIR, "results")           # 评估结果、图表保存目录

# ──────────────── 病害类别定义（英文 → 中文映射） ────────────────
# 注意: 数据集中的文件夹名称必须与CLASS_NAMES_EN中的英文名一致
# 系统会自动从训练集文件夹读取类别，此处用于中文显示和防治建议匹配
CLASS_NAMES_EN = [
    "bacterial_leaf_blight",  # 白叶枯病 - 细菌性病害，叶片出现水渍状条斑
    "brown_spot",              # 褐斑病 - 真菌性病害，叶片出现褐色椭圆形病斑
    "healthy",                 # 健康植株 - 无病害症状
    "leaf_blast",              # 叶瘟病 - 稻瘟病的一种，叶片出现梭形病斑
    "leaf_scald",              # 叶烧病 - 细菌性病害，叶片边缘出现黄白色干枯
    "narrow_brown_spot",       # 窄褐条斑病 - 真菌性病害，叶片出现细长褐色条纹
    "neck_blast",              # 颈瘟病 - 稻瘟病的一种，危害穗颈部位
    "rice_hispa",              # 水稻铁甲虫 - 虫害，叶片出现白色条斑
    "sheath_blight",           # 纹枯病 - 真菌性病害，叶鞘出现云纹状病斑
    "tungro",                  # 东格鲁病 - 病毒病害，植株矮化叶片橙黄
]

# 英文名称到中文名称的映射字典，用于结果展示
CLASS_CN = {
    "bacterial_leaf_blight": "白叶枯病",
    "brown_spot":             "褐斑病",
    "healthy":                "健康",
    "leaf_blast":             "叶瘟病",
    "leaf_scald":             "叶烧病",
    "narrow_brown_spot":      "窄褐条斑病",
    "neck_blast":             "颈瘟病",
    "rice_hispa":             "水稻铁甲虫",
    "sheath_blight":          "纹枯病",
    "tungro":                 "东格鲁病",
}

# 类别总数（自动计算，勿手动修改）
NUM_CLASSES = len(CLASS_NAMES_EN)

# ──────────────── 病害防治建议知识库 ────────────────
# 针对每种病害提供3条主要防治措施，用于预测结果输出
# 这些建议基于农业专家知识和相关文献整理
TREATMENT = {
    "bacterial_leaf_blight": "1. 选用抗病品种；2. 合理施肥，避免偏施氮肥；3. 发病初期喷施噻菌铜或叶枯唑。",
    "brown_spot":            "1. 增施钾肥，改善土壤通透性；2. 发病初期喷施三环唑或稻瘟灵；3. 及时排水晒田。",
    "healthy":               "植株健康，请继续保持良好的田间管理。",
    "leaf_blast":            "1. 选用抗瘟品种；2. 合理密植，保持通风；3. 发病初期喷施三环唑或稻瘟灵。",
    "leaf_scald":            "1. 避免过量施氮肥；2. 保持田间适当水层；3. 发病初期喷施井冈霉素。",
    "narrow_brown_spot":     "1. 增施有机肥与钾肥；2. 合理灌溉，避免深水淹灌；3. 发病初期喷施多菌灵。",
    "neck_blast":            "1. 选用抗瘟品种；2. 种子消毒处理；3. 破口期喷施三环唑预防穗颈瘟。",
    "rice_hispa":            "1. 清除田间杂草；2. 保护天敌昆虫；3. 发生严重时喷施杀虫剂如吡虫啉。",
    "sheath_blight":         "1. 合理密植，保持通风透光；2. 避免过量施氮肥；3. 发病初期喷施井冈霉素或噻呋酰胺。",
    "tungro":                "1. 种植抗病品种；2. 防治传毒媒介叶蝉；3. 拔除病株，减少毒源。",
}

# ──────────────── 训练超参数配置 ────────────────
# 图像预处理参数
IMAGE_SIZE = 224          # 输入图像尺寸（像素），与预训练模型要求一致
BATCH_SIZE = 32           # 批次大小，根据GPU显存调整（显存小可改为16）
NUM_WORKERS = 4           # 数据加载线程数，建议设置为CPU核心数的一半
EPOCHS = 30               # 最大训练轮数，配合早停机制使用

# 优化器参数 - 采用差异化学习率策略
LEARNING_RATE = 1e-3      # 分类头（全连接层）学习率，较大以快速适应新任务
BACKBONE_LR = 1e-4        # 骨干网络微调学习率，较小以避免破坏预训练特征
WEIGHT_DECAY = 1e-4       # L2正则化系数，防止过拟合
EARLY_STOP_PATIENCE = 7   # 早停耐心值：验证准确率连续7轮未提升则停止训练

# ──────────────── 模型选择配置 ────────────────
# 支持的深度学习模型列表（均使用ImageNet预训练权重进行迁移学习）
AVAILABLE_MODELS = ["resnet50", "mobilenetv3", "efficientnet_b0"]
DEFAULT_MODEL = "mobilenetv3"  # 默认模型，兼顾精度与速度，适合移动端部署

# 各模型特点:
# - ResNet50: 经典深度残差网络，精度高但参数量大（~25M），推理速度较慢
# - MobileNetV3-Small: 轻量级网络，参数量少（~2.5M），推理速度快，适合移动端
# - EfficientNet-B0: 复合缩放网络，精度与效率的最佳平衡（~5M参数）

# ──────────────── Web API 服务配置 ────────────────
API_HOST = "0.0.0.0"  # 监听地址（0.0.0.0表示允许外部访问）
API_PORT = 5000       # 服务端口号

# ──────────────── 自动创建必要目录 ────────────────
# 确保模型检查点、日志和结果目录存在，避免运行时错误
for _d in [CHECKPOINT_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(_d, exist_ok=True)
