# 水稻病害识别系统 - 基于深度学习的智能诊断平台

## 项目简介

本系统是基于深度学习的水稻病害智能识别平台，能够自动识别6种常见水稻病害（包括健康植株），并提供相应的防治建议。系统采用迁移学习技术，对比了ResNet50、MobileNetV3和EfficientNet-B0三种深度学习模型，并与传统SVM方法进行了性能对比实验。

**作者**: 廖衡林  
**学院**: 信息学院  
**专业**: 智能科学与技术  
**指导老师**: 胡茂  

---

## 功能特点

### 1. 多模型支持
- **ResNet50**: 经典深度残差网络，精度高（~25M参数）
- **MobileNetV3-Small**: 轻量级网络，推理速度快（~2.5M参数），适合移动端部署
- **EfficientNet-B0**: 复合缩放网络，精度与效率的最佳平衡（~5M参数）

### 2. 完整工作流程
- ✅ 数据集加载与增强
- ✅ 模型训练（支持早停、学习率调度）
- ✅ 模型评估（准确率、精确率、召回率、F1分数）
- ✅ 混淆矩阵可视化
- ✅ 与传统SVM基线对比
- ✅ 单张图像预测
- ✅ Web API服务

### 3. 智能诊断
- 输出预测类别及置信度
- 显示所有类别的概率分布
- 提供针对性的防治建议

---

## 环境要求

```bash
Python >= 3.8
PyTorch >= 1.9.0
torchvision >= 0.10.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0
Pillow >= 8.0.0
tensorboard >= 2.5.0
scikit-image >= 0.18.0  # SVM基线需要
flask >= 2.0.0          # API服务需要
requests >= 2.25.0      # URL预测需要
```

### 快速安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn matplotlib Pillow tensorboard scikit-image flask requests
```

---

## 数据集准备

### 目录结构

```
data/rice_disease/
├── train/
│   ├── bacterial_blight/    # 白叶枯病
│   │   ├── img1.jpg
│   │   └── ...
│   ├── brown_spot/          # 褐斑病
│   ├── healthy/             # 健康
│   ├── leaf_blast/          # 叶瘟病
│   ├── leaf_scald/          # 叶烧病
│   └── narrow_brown_spot/   # 窄褐条斑病
├── val/                     # 验证集（可选，结构同上）
└── test/                    # 测试集（可选，结构同上）
```

**注意**: 
- 文件夹名称必须与 `config.py` 中 `CLASS_NAMES_EN` 定义的英文名称一致
- 如果没有独立的val/test目录，系统会自动从train中划分20%作为验证集

### 公开数据集推荐

1. **Rice Disease Dataset** (Kaggle)
2. **PlantVillage Dataset** (包含多种作物病害)
3. 国内农业科研机构发布的水稻病害图像资源

---

## 使用方法

### 1. 训练模型

```bash
# 训练单个模型
python main.py train --model mobilenetv3

# 训练所有模型进行对比
python main.py train --model all

# 自定义训练参数
python main.py train --model resnet50 --epochs 50 --batch-size 16 --lr 0.001
```

**训练过程**:
- 使用ImageNet预训练权重进行迁移学习
- 冻结浅层网络，微调深层网络和分类头
- 采用差异化学习率策略
- 余弦退火学习率调度
- 早停机制防止过拟合
- TensorBoard实时日志记录

### 2. 评估模型

```bash
# 评估所有已训练的模型
python main.py evaluate

# 评估指定模型
python main.py evaluate --model efficientnet_b0

# 与SVM基线对比
python main.py evaluate --with-svm
```

**评估指标**:
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 推理速度 (ms/image)
- 参数量

**输出内容**:
- 混淆矩阵热力图
- 训练曲线图（损失&准确率）
- 多模型性能对比柱状图
- Markdown格式对比表格
- JSON格式详细报告

### 3. 单张图像预测

```bash
# 预测本地图片
python main.py predict --image test.jpg

# 预测网络图片
python main.py predict --image-url http://example.com/rice.jpg

# 指定模型
python main.py predict --image test.jpg --model efficientnet_b0
```

**输出示例**:
```
==================================================
  识别结果: 叶瘟病 (leaf_blast)
  置信度:   92.35%
==================================================

各类别概率:
  叶瘟病       0.9235 ████████████████████████████████████████
  健康         0.0421 █
  白叶枯病     0.0198 
  褐斑病       0.0089 
  叶烧病       0.0037 
  窄褐条斑病   0.0020 

防治建议:
  1. 选用抗瘟品种；2. 合理密植，保持通风；3. 发病初期喷施三环唑或稻瘟灵。
```

### 4. 启动Web API服务

```bash
# 启动API服务（默认使用MobileNetV3）
python main.py api

# 指定模型
python main.py api --model resnet50
```

**API接口**:

1. **健康检查**
```bash
GET http://localhost:5000/health
```

2. **上传图像文件预测**
```bash
POST http://localhost:5000/predict
Content-Type: multipart/form-data

Form Data:
  image: [选择图片文件]
```

3. **Base64编码图像预测**
```bash
POST http://localhost:5000/predict_base64
Content-Type: application/json

{
  "image": "base64_encoded_string"
}
```

---

## 项目结构

```
Py/
├── config.py           # 全局配置（路径、超参数、类别信息等）
├── dataset.py          # 数据集加载与图像增强
├── models.py           # 模型定义（ResNet50, MobileNetV3, EfficientNet-B0）
├── train.py            # 训练脚本
├── evaluate.py         # 评估与对比分析
├── predict.py          # 单张图像预测
├── main.py             # 主程序入口（统一命令行接口）
├── data/               # 数据集目录
│   └── rice_disease/
│       ├── train/
│       ├── val/
│       └── test/
├── checkpoints/        # 保存训练好的模型权重
├── logs/               # TensorBoard日志
├── results/            # 评估结果、图表、报告
└── README.md           # 本文件
```

---

## 技术亮点

### 1. 迁移学习策略
- 使用ImageNet预训练权重初始化
- 冻结浅层网络保留通用特征提取能力
- 微调深层网络适应水稻病害任务
- 差异化学习率：骨干网络1e-4，分类头1e-3

### 2. 数据增强
- 随机缩放裁剪（模拟不同拍摄距离）
- 水平/垂直翻转（增加样本多样性）
- 随机旋转±15°（应对叶片不同角度）
- 颜色抖动（模拟不同光照条件）
- 随机仿射变换（增加几何变换鲁棒性）

### 3. 优化技巧
- AdamW优化器（带权重衰减的Adam）
- 余弦退火学习率调度
- 早停机制（patience=7）
- Dropout正则化（p=0.3）
- L2正则化（weight_decay=1e-4）

### 4. 对比实验设计
- 三种深度学习模型横向对比
- 与传统SVM+HOG方法纵向对比
- 多维度评估指标（精度、速度、参数量）

---

## 实验结果示例

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | 推理速度(ms) | 参数量 |
|------|--------|--------|--------|--------|-------------|--------|
| ResNet50 | 0.9234 | 0.9187 | 0.9201 | 0.9194 | 45.23 | 25,557,030 |
| MobileNetV3 | 0.8956 | 0.8912 | 0.8934 | 0.8923 | 12.67 | 2,542,982 |
| EfficientNet-B0 | 0.9178 | 0.9145 | 0.9162 | 0.9153 | 28.45 | 5,288,548 |
| SVM + HOG | 0.7823 | 0.7756 | 0.7801 | 0.7778 | 8.34 | N/A |

*注：以上为示例数据，实际结果取决于数据集和训练配置*

---

## 常见问题

### Q1: CUDA out of memory 错误
**A**: 减小批次大小，在 `config.py` 中修改 `BATCH_SIZE = 16` 或使用命令行参数 `--batch-size 16`

### Q2: 找不到模型权重文件
**A**: 先运行训练命令生成权重文件，权重保存在 `checkpoints/` 目录下

### Q3: 数据集在哪里下载
**A**: 参考"数据集准备"章节中的公开数据集推荐，或联系指导老师获取

### Q4: 如何添加新的病害类别
**A**: 
1. 在 `config.py` 的 `CLASS_NAMES_EN` 中添加新类别英文名
2. 在 `CLASS_CN` 中添加中文映射
3. 在 `TREATMENT` 中添加防治建议
4. 在数据集中创建对应的子文件夹

### Q5: TensorBoard如何使用
**A**: 
```bash
tensorboard --logdir logs
# 然后在浏览器访问 http://localhost:6006
```

---

## 参考文献

[1] 张永玲. 基于Android的水稻病虫害图像识别与诊断系统的研究[D]. 杭州: 浙江理工大学信息学院, 2018.

[2] 杨孟辑. 基于迁移学习的农作物病虫害预警系统研究[D]. 成都: 成都大学电子信息与电气工程学院, 2021.

[3] 戴子兵. 基于语义分割的水稻病害检测技术[D]. 成都: 西华大学机械工程与自动化学院, 2020.

[4] 涂智潇. 基于计算机视觉的农作物病虫害检测系统[D]. 武汉: 中南财经政法大学信息与安全工程学院, 2020.

[5] 戴晖. 基于图像处理的水稻病虫害检测研究[D]. 南京林业大学, 2022.

---

## 许可证

本项目仅用于学术研究和毕业设计，请勿用于商业用途。

---

## 联系方式

如有问题或建议，请联系：
- 学生：廖衡林
- 指导老师：胡茂
- 学院：信息学院
- 专业：智能科学与技术

---

**祝使用愉快！** 🌾
