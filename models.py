"""
模型定义：提供三种深度学习模型 + 一种传统机器学习基线，用于对比实验。

深度学习模型均采用 ImageNet 预训练权重 + 迁移学习策略：
  - 冻结骨干前几层，微调后面的层和分类头
  - 替换最后的全连接层匹配目标类别数

1. ResNet-50        — 经典深度残差网络，作为深度学习基线
2. MobileNetV3-Small — 轻量级网络，适合移动端/边缘设备部署
3. EfficientNet-B0   — 复合缩放网络，精度与效率的最佳平衡
"""
import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    ResNet-50 迁移学习模型。
    冻结前 6 层（conv1 ~ layer2），微调 layer3/layer4 + 新分类头。
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # 冻结浅层特征提取器（保留ImageNet学到的通用特征）
    # 浅层主要检测边缘、纹理等低级特征，对各类图像任务通用
    frozen = [model.conv1, model.bn1, model.layer1, model.layer2]
    for layer in frozen:
        for param in layer.parameters():
            param.requires_grad = False  # 设置参数不参与梯度更新

    # 替换最后的全连接层，适配新的分类任务
    # ImageNet有1000类，本任务只有num_classes类
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Dropout正则化，防止过拟合（30%概率丢弃神经元）
        nn.Linear(in_features, num_classes),  # 新的分类层
    )
    return model


def build_mobilenetv3(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    构建MobileNetV3-Small轻量级模型。
    
    MobileNetV3采用深度可分离卷积（Depthwise Separable Convolution）
    和逆残差结构（Inverted Residual），大幅减少参数量和计算量，
    专为移动端和边缘设备设计。
    
    特点:
    - 参数量仅约2.5M（ResNet50的1/10）
    - 推理速度快，适合实时应用
    - 精度略低于ResNet50，但差距不大
    
    Args:
        num_classes (int): 目标分类类别数
        pretrained (bool): 是否使用预训练权重
    
    Returns:
        nn.Module: MobileNetV3-Small模型对象
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # 冻结前8个InvertedResidual块（浅层特征提取器）
    # MobileNetV3的特征提取器由多个InvertedResidual模块组成
    for i, block in enumerate(model.features):
        if i < 8:  # 前8个块冻结，后面的块参与微调
            for param in block.parameters():
                param.requires_grad = False

    # 替换分类器的最后一层，适配新任务的类别数
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    构建EfficientNet-B0模型。
    
    EfficientNet采用复合缩放方法（Compound Scaling），同时调整
    网络的深度、宽度和分辨率，在精度和效率之间取得最佳平衡。
    
    特点:
    - 参数量约5M，介于ResNet50和MobileNetV3之间
    - 精度通常优于同参数量的其他模型
    - B0是基础版本，还有B1-B7等更大模型
    
    Args:
        num_classes (int): 目标分类类别数
        pretrained (bool): 是否使用预训练权重
    
    Returns:
        nn.Module: EfficientNet-B0模型对象
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # 冻结前5个MBConv块（EfficientNet的核心组件）
    # MBConv = Mobile Inverted Bottleneck Convolution
    for i, block in enumerate(model.features):
        if i < 5:  # 浅层冻结，深层微调
            for param in block.parameters():
                param.requires_grad = False

    # 替换分类层，适配新任务
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ──────────────── 工厂函数 ────────────────

MODEL_BUILDERS = {
    "resnet50":        build_resnet50,
    "mobilenetv3":     build_mobilenetv3,
    "efficientnet_b0": build_efficientnet_b0,
}


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """按名称创建模型。"""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"不支持的模型: {model_name}，可选: {list(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[model_name](num_classes, pretrained)


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数量（总参数 / 可训练参数）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
