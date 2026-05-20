"""
快速开始示例脚本

本脚本演示如何使用水稻病害识别系统的核心功能。
运行前请确保:
1. 已安装所有依赖包 (pip install -r requirements.txt)
2. 已准备好数据集 (放在 data/rice_disease/ 目录下)
3. 已训练好模型 (运行 python main.py train --model mobilenetv3)
"""

import os
import sys

import config


def example_1_check_environment():
    """示例1: 检查环境配置"""
    print("=" * 60)
    print("示例1: 检查环境配置")
    print("=" * 60)
    
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    import config
    print(f"\n数据集目录: {config.DATA_DIR}")
    print(f"类别数量: {config.NUM_CLASSES}")
    print(f"类别列表: {config.CLASS_NAMES_EN}")
    print(f"默认模型: {config.DEFAULT_MODEL}")
    print()


def example_2_inspect_dataset():
    """示例2: 检查数据集结构"""
    print("=" * 60)
    print("示例2: 检查数据集结构")
    print("=" * 60)
    
    import config
    
    for split in ['train', 'val', 'test']:
        split_dir = getattr(config, f"{split.upper()}_DIR")
        if os.path.isdir(split_dir):
            classes = [d for d in os.listdir(split_dir) 
                      if os.path.isdir(os.path.join(split_dir, d))]
            total_images = sum(
                len([f for f in os.listdir(os.path.join(split_dir, c)) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                for c in classes
            )
            print(f"{split:>5}集: {len(classes)}个类别, {total_images}张图片")
            if classes:
                print(f"       类别: {', '.join(classes[:3])}{'...' if len(classes) > 3 else ''}")
        else:
            print(f"{split:>5}集: 目录不存在 ({split_dir})")
    print()


def example_3_load_and_preview_data():
    """示例3: 加载数据并查看样本信息"""
    print("=" * 60)
    print("示例3: 加载数据并查看样本信息")
    print("=" * 60)
    
    try:
        from dataset import get_dataloaders
        
        train_loader, val_loader, test_loader, class_names = get_dataloaders()
        
        print(f"\n训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        if test_loader:
            print(f"测试集批次数: {len(test_loader)}")
        
        # 查看一个批次的数据
        images, labels = next(iter(train_loader))
        print(f"\n批次形状: images={images.shape}, labels={labels.shape}")
        print(f"标签范围: [{labels.min()}, {labels.max()}]")
        print(f"类别名称: {class_names}")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先准备数据集或下载公开数据集")
    print()


def example_4_create_and_inspect_model():
    """示例4: 创建模型并查看参数信息"""
    print("=" * 60)
    print("示例4: 创建模型并查看参数信息")
    print("=" * 60)
    
    from models import create_model, count_parameters
    import config
    
    for model_name in config.AVAILABLE_MODELS:
        model = create_model(model_name, config.NUM_CLASSES, pretrained=False)
        params = count_parameters(model)
        
        print(f"\n{model_name}:")
        print(f"  总参数: {params['total']:,}")
        print(f"  可训练: {params['trainable']:,}")
        print(f"  冻结: {params['frozen']:,}")
        
        # 计算模型大小（MB）
        model_size = params['total'] * 4 / (1024 ** 2)  # 假设float32
        print(f"  模型大小: ~{model_size:.2f} MB")
    print()


def example_5_quick_prediction():
    """示例5: 快速预测示例（需要已训练的模型）"""
    print("=" * 60)
    print("示例5: 快速预测示例")
    print("=" * 60)
    
    try:
        from predict import RiceDiseasePredictor
        from PIL import Image
        import numpy as np
        
        # 创建预测器
        predictor = RiceDiseasePredictor(model_name="mobilenetv3")
        
        # 创建一个测试图像（全黑图像作为示例）
        # 实际使用时应替换为真实的水稻叶片图片
        test_image = Image.new('RGB', (224, 224), color=(100, 150, 100))
        
        print("\n对测试图像进行预测...")
        result = predictor.predict(test_image)
        
        print(f"\n预测结果:")
        print(f"  类别: {result['class_cn']} ({result['class_en']})")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"\n防治建议:")
        print(f"  {result['treatment']}")
        
        print(f"\n各类别概率分布:")
        for cls, prob in sorted(result['all_probs'].items(), key=lambda x: -x[1]):
            cn = config.CLASS_CN.get(cls, cls)
            bar = "█" * int(prob * 30)
            print(f"  {cn:<10} {prob:.4f} {bar}")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先训练模型: python main.py train --model mobilenetv3")
    except Exception as e:
        print(f"预测失败: {e}")
    print()


def example_6_training_workflow():
    """示例6: 展示训练工作流程（不实际执行）"""
    print("=" * 60)
    print("示例6: 训练工作流程说明")
    print("=" * 60)
    
    print("""
完整的训练流程如下:

1. 准备数据集
   - 将图片按类别放入 data/rice_disease/train/子文件夹
   - 可选: 准备独立的验证集和测试集

2. 运行训练命令
   python main.py train --model mobilenetv3

3. 训练过程监控
   - 终端实时显示训练进度和指标
   - TensorBoard可视化: tensorboard --logdir logs

4. 查看训练结果
   - 模型权重: checkpoints/mobilenetv3_best.pth
   - 训练历史: results/mobilenetv3_history.json
   - TensorBoard日志: logs/mobilenetv3/

5. 评估模型性能
   python main.py evaluate --model mobilenetv3

6. 进行预测
   python main.py predict --image test.jpg

提示: 使用 --model all 可同时训练所有模型进行对比
    """)
    print()


def main():
    """运行所有示例"""
    print("\n" + "🌾" * 30)
    print("  水稻病害识别系统 - 快速开始示例")
    print("🌾" * 30 + "\n")
    
    examples = [
        ("检查环境配置", example_1_check_environment),
        ("检查数据集结构", example_2_inspect_dataset),
        ("加载数据预览", example_3_load_and_preview_data),
        ("创建模型检查", example_4_create_and_inspect_model),
        ("快速预测示例", example_5_quick_prediction),
        ("训练流程说明", example_6_training_workflow),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"示例{i}执行出错: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print("示例运行完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 准备数据集到 data/rice_disease/ 目录")
    print("  2. 运行训练: python main.py train --model mobilenetv3")
    print("  3. 评估模型: python main.py evaluate")
    print("  4. 进行预测: python main.py predict --image your_image.jpg")
    print("  5. 启动API: python main.py api")
    print("\n详细说明请查看 README.md 文件\n")


if __name__ == "__main__":
    main()
