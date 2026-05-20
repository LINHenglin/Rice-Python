"""
训练脚本：支持选择不同模型进行训练，包含早停、学习率调度、TensorBoard 日志。

使用方法:
    python train.py --model resnet50
    python train.py --model mobilenetv3
    python train.py --model efficientnet_b0
    python train.py --model all            # 依次训练全部模型
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import get_dataloaders
from models import create_model, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    训练一个完整的epoch（遍历整个训练集一次）。
    
    Args:
        model: 神经网络模型
        loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备（CPU/GPU）
    
    Returns:
        tuple: (平均损失, 准确率)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # 将数据移至指定设备（GPU或CPU）
        images, labels = images.to(device), labels.to(device)
        
        # 清零梯度（PyTorch默认累积梯度，需手动清零）
        optimizer.zero_grad()
        
        # 前向传播：计算模型预测
        outputs = model(images)
        
        # 计算损失：预测值与真实标签的差异
        loss = criterion(outputs, labels)
        
        # 反向传播：计算梯度
        loss.backward()
        
        # 参数更新：根据梯度和学习率调整模型参数
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    在验证集上评估模型性能。
    
    使用@torch.no_grad()装饰器禁用梯度计算，节省内存并加速推理。
    验证阶段不更新模型参数，仅计算损失和准确率。
    
    Args:
        model: 神经网络模型
        loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
    
    Returns:
        tuple: (平均损失, 准确率)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_model(model_name: str, device: torch.device,
                train_loader, val_loader, class_names):
    """训练单个模型并保存最佳权重。"""
    num_classes = len(class_names)
    model = create_model(model_name, num_classes, pretrained=True).to(device)
    params_info = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"  模型: {model_name}")
    print(f"  总参数: {params_info['total']:,}")
    print(f"  可训练: {params_info['trainable']:,}  冻结: {params_info['frozen']:,}")
    print(f"{'='*60}")

    # 差异化学习率策略：骨干网络用小学习率，分类头用大学习率
    # 原因：预训练的骨干网络已学到良好特征，只需微调；
    #      而分类头是随机初始化的，需要更快学习新任务
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "fc" not in n
                       and "classifier" not in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and ("fc" in n or "classifier" in n)]

    # 使用AdamW优化器（带权重衰减的Adam）
    # AdamW比传统Adam有更好的泛化性能
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": config.BACKBONE_LR},  # 骨干网络小学习率
        {"params": head_params, "lr": config.LEARNING_RATE},     # 分类头大学习率
    ], weight_decay=config.WEIGHT_DECAY)

    # 余弦退火学习率调度器：学习率按余弦曲线逐渐减小
    # 有助于模型在训练后期精细调整参数，收敛到更好的局部最优
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    log_dir = os.path.join(config.LOG_DIR, model_name,
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0
    patience_counter = 0
    best_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"[{model_name}] Epoch {epoch:>3}/{config.EPOCHS} "
              f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"| {elapsed:.1f}s")

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 早停机制（Early Stopping）：防止过拟合
        # 如果验证准确率连续多轮未提升，提前终止训练
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型权重（包含类别信息，便于后续加载）
            torch.save({
                "model_name": model_name,
                "num_classes": num_classes,
                "class_names": class_names,
                "state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "epoch": epoch,
            }, best_ckpt)
            print(f"  ✓ 保存最佳模型 (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"  早停触发（连续 {config.EARLY_STOP_PATIENCE} 轮未改善）")
                break

    writer.close()

    history_path = os.path.join(config.RESULT_DIR, f"{model_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[{model_name}] 最佳验证准确率: {best_val_acc:.4f}")
    print(f"[{model_name}] 权重: {best_ckpt}")
    print(f"[{model_name}] 训练历史: {history_path}")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="水稻病害识别 — 模型训练")
    parser.add_argument("--model", type=str, default="all",
                        choices=config.AVAILABLE_MODELS + ["all"],
                        help="要训练的模型名称，all 表示全部训练")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, _, class_names = get_dataloaders()
    model_list = config.AVAILABLE_MODELS if args.model == "all" else [args.model]

    results = {}
    for m in model_list:
        acc = train_model(m, device, train_loader, val_loader, class_names)
        results[m] = acc

    print("\n" + "=" * 60)
    print("  训练结果汇总")
    print("=" * 60)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} 验证准确率: {acc:.4f}")

    summary_path = os.path.join(config.RESULT_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
