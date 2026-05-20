"""
模型评估与对比分析脚本。

功能:
  1. 加载训练好的模型，在测试/验证集上进行推理
  2. 计算准确率、精确率、召回率、F1-score
  3. 绘制混淆矩阵
  4. 对比多模型性能（含推理速度）
  5. 生成 LaTeX/Markdown 格式的结果表格

使用方法:
    python evaluate.py                         # 评估所有已训练模型
    python evaluate.py --model mobilenetv3     # 评估指定模型
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

import config
from dataset import get_dataloaders, get_val_transforms
from models import create_model, count_parameters


"文字显示问题"
matplotlib.use("Agg")
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False



def load_trained_model(model_name: str, device: torch.device):
    """从 checkpoint 加载训练好的模型。"""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}，请先运行 train.py")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = create_model(model_name, ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["class_names"]


@torch.no_grad()
def predict_all(model, loader, device):
    """对整个数据集进行推理，返回预测标签和真实标签。"""
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def measure_inference_speed(model, device, input_size=(1, 3, 224, 224),
                            warmup=10, runs=100):
    """
    测量模型推理速度（单张图像的平均处理时间）。
    
    推理速度是评估模型部署可行性的关键指标，特别是对于
    移动端和实时应用场景。
    
    Args:
        model: 神经网络模型
        device: 计算设备
        input_size: 输入张量尺寸 (batch, channels, height, width)
        warmup: 预热次数（首次推理较慢，需预热）
        runs: 正式测试次数
    
    Returns:
        float: 平均推理时间（毫秒）
    """
    model.eval()
    dummy = torch.randn(*input_size).to(device)

    for _ in range(warmup):
        model(dummy)

    # GPU需要同步以确保准确计时
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()  # 高精度计时器
    for _ in range(runs):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / runs * 1000  # 转换为毫秒
    return elapsed


def plot_confusion_matrix(cm, class_names, model_name, save_path):
    """绘制混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"混淆矩阵 — {model_name}", fontsize=14)
    fig.colorbar(im, ax=ax)

    cn_names = [config.CLASS_CN.get(c, c) for c in class_names]
    tick_marks = np.arange(len(cn_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(cn_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(cn_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("真实类别")
    ax.set_xlabel("预测类别")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  混淆矩阵已保存: {save_path}")


def plot_training_curves(model_name):
    """绘制训练曲线（损失 & 准确率）。"""
    history_path = os.path.join(config.RESULT_DIR, f"{model_name}_history.json")
    if not os.path.exists(history_path):
        return
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="训练损失")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=3, label="验证损失")
    ax1.set_title(f"{model_name} — 损失曲线")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="训练准确率")
    ax2.plot(epochs, history["val_acc"], "r-o", markersize=3, label="验证准确率")
    ax2.set_title(f"{model_name} — 准确率曲线")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    save_path = os.path.join(config.RESULT_DIR, f"{model_name}_curves.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  训练曲线已保存: {save_path}")


def evaluate_model(model_name, val_loader, class_names, device):
    """完整评估单个深度学习模型。"""
    print(f"\n{'='*60}")
    print(f"  评估模型: {model_name}")
    print(f"{'='*60}")

    model, _ = load_trained_model(model_name, device)
    params = count_parameters(model)
    preds, labels = predict_all(model, val_loader, device)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    speed = measure_inference_speed(model, device)

    cn_names = [config.CLASS_CN.get(c, c) for c in class_names]
    report = classification_report(labels, preds, target_names=cn_names, digits=4)
    print(report)

    cm = confusion_matrix(labels, preds)
    cm_path = os.path.join(config.RESULT_DIR, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, model_name, cm_path)
    plot_training_curves(model_name)

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "inference_ms": speed,
        "params": params["total"],
    }


def plot_comparison(results):
    """绘制多模型对比柱状图。"""
    names = [r["model"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_cn = {"accuracy": "准确率", "precision": "精确率",
                 "recall": "召回率", "f1_score": "F1 分数"}

    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=metric_cn[metric])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("模型")
    ax.set_ylabel("分数")
    ax.set_title("模型性能对比")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    save_path = os.path.join(config.RESULT_DIR, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存: {save_path}")


def print_comparison_table(results):
    """打印 Markdown 格式的对比表格。"""
    print("\n## 模型性能对比表\n")
    print("| 模型 | 准确率 | 精确率 | 召回率 | F1 分数 | 推理速度(ms) | 参数量 |")
    print("|------|--------|--------|--------|---------|-------------|--------|")
    for r in results:
        params = f"{r['params']:,}" if isinstance(r['params'], int) else r['params']
        print(f"| {r['model']:<18} "
              f"| {r['accuracy']:.4f} "
              f"| {r['precision']:.4f} "
              f"| {r['recall']:.4f} "
              f"| {r['f1_score']:.4f}  "
              f"| {r['inference_ms']:.2f}       "
              f"| {params} |")


def main():
    parser = argparse.ArgumentParser(description="水稻病害识别 — 模型评估")
    parser.add_argument("--model", type=str, default="all",
                        choices=config.AVAILABLE_MODELS + ["all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    _, val_loader, test_loader, class_names = get_dataloaders()
    eval_loader = test_loader if test_loader is not None else val_loader

    model_list = config.AVAILABLE_MODELS if args.model == "all" else [args.model]
    results = []

    for m in model_list:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{m}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"[跳过] {m}（未找到权重文件）")
            continue
        r = evaluate_model(m, eval_loader, class_names, device)
        results.append(r)


    if results:
        print_comparison_table(results)
        plot_comparison(results)
        report_path = os.path.join(config.RESULT_DIR, "evaluation_report.json")
        serializable = []
        for r in results:
            sr = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                  for k, v in r.items()}
            serializable.append(sr)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\n评估报告已保存: {report_path}")


if __name__ == "__main__":
    main()
