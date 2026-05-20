"""
水稻病害识别系统 - 主程序入口

功能说明:
    本系统提供完整的水稻病害智能识别解决方案，包括:
    1. 数据集加载与预处理
    2. 多模型训练（ResNet50、MobileNetV3、EfficientNet-B0）
    3. 模型评估与对比分析
    4. 单张图像预测与防治建议
    5. Web API 服务（可选）

使用方法:
    python main.py train --model mobilenetv3           # 训练指定模型
    python main.py train --model all                   # 训练所有模型
    python main.py evaluate                            # 评估模型性能
    python main.py predict --image leaf.png --model efficientnet_b0           # 预测单张图片
    python main.py api                                # 启动Web API服务
"""
import argparse
import sys
import os

# 导入核心模块
import config
from train import train_model
from dataset import get_dataloaders
from evaluate import evaluate_model, print_comparison_table, plot_comparison
from predict import RiceDiseasePredictor


def cmd_train(args):
    """执行模型训练命令。"""
    import torch
    from models import count_parameters
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # 获取数据加载器
    train_loader, val_loader, _, class_names = get_dataloaders()
    
    # 确定要训练的模型列表
    model_list = config.AVAILABLE_MODELS if args.model == "all" else [args.model]
    
    results = {}
    for m in model_list:
        acc = train_model(m, device, train_loader, val_loader, class_names)
        results[m] = acc
    
    # 打印训练结果汇总
    print("\n" + "=" * 60)
    print("  训练结果汇总")
    print("=" * 60)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} 验证准确率: {acc:.4f}")
    
    # 保存训练摘要
    import json
    summary_path = os.path.join(config.RESULT_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n训练摘要已保存: {summary_path}")


def cmd_evaluate(args):
    """执行模型评估命令。"""
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")
    
    # 获取数据加载器（优先使用测试集，否则使用验证集）
    _, val_loader, test_loader, class_names = get_dataloaders()
    eval_loader = test_loader if test_loader is not None else val_loader
    
    # 确定要评估的模型列表
    model_list = config.AVAILABLE_MODELS if args.model == "all" else [args.model]
    results = []
    
    # 评估深度学习模型
    for m in model_list:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{m}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"[跳过] {m}（未找到权重文件，请先训练）")
            continue
        r = evaluate_model(m, eval_loader, class_names, device)
        results.append(r)
    
    # 输出对比结果
    if results:
        print_comparison_table(results)
        plot_comparison(results)
        
        # 保存评估报告
        import json
        import numpy as np
        report_path = os.path.join(config.RESULT_DIR, "evaluation_report.json")
        serializable = []
        for r in results:
            sr = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                  for k, v in r.items()}
            serializable.append(sr)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\n评估报告已保存: {report_path}")


def cmd_predict(args):
    """执行单张图像预测命令。"""
    if not args.image and not args.image_url:
        print("错误: 请提供 --image 或 --image-url 参数")
        sys.exit(1)
    
    # 创建预测器实例
    predictor = RiceDiseasePredictor(model_name=args.model)
    
    # 执行预测
    if args.image:
        result = predictor.predict_from_path(args.image)
    else:
        result = predictor.predict_from_url(args.image_url)
    
    # 格式化输出预测结果
    print(f"\n{'='*50}")
    print(f"  识别结果: {result['class_cn']} ({result['class_en']})")
    print(f"  置信度:   {result['confidence']:.2%}")
    print(f"{'='*50}")
    print(f"\n各类别概率:")
    for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        cn = config.CLASS_CN.get(cls, cls)
        bar = "█" * int(prob * 40)
        print(f"  {cn:<10} {prob:.4f} {bar}")
    print(f"\n防治建议:\n  {result['treatment']}")


def cmd_api(args):
    """启动Web API服务。"""
    try:
        from flask import Flask, request, jsonify
        from PIL import Image
        import io
    except ImportError:
        print("错误: 缺少Flask依赖，请运行: pip install flask")
        sys.exit(1)
    
    app = Flask(__name__)
    predictor = RiceDiseasePredictor(model_name=args.model)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """健康检查接口。"""
        return jsonify({"status": "ok", "model": predictor.model_name})
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """图像预测接口。"""
        if 'image' not in request.files:
            return jsonify({"error": "请上传图像文件"}), 400
        
        try:
            file = request.files['image']
            image_bytes = file.read()
            result = predictor.predict_from_bytes(image_bytes)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/predict_base64', methods=['POST'])
    def predict_base64():
        """Base64编码图像预测接口。"""
        import base64
        try:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({"error": "请提供base64编码的图像"}), 400
            
            image_bytes = base64.b64decode(data['image'])
            result = predictor.predict_from_bytes(image_bytes)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    print(f"\n{'='*60}")
    print(f"  水稻病害识别 API 服务")
    print(f"  模型: {predictor.model_name}")
    print(f"  地址: http://{config.API_HOST}:{config.API_PORT}")
    print(f"  接口:")
    print(f"    GET  /health         - 健康检查")
    print(f"    POST /predict        - 上传图像文件预测")
    print(f"    POST /predict_base64 - Base64图像预测")
    print(f"{'='*60}\n")
    
    app.run(host=config.API_HOST, port=config.API_PORT, debug=False)


def main():
    """主函数：解析命令行参数并执行相应命令。"""
    parser = argparse.ArgumentParser(
        description="水稻病害识别系统 - 基于深度学习的智能诊断平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py train --model mobilenetv3           # 训练MobileNetV3模型
  python main.py train --model all                    # 训练所有模型进行对比
  python main.py evaluate                             # 评估所有已训练模型
  python main.py predict --image test.jpg             # 预测本地图片
  python main.py predict --image-url http://...       # 预测网络图片
  python main.py api                                  # 启动Web API服务
        """
    )
    
    # 定义子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--model", type=str, default="all",
        choices=config.AVAILABLE_MODELS + ["all"],
        help="要训练的模型名称，all表示全部训练"
    )
    train_parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=None, help="批次大小")
    train_parser.add_argument("--lr", type=float, default=None, help="学习率")
    
    # 评估命令
    eval_parser = subparsers.add_parser("evaluate", help="评估模型性能")
    eval_parser.add_argument(
        "--model", type=str, default="all",
        choices=config.AVAILABLE_MODELS + ["all"],
        help="要评估的模型名称"
    )
    
    # 预测命令
    pred_parser = subparsers.add_parser("predict", help="单张图像预测")
    pred_parser.add_argument("--image", type=str, help="本地图片路径")
    pred_parser.add_argument("--image-url", type=str, help="图片URL")
    pred_parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL,
        choices=config.AVAILABLE_MODELS,
        help="使用的模型名称"
    )
    
    # API服务命令
    api_parser = subparsers.add_parser("api", help="启动Web API服务")
    api_parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL,
        choices=config.AVAILABLE_MODELS,
        help="使用的模型名称"
    )
    
    args = parser.parse_args()
    
    # 如果没有提供命令，显示帮助信息
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # 执行对应命令
    try:
        if args.command == "train":
            # 应用命令行覆盖的参数
            if args.epochs:
                config.EPOCHS = args.epochs
            if args.batch_size:
                config.BATCH_SIZE = args.batch_size
            if args.lr:
                config.LEARNING_RATE = args.lr
            cmd_train(args)
        
        elif args.command == "evaluate":
            cmd_evaluate(args)
        
        elif args.command == "predict":
            cmd_predict(args)
        
        elif args.command == "api":
            cmd_api(args)
    
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
