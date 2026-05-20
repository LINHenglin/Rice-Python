"""
单张图像预测工具。

使用方法:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --model efficientnet_b0
    python predict.py --image-url http://example.com/rice.jpg
"""
import argparse
import io
import os

import torch
from PIL import Image
from torchvision import transforms

import config
from models import create_model

# 预测时的图像变换（与验证集一致）
_predict_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class RiceDiseasePredictor:
    """
    水稻病害识别预测器。
    
    封装模型加载、图像预处理和推理逻辑，提供简洁的预测接口。
    支持从文件路径、字节流、URL等多种方式加载图像。
    
    Attributes:
        model_name: 使用的模型名称
        device: 计算设备（CPU/GPU）
        model: 加载的神经网络模型
        class_names: 类别名称列表
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.class_names = None
        self._load_model()

    def _load_model(self):
        ckpt_path = os.path.join(
            config.CHECKPOINT_DIR, f"{self.model_name}_best.pth"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"模型权重不存在: {ckpt_path}\n请先运行 train.py 训练模型。"
            )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.class_names = ckpt["class_names"]
        self.model = create_model(
            self.model_name, ckpt["num_classes"], pretrained=False
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        print(f"[预测器] 已加载 {self.model_name}，类别: {self.class_names}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """
        对单张 PIL 图像进行预测。
        
        这是预测器的核心方法，执行以下步骤:
        1. 图像预处理（调整尺寸、归一化）
        2. 前向传播获取 logits
        3. Softmax 转换为概率分布
        4. 提取最高概率的类别作为预测结果
        5. 查询防治建议知识库
        
        Args:
            image (PIL.Image): 输入图像（RGB格式）
        
        Returns:
            dict: 预测结果字典，包含:
                - class_en: 英文类别名
                - class_cn: 中文类别名
                - confidence: 置信度 (0~1)
                - all_probs: 所有类别的概率分布
                - treatment: 防治建议
        """
        # 图像预处理：应用与训练时相同的变换
        img_tensor = _predict_transform(image.convert("RGB"))
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # 添加batch维度

        # 前向传播：获取原始输出（logits）
        outputs = self.model(img_tensor)
        
        # Softmax激活：将logits转换为概率分布
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        conf, pred_idx = probs.max(0)  # 取最大概率及其索引

        # 根据预测索引获取类别名称和防治建议
        class_en = self.class_names[pred_idx.item()]
        class_cn = config.CLASS_CN.get(class_en, class_en)
        treatment = config.TREATMENT.get(class_en, "暂无防治建议。")

        # 构建所有类别的概率分布（用于可视化）
        all_probs = {
            self.class_names[i]: round(probs[i].item(), 4)
            for i in range(len(self.class_names))
        }

        return {
            "class_en": class_en,
            "class_cn": class_cn,
            "confidence": round(conf.item(), 4),
            "all_probs": all_probs,
            "treatment": treatment,
        }

    def predict_from_path(self, image_path: str) -> dict:
        """从文件路径加载图片并预测。"""
        image = Image.open(image_path)
        return self.predict(image)

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """从字节流加载图片并预测。"""
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict(image)

    def predict_from_url(self, image_url: str) -> dict:
        """从 URL 下载图片并预测。"""
        import requests
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        return self.predict_from_bytes(resp.content)


def main():
    parser = argparse.ArgumentParser(description="水稻病害识别 — 单张预测")
    parser.add_argument("--image", type=str, help="本地图片路径")
    parser.add_argument("--image-url", type=str, help="图片 URL")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        choices=config.AVAILABLE_MODELS)
    args = parser.parse_args()

    if not args.image and not args.image_url:
        parser.error("请提供 --image 或 --image-url 参数")

    predictor = RiceDiseasePredictor(model_name=args.model)

    if args.image:
        result = predictor.predict_from_path(args.image)
    else:
        result = predictor.predict_from_url(args.image_url)

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


if __name__ == "__main__":
    main()
