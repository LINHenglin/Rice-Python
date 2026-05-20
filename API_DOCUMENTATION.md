# 水稻病害识别系统 - API接口文档

## 目录
- [概述](#概述)
- [快速开始](#快速开始)
- [Web API接口](#web-api接口)
- [Python SDK接口](#python-sdk接口)
- [命令行接口](#命令行接口)
- [数据格式说明](#数据格式说明)
- [错误码说明](#错误码说明)

---

## 概述

本系统提供基于深度学习的水稻病害智能识别服务，支持10种水稻常见病害及健康状态的识别。系统提供多种调用方式，包括RESTful API、Python SDK和命令行工具。

### 支持的病害类别

| 英文类别名 | 中文名称 | 类型 |
|-----------|---------|------|
| bacterial_leaf_blight | 白叶枯病 | 细菌性病害 |
| brown_spot | 褐斑病 | 真菌性病害 |
| healthy | 健康植株 | 无病害 |
| leaf_blast | 叶瘟病 | 真菌性病害 |
| leaf_scald | 叶烧病 | 细菌性病害 |
| narrow_brown_spot | 窄褐条斑病 | 真菌性病害 |
| neck_blast | 颈瘟病 | 真菌性病害 |
| rice_hispa | 水稻铁甲虫 | 虫害 |
| sheath_blight | 纹枯病 | 真菌性病害 |
| tungro | 东格鲁病 | 病毒病害 |

### 支持的模型

| 模型名称 | 参数量 | 推理速度 | 准确率 | 适用场景 |
|---------|-------|---------|--------|---------|
| resnet50 | ~23.5M | 7.36ms | 96.64% | 高精度需求 |
| mobilenetv3 | ~1.5M | 6.58ms | 93.78% | 移动端/实时应用 |
| efficientnet_b0 | ~4.0M | 10.26ms | 96.06% | 精度与效率平衡 |

---

## 快速开始

### 启动API服务

```bash
# 使用默认模型（MobileNetV3）
python main.py api

# 指定模型
python main.py api --model resnet50
```

服务将在 `http://0.0.0.0:5000` 启动。

---

## Web API接口

### 基础信息

- **Base URL**: `http://localhost:5000`
- **Content-Type**: `application/json` 或 `multipart/form-data`
- **字符编码**: UTF-8

### 1. 健康检查接口

检查服务是否正常运行。

**请求**
```
GET /health
```

**响应示例**
```json
{
  "status": "ok",
  "model": "mobilenetv3"
}
```

**状态码**
- `200`: 服务正常

---

### 2. 图像预测接口（文件上传）

通过上传图像文件进行病害识别。

**请求**
```
POST /predict
Content-Type: multipart/form-data
```

**请求参数**

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| image | File | 是 | 水稻叶片图像文件（JPG/PNG格式） |

**cURL示例**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@rice_leaf.jpg"
```

**Python示例**
```python
import requests

url = "http://localhost:5000/predict"
files = {"image": open("rice_leaf.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()
print(result)
```

**响应示例**
```json
{
  "class_en": "leaf_blast",
  "class_cn": "叶瘟病",
  "confidence": 0.9234,
  "all_probs": {
    "bacterial_leaf_blight": 0.0012,
    "brown_spot": 0.0045,
    "healthy": 0.0023,
    "leaf_blast": 0.9234,
    "leaf_scald": 0.0018,
    "narrow_brown_spot": 0.0067,
    "neck_blast": 0.0456,
    "rice_hispa": 0.0008,
    "sheath_blight": 0.0123,
    "tungro": 0.0014
  },
  "treatment": "1. 选用抗瘟品种；2. 合理密植，保持通风；3. 发病初期喷施三环唑或稻瘟灵。"
}
```

**响应字段说明**

| 字段名 | 类型 | 说明 |
|-------|------|------|
| class_en | string | 预测类别的英文名称 |
| class_cn | string | 预测类别的中文名称 |
| confidence | float | 预测置信度（0~1之间） |
| all_probs | object | 所有类别的概率分布 |
| treatment | string | 防治建议 |

**状态码**
- `200`: 预测成功
- `400`: 请求参数错误（未上传图像）
- `500`: 服务器内部错误

---

### 3. 图像预测接口（Base64编码）

通过Base64编码的图像数据进行病害识别，适用于移动端或前端应用。

**请求**
```
POST /predict_base64
Content-Type: application/json
```

**请求参数**

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| image | string | 是 | Base64编码的图像数据（不含前缀） |

**请求体示例**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**JavaScript示例**
```javascript
// 将图片转换为Base64
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      // 移除 data:image/jpeg;base64, 前缀
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = error => reject(error);
  });
}

// 调用API
async function predictImage(file) {
  const base64Image = await fileToBase64(file);
  
  const response = await fetch('http://localhost:5000/predict_base64', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      image: base64Image
    })
  });
  
  const result = await response.json();
  console.log(result);
}
```

**Python示例**
```python
import requests
import base64

url = "http://localhost:5000/predict_base64"

# 读取图片并转换为Base64
with open("rice_leaf.jpg", "rb") as f:
    image_bytes = f.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

# 发送请求
response = requests.post(url, json={"image": base64_image})
result = response.json()
print(result)
```

**响应示例**
```json
{
  "class_en": "healthy",
  "class_cn": "健康",
  "confidence": 0.9876,
  "all_probs": {
    "bacterial_leaf_blight": 0.0002,
    "brown_spot": 0.0003,
    "healthy": 0.9876,
    "leaf_blast": 0.0005,
    "leaf_scald": 0.0001,
    "narrow_brown_spot": 0.0004,
    "neck_blast": 0.0003,
    "rice_hispa": 0.0002,
    "sheath_blight": 0.0003,
    "tungro": 0.0001
  },
  "treatment": "植株健康，请继续保持良好的田间管理。"
}
```

**状态码**
- `200`: 预测成功
- `400`: 请求参数错误（缺少image字段）
- `500`: 服务器内部错误（Base64解码失败等）

---

## Python SDK接口

### 安装依赖

```bash
pip install torch torchvision Pillow requests
```

### 基本使用

```python
from predict import RiceDiseasePredictor

# 创建预测器（默认使用mobilenetv3模型）
predictor = RiceDiseasePredictor(model_name="mobilenetv3")
```

### 从文件路径预测

```python
result = predictor.predict_from_path("path/to/image.jpg")

print(f"类别: {result['class_cn']}")
print(f"置信度: {result['confidence']:.2%}")
print(f"防治建议: {result['treatment']}")
```

### 从字节流预测

```python
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

result = predictor.predict_from_bytes(image_bytes)
```

### 从URL预测

```python
result = predictor.predict_from_url("http://example.com/rice_leaf.jpg")
```

### 直接使用PIL图像对象

```python
from PIL import Image

image = Image.open("rice_leaf.jpg")
result = predictor.predict(image)

# 查看完整结果
print("预测结果:")
print(f"  英文类别: {result['class_en']}")
print(f"  中文类别: {result['class_cn']}")
print(f"  置信度: {result['confidence']:.4f}")
print(f"\n各类别概率:")
for cls, prob in sorted(result['all_probs'].items(), key=lambda x: -x[1]):
    print(f"  {cls}: {prob:.4f}")
print(f"\n防治建议:\n  {result['treatment']}")
```

### 指定计算设备

```python
# 使用GPU（如果可用）
predictor = RiceDiseasePredictor(model_name="resnet50", device="cuda")

# 强制使用CPU
predictor = RiceDiseasePredictor(model_name="efficientnet_b0", device="cpu")
```

### 返回数据结构

```python
{
    "class_en": "leaf_blast",           # 英文类别名
    "class_cn": "叶瘟病",                # 中文类别名
    "confidence": 0.9234,               # 置信度 (0~1)
    "all_probs": {                      # 所有类别的概率分布
        "bacterial_leaf_blight": 0.0012,
        "brown_spot": 0.0045,
        ...
    },
    "treatment": "1. 选用抗瘟品种..."   # 防治建议
}
```

---

## 命令行接口

### 训练模型

```bash
# 训练单个模型
python main.py train --model mobilenetv3

# 训练所有模型进行对比
python main.py train --model all

# 自定义训练参数
python main.py train --model resnet50 --epochs 50 --batch-size 16 --lr 0.001
```

**参数说明**

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| --model | 模型名称（resnet50/mobilenetv3/efficientnet_b0/all） | all |
| --epochs | 训练轮数 | 30 |
| --batch-size | 批次大小 | 32 |
| --lr | 学习率 | 0.001 |

---

### 评估模型

```bash
# 评估所有已训练模型
python main.py evaluate

# 评估指定模型
python main.py evaluate --model efficientnet_b0
```

**输出内容**
- 准确率、精确率、召回率、F1分数
- 推理速度（毫秒）
- 模型参数量
- 混淆矩阵可视化
- 模型对比图表

---

### 单图预测

```bash
# 预测本地图片
python main.py predict --image test.jpg

# 预测网络图片
python main.py predict --image-url http://example.com/rice.jpg

# 指定模型
python main.py predict --image test.jpg --model resnet50
```

**参数说明**

| 参数 | 说明 | 必填 |
|-----|------|------|
| --image | 本地图片路径 | 二选一 |
| --image-url | 图片URL | 二选一 |
| --model | 使用的模型名称 | 否（默认mobilenetv3） |

---

### 启动API服务

```bash
# 使用默认模型
python main.py api

# 指定模型
python main.py api --model efficientnet_b0
```

---

## 数据格式说明

### 输入图像要求

- **格式**: JPG、PNG等常见图像格式
- **尺寸**: 任意尺寸（系统会自动调整为224x224）
- **颜色空间**: RGB（自动转换）
- **文件大小**: 建议小于10MB
- **内容**: 清晰的水稻叶片照片，建议拍摄时：
  - 光线充足且均匀
  - 背景简洁
  - 病斑特征清晰可见
  - 避免过度曝光或欠曝光

### 输出数据格式

所有接口返回的JSON数据结构一致：

```json
{
  "class_en": "string",      // 英文类别名
  "class_cn": "string",      // 中文类别名
  "confidence": 0.0,         // 置信度 (float, 0~1)
  "all_probs": {             // 所有类别的概率分布
    "class_name_1": 0.0,
    "class_name_2": 0.0,
    ...
  },
  "treatment": "string"      // 防治建议
}
```

---

## 错误码说明

### HTTP状态码

| 状态码 | 说明 | 可能原因 |
|-------|------|---------|
| 200 | 成功 | 请求处理成功 |
| 400 | 请求错误 | 缺少必要参数、参数格式错误 |
| 500 | 服务器错误 | 模型加载失败、图像处理异常 |

### 常见错误及解决方案

#### 1. 模型权重文件不存在

**错误信息**
```
FileNotFoundError: 模型权重不存在: checkpoints/xxx_best.pth
请先运行 train.py 训练模型。
```

**解决方案**
```bash
python main.py train --model <model_name>
```

#### 2. 未上传图像文件

**错误响应**
```json
{
  "error": "请上传图像文件"
}
```

**解决方案**
确保在请求中包含`image`字段（文件上传）或`image`字段（Base64）。

#### 3. Base64解码失败

**错误响应**
```json
{
  "error": "Invalid base64-encoded string"
}
```

**解决方案**
- 检查Base64字符串是否完整
- 确保不包含`data:image/jpeg;base64,`前缀
- 验证Base64编码是否正确

#### 4. 图像格式不支持

**错误响应**
```json
{
  "error": "cannot identify image file"
}
```

**解决方案**
- 使用JPG或PNG格式
- 确保图像文件未损坏
- 检查文件扩展名与实际格式是否一致

---

## 性能指标

### 模型性能对比

| 模型 | 准确率 | 推理速度 | 参数量 | GPU显存占用 |
|-----|--------|---------|--------|------------|
| ResNet50 | 96.64% | 7.36ms | 23.5M | ~2GB |
| MobileNetV3 | 93.78% | 6.58ms | 1.5M | ~500MB |
| EfficientNet-B0 | 96.06% | 10.26ms | 4.0M | ~1GB |

*测试环境: NVIDIA GPU, batch_size=1*

### 推荐配置

**生产环境**
- GPU: NVIDIA GTX 1060或更高
- 显存: 4GB以上
- CPU: 4核以上
- 内存: 8GB以上

**开发环境**
- CPU模式即可（推理速度约100-200ms/张）
- 内存: 4GB以上

---

## 最佳实践

### 1. 提高识别准确率

- 拍摄清晰的叶片照片
- 确保病斑特征明显
- 避免强光直射或阴影遮挡
- 多角度拍摄并取平均结果

### 2. 批量处理

```python
from predict import RiceDiseasePredictor
import os

predictor = RiceDiseasePredictor(model_name="mobilenetv3")

image_dir = "path/to/images"
results = []

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png')):
        filepath = os.path.join(image_dir, filename)
        result = predictor.predict_from_path(filepath)
        results.append({
            "filename": filename,
            "prediction": result
        })
        print(f"{filename}: {result['class_cn']} ({result['confidence']:.2%})")
```

### 3. 置信度阈值过滤

```python
result = predictor.predict_from_path("image.jpg")

# 设置置信度阈值
if result['confidence'] < 0.7:
    print("置信度较低，建议人工复核")
else:
    print(f"识别结果: {result['class_cn']}")
```

### 4. 多模型集成

```python
# 使用多个模型进行投票
models = ["resnet50", "mobilenetv3", "efficientnet_b0"]
predictions = {}

for model_name in models:
    predictor = RiceDiseasePredictor(model_name=model_name)
    result = predictor.predict_from_path("image.jpg")
    predictions[model_name] = result

# 选择置信度最高的结果
best_model = max(predictions.items(), key=lambda x: x[1]['confidence'])
print(f"最佳预测: {best_model[0]} -> {best_model[1]['class_cn']}")
```

---

## 常见问题

### Q1: 如何提高识别准确率？

A: 
1. 使用更高分辨率的图像
2. 确保拍摄条件良好（光线、角度）
3. 尝试不同的模型（ResNet50精度最高）
4. 对同一叶片多角度拍摄并综合判断

### Q2: API服务如何部署到生产环境？

A:
1. 使用Gunicorn替代Flask开发服务器：
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

2. 使用Nginx作为反向代理
3. 启用HTTPS加密传输
4. 添加身份认证机制

### Q3: 如何添加新的病害类别？

A:
1. 在数据集中添加新类别文件夹
2. 更新`config.py`中的`CLASS_NAMES_EN`和`CLASS_CN`
3. 添加对应的防治建议到`TREATMENT`
4. 重新训练模型

### Q4: 支持哪些编程语言调用？

A:
任何支持HTTP请求的语言都可以调用Web API，包括：
- Python (requests)
- JavaScript (fetch/axios)
- Java (HttpClient)
- C# (HttpClient)
- Go (net/http)
- 等等

---

## 技术支持

如有问题或建议，请联系开发团队。

**版本**: v1.0  
**更新日期**: 2026-04-14
