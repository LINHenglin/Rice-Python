"""
数据集加载与图像增强。
支持从文件夹结构自动读取类别，提供训练/验证/测试数据加载器。
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config


def get_train_transforms():
    """
    训练集数据增强策略。
    
    数据增强是深度学习中防止过拟合、提升模型泛化能力的重要手段。
    通过随机变换生成“新”样本，使模型学习到更鲁棒的特征表示。
    
    增强方法包括:
    - 随机缩放裁剪: 模拟不同拍摄距离
    - 水平/垂直翻转: 增加样本多样性
    - 随机旋转: 应对叶片不同角度
    - 颜色抖动: 模拟不同光照条件
    - 随机仿射变换: 增加几何变换鲁棒性
    
    Returns:
        torchvision.transforms.Compose: 组合变换对象
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """
    验证集/测试集图像预处理。
    
    与训练集不同，验证/测试阶段不进行随机增强，仅进行确定性的
    尺寸调整和归一化，确保评估结果的稳定性和可重复性。
    
    Returns:
        torchvision.transforms.Compose: 组合变换对象
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_dataset(data_dir, transform):
    """
    从文件夹结构加载ImageFolder数据集。
    
    PyTorch的ImageFolder会自动根据子文件夹名称分配类别标签。
    期望的目录结构:
        data_dir/
            class_name_1/
                img1.jpg
                img2.jpg
            class_name_2/
                img3.jpg
                ...
    
    Args:
        data_dir (str): 数据集根目录路径
        transform: 图像变换函数
    
    Returns:
        torchvision.datasets.ImageFolder: 数据集对象
    
    Raises:
        FileNotFoundError: 当数据集目录不存在时抛出异常
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"数据集目录不存在: {data_dir}\n"
            "请运行 download_dataset.py 下载数据集，或手动将图片按类别放入对应子文件夹。"
        )
    return datasets.ImageFolder(root=data_dir, transform=transform)


def get_dataloaders(train_dir=None, val_dir=None, test_dir=None,
                    batch_size=None, num_workers=None,
                    val_split=0.2):
    """
    创建训练、验证、测试数据加载器（DataLoader）。
    
    DataLoader是PyTorch中用于批量加载数据的核心类，支持:
    - 自动批处理（batching）
    - 多线程数据加载（num_workers）
    - 数据打乱（shuffle）
    - 内存优化（pin_memory加速GPU传输）
    
    如果未提供独立的验证集目录，则从训练集中按比例划分。
    
    Args:
        train_dir (str): 训练集目录
        val_dir (str): 验证集目录（可选）
        test_dir (str): 测试集目录（可选）
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        val_split (float): 验证集划分比例（默认0.2即20%）
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
               如果test_dir不存在，test_loader为None
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    train_dir = train_dir or config.TRAIN_DIR
    val_dir = val_dir or config.VAL_DIR
    test_dir = test_dir or config.TEST_DIR

    train_dataset = load_dataset(train_dir, get_train_transforms())
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"[数据集] 检测到 {num_classes} 个类别: {class_names}")
    print(f"[数据集] 训练样本总数: {len(train_dataset)}")

    if os.path.isdir(val_dir) and len(os.listdir(val_dir)) > 0:
        val_dataset = load_dataset(val_dir, get_val_transforms())
        print(f"[数据集] 验证样本总数: {len(val_dataset)}")
    else:
        total = len(train_dataset)
        val_size = int(total * val_split)
        train_size = total - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        print(f"[数据集] 自动划分 → 训练: {train_size}, 验证: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    test_loader = None
    if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
        test_dataset = load_dataset(test_dir, get_val_transforms())
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        print(f"[数据集] 测试样本总数: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names
