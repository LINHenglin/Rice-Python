"""
数据集重组脚本：统一类别命名并重新划分训练集、验证集、测试集

功能：
1. 统一文件夹命名为小写+下划线格式
2. 按比例划分：训练集70%、验证集15%、测试集15%
3. 确保每个类别在三个数据集中都有样本
4. 保持类别平衡性
"""
import os
import shutil
import random
from collections import defaultdict

# 设置随机种子保证可重复性
random.seed(42)

# 原始数据路径
ORIGINAL_TRAIN = 'data/train'
ORIGINAL_TEST = 'data/test'

# 目标数据路径
OUTPUT_DIR = 'data/rice_disease'
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_VAL = os.path.join(OUTPUT_DIR, 'val')
OUTPUT_TEST = os.path.join(OUTPUT_DIR, 'test')

# 划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def normalize_folder_name(name):
    """统一文件夹命名为小写+下划线格式"""
    # 转换为小写
    name = name.lower()
    # 替换空格为下划线
    name = name.replace(' ', '_')
    # 移除多余的下划线
    while '__' in name:
        name = name.replace('__', '_')
    return name

def collect_all_images():
    """收集所有图像文件并按类别分组"""
    print("=" * 80)
    print("步骤1: 收集所有图像文件")
    print("=" * 80)
    
    category_images = defaultdict(list)
    
    # 从原始训练集收集
    if os.path.exists(ORIGINAL_TRAIN):
        for category in os.listdir(ORIGINAL_TRAIN):
            cat_path = os.path.join(ORIGINAL_TRAIN, category)
            if not os.path.isdir(cat_path):
                continue
            
            normalized_cat = normalize_folder_name(category)
            images = [f for f in os.listdir(cat_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img in images:
                category_images[normalized_cat].append(os.path.join(cat_path, img))
            
            print(f"  从训练集收集: {category:30} -> {normalized_cat:30} ({len(images)} 张)")
    
    # 从原始测试集收集
    if os.path.exists(ORIGINAL_TEST):
        for category in os.listdir(ORIGINAL_TEST):
            cat_path = os.path.join(ORIGINAL_TEST, category)
            if not os.path.isdir(cat_path):
                continue
            
            normalized_cat = normalize_folder_name(category)
            images = [f for f in os.listdir(cat_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img in images:
                # 避免重复添加（如果文件名相同）
                img_path = os.path.join(cat_path, img)
                if img_path not in category_images[normalized_cat]:
                    category_images[normalized_cat].append(img_path)
            
            print(f"  从测试集收集: {category:30} -> {normalized_cat:30} ({len(images)} 张)")
    
    print(f"\n共收集 {len(category_images)} 个类别的图像")
    for cat, imgs in sorted(category_images.items()):
        print(f"  {cat:30}: {len(imgs)} 张")
    
    return category_images

def split_dataset(category_images):
    """按比例划分数据集"""
    print("\n" + "=" * 80)
    print("步骤2: 划分数据集 (训练集70% / 验证集15% / 测试集15%)")
    print("=" * 80)
    
    splits = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }
    
    for category, images in sorted(category_images.items()):
        # 打乱顺序
        random.shuffle(images)
        
        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)
        test_count = total - train_count - val_count  # 确保总数一致
        
        # 确保每个集合至少有1个样本（对于小类别）
        if total >= 3:
            train_imgs = images[:train_count]
            val_imgs = images[train_count:train_count + val_count]
            test_imgs = images[train_count + val_count:]
        elif total == 2:
            train_imgs = images[:1]
            val_imgs = images[1:2]
            test_imgs = images[1:2]  # 复用验证集样本
        else:  # total == 1
            train_imgs = images
            val_imgs = images
            test_imgs = images
        
        splits['train'][category] = train_imgs
        splits['val'][category] = val_imgs
        splits['test'][category] = test_imgs
        
        print(f"  {category:30}: 总计{total:4d} -> 训练:{len(train_imgs):4d}, 验证:{len(val_imgs):4d}, 测试:{len(test_imgs):4d}")
    
    return splits

def copy_images(splits):
    """复制图像到目标目录"""
    print("\n" + "=" * 80)
    print("步骤3: 复制图像到目标目录")
    print("=" * 80)
    
    # 创建目标目录结构
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for category in splits[split_name].keys():
            cat_dir = os.path.join(split_dir, category)
            os.makedirs(cat_dir, exist_ok=True)
    
    # 复制文件
    total_copied = 0
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        
        for category, images in splits[split_name].items():
            cat_dir = os.path.join(split_dir, category)
            
            for i, img_path in enumerate(images):
                # 生成新文件名（避免冲突）
                ext = os.path.splitext(img_path)[1]
                new_filename = f"{category}_{i:05d}{ext}"
                dst_path = os.path.join(cat_dir, new_filename)
                
                try:
                    shutil.copy2(img_path, dst_path)
                    total_copied += 1
                except Exception as e:
                    print(f"  ⚠️  复制失败: {img_path} -> {dst_path}, 错误: {e}")
        
        print(f"  ✓ {split_name} 集复制完成")
    
    print(f"\n总共复制 {total_copied} 个文件")

def verify_dataset():
    """验证重组后的数据集"""
    print("\n" + "=" * 80)
    print("步骤4: 验证数据集完整性")
    print("=" * 80)
    
    summary = {}
    
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        if not os.path.exists(split_dir):
            print(f"  ⚠️  {split_name} 目录不存在!")
            continue
        
        categories = sorted(os.listdir(split_dir))
        summary[split_name] = {}
        
        print(f"\n  {split_name.upper()} 集:")
        total = 0
        for cat in categories:
            cat_dir = os.path.join(split_dir, cat)
            if os.path.isdir(cat_dir):
                count = len([f for f in os.listdir(cat_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                summary[split_name][cat] = count
                total += count
                print(f"    {cat:30}: {count:5d} 张")
        
        print(f"    {'总计':30}: {total:5d} 张")
    
    # 检查一致性
    print("\n" + "-" * 80)
    print("一致性检查:")
    
    all_categories = set()
    for split_data in summary.values():
        all_categories.update(split_data.keys())
    
    missing_cats = {}
    for split_name, split_data in summary.items():
        missing = all_categories - set(split_data.keys())
        if missing:
            missing_cats[split_name] = missing
    
    if missing_cats:
        print("  ⚠️  以下类别在某些数据集中缺失:")
        for split_name, missing in missing_cats.items():
            print(f"    {split_name}: {missing}")
    else:
        print("  ✓ 所有类别在所有数据集中都存在")
    
    # 统计信息
    print("\n" + "-" * 80)
    print("数据统计摘要:")
    print(f"  类别总数: {len(all_categories)}")
    for split_name in ['train', 'val', 'test']:
        if split_name in summary:
            total = sum(summary[split_name].values())
            print(f"  {split_name.upper():6} 集: {total:6d} 张图像")
    
    total_all = sum(sum(s.values()) for s in summary.values())
    print(f"  {'TOTAL':6}: {total_all:6d} 张图像")
    
    return summary

def main():
    """主函数"""
    print("\n🌾 水稻病害数据集重组工具")
    print("=" * 80)
    print(f"原始训练集路径: {ORIGINAL_TRAIN}")
    print(f"原始测试集路径: {ORIGINAL_TEST}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"划分比例: 训练集{TRAIN_RATIO*100:.0f}% / 验证集{VAL_RATIO*100:.0f}% / 测试集{TEST_RATIO*100:.0f}%")
    print("=" * 80)
    
    # 询问用户确认
    response = input("\n⚠️  此操作将创建新的数据集结构，是否继续? (y/n): ")
    if response.lower() != 'y':
        print("操作已取消")
        return
    
    # 执行重组
    try:
        # 步骤1: 收集所有图像
        category_images = collect_all_images()
        
        if not category_images:
            print("\n❌ 未找到任何图像文件!")
            return
        
        # 步骤2: 划分数据集
        splits = split_dataset(category_images)
        
        # 步骤3: 复制图像
        copy_images(splits)
        
        # 步骤4: 验证
        summary = verify_dataset()
        
        print("\n" + "=" * 80)
        print("✅ 数据集重组完成!")
        print("=" * 80)
        print(f"\n新数据集位置: {os.path.abspath(OUTPUT_DIR)}")
        print("\n目录结构:")
        print(f"  data/rice_disease/")
        print(f"    ├── train/    (训练集)")
        print(f"    ├── val/      (验证集)")
        print(f"    └── test/     (测试集)")
        print("\n下一步:")
        print("  1. 更新 config.py 中的 DATA_DIR 配置")
        print("  2. 运行训练脚本进行测试")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
