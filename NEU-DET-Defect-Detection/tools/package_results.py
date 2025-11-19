#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将训练结果分类整理并打包到桌面
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
import zipfile

# 桌面路径
desktop_path = Path.home() / "Desktop"

# 创建结果文件夹（带时间戳）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"NEU-DET训练结果_{timestamp}"
result_folder = desktop_path / folder_name

# 创建分类子文件夹
folders = {
    "模型权重": result_folder / "模型权重",
    "训练图表": result_folder / "训练图表",
    "验证结果": result_folder / "验证结果",
    "训练批次": result_folder / "训练批次",
    "配置文件": result_folder / "配置文件"
}

for folder in folders.values():
    folder.mkdir(parents=True, exist_ok=True)

print(f"创建结果文件夹: {result_folder}")

# 源目录
source_dir = Path("neu_det_results/quick_test")

# 文件分类映射
file_mapping = {
    "模型权重": ["weights/best.pt", "weights/last.pt", "weights/epoch0.pt"],
    "训练图表": [
        "results.png", "results.csv",
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png",
        "labels.jpg", "labels_correlogram.jpg"
    ],
    "验证结果": [
        "val_batch0_labels.jpg", "val_batch0_pred.jpg",
        "val_batch1_labels.jpg", "val_batch1_pred.jpg",
        "val_batch2_labels.jpg", "val_batch2_pred.jpg"
    ],
    "训练批次": [
        "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg"
    ],
    "配置文件": ["args.yaml"]
}

# 复制文件
copied_count = 0
for category, files in file_mapping.items():
    target_folder = folders[category]
    for file_path in files:
        source_file = source_dir / file_path
        if source_file.exists():
            if "/" in file_path:  # 有子目录的情况
                target_file = target_folder / Path(file_path).name
            else:
                target_file = target_folder / file_path
            
            shutil.copy2(source_file, target_file)
            copied_count += 1
            print(f"  复制: {file_path} -> {category}/{target_file.name}")
        else:
            print(f"  警告: 文件不存在 {file_path}")

print(f"\n共复制 {copied_count} 个文件")

# 创建README文件
readme_content = f"""# NEU-DET 训练结果

训练时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 文件夹说明

- **模型权重**: 训练好的模型文件
  - `best.pt`: 验证集上表现最好的模型（推荐使用）
  - `last.pt`: 最后一个epoch的模型
  - `epoch0.pt`: 第0个epoch的模型

- **训练图表**: 训练过程中的各种图表和指标
  - `results.png`: 训练曲线总览
  - `results.csv`: 详细训练数据
  - `confusion_matrix.png`: 混淆矩阵
  - `PR_curve.png`: 精确率-召回率曲线
  - 其他曲线图...

- **验证结果**: 验证集上的预测结果可视化
  - `val_batch*_labels.jpg`: 真实标签
  - `val_batch*_pred.jpg`: 模型预测结果

- **训练批次**: 训练批次的可视化图片

- **配置文件**: 训练时使用的参数配置

## 使用训练好的模型

```python
from ultralytics import YOLO

# 加载最佳模型
model = YOLO('模型权重/best.pt')

# 进行预测
results = model.predict('your_image.jpg')
results[0].show()  # 显示结果
```

## 数据集信息

- 数据集: NEU-DET (东北大学表面缺陷检测数据集)
- 类别数: 6
- 类别: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- 训练图片: 20张（快速测试）
- 验证图片: 342张
- 训练轮次: 5 epochs
"""

readme_file = result_folder / "README.txt"
with open(readme_file, 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"创建说明文件: README.txt")

# 创建ZIP压缩包
zip_filename = f"{folder_name}.zip"
zip_path = desktop_path / zip_filename

print(f"\n正在创建压缩包: {zip_filename}")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(result_folder):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(desktop_path)
            zipf.write(file_path, arcname)
            print(f"  添加: {arcname}")

print(f"\n完成！")
print(f"结果文件夹: {result_folder}")
print(f"压缩包: {zip_path}")

