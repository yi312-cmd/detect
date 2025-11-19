#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的NEU-DET模型进行预测
"""

from ultralytics import YOLO
from pathlib import Path
import os

# 加载训练好的最佳模型
model_path = 'neu_det_results/quick_test/weights/best.pt'
print(f"加载模型: {model_path}")

if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在 {model_path}")
    exit(1)

model = YOLO(model_path)

# 预测选项
# 1. 单张图片预测
# 2. 批量预测（验证集）
# 3. 自定义图片路径

print("\n=== NEU-DET 表面缺陷检测预测 ===\n")

# 选项1: 预测训练集中前20张图片
train_images_dir = Path(r"C:\Users\刘嘉淯\Desktop\NEU-DET\images\train")
if train_images_dir.exists():
    # 获取训练集中的图片（支持 jpg/png/bmp）
    image_files = (
        list(train_images_dir.glob("*.jpg"))
        + list(train_images_dir.glob("*.png"))
        + list(train_images_dir.glob("*.bmp"))
    )

    if len(image_files) > 0:
        test_images = image_files[:20]
        print(f"从训练集中选择 {len(test_images)} 张图片进行预测:")

        for img_path in test_images:
            print(f"\n预测图片: {img_path.name}")
            
            # 进行预测
            results = model.predict(
                source=str(img_path),
                conf=0.25,        # 置信度阈值
                save=True,        # 保存预测结果
                save_txt=False,   # 不保存txt标注
                save_conf=True,   # 保存置信度
                project='predictions',  # 保存目录
                name='neu_det_train20',    # 子目录名
                exist_ok=True     # 允许覆盖
            )
            
            # 显示结果信息
            result = results[0]
            print(f"  检测到 {len(result.boxes)} 个目标")
            
            if len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]
                    print(f"    目标 {i+1}: {class_name} (置信度: {conf:.2f})")
            else:
                print("    未检测到任何缺陷")
        
        print(f"\n预测结果已保存到: predictions/neu_det_train20/")
    else:
        print("训练集目录中没有找到图片文件")
else:
    print(f"训练集目录不存在: {train_images_dir}")

print("\n=== 预测完成 ===")
print("\n提示: 您也可以使用以下代码进行自定义预测:")
print("""
from ultralytics import YOLO

model = YOLO('neu_det_results/quick_test/weights/best.pt')

# 预测单张图片
results = model.predict('your_image.jpg', conf=0.25, save=True)

# 预测整个文件夹
results = model.predict('path/to/images/', conf=0.25, save=True)

# 显示结果
results[0].show()
""")

