#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练NEU-DET数据集的YOLOv8模型
"""

from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 可以选择 yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# 训练模型
results = model.train(
    data='neu_det.yaml',      # 数据集配置文件
    epochs=100,                # 训练轮数
    imgsz=640,                 # 图片尺寸
    batch=16,                  # 批次大小（根据GPU内存调整）
    device='cpu',              # 使用CPU，如果有GPU可以改为 'cuda' 或 '0'
    project='neu_det_results', # 结果保存目录
    name='train_run1',         # 实验名称
    save=True,                 # 保存检查点
    save_period=10,            # 每10个epoch保存一次
    val=True,                  # 训练时进行验证
    plots=True,                # 生成训练图表
    verbose=True               # 显示详细信息
)

print("训练完成！")
print(f"结果保存在: {results.save_dir}")

