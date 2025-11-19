#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试训练NEU-DET数据集（使用少量图片和轮次）
"""

from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 使用最小的模型进行快速测试

# 训练模型（快速测试配置）
results = model.train(
    data='neu_det.yaml',      # 数据集配置文件
    epochs=5,                 # 只训练5轮（快速测试）
    imgsz=640,                # 图片尺寸
    batch=4,                  # 小批次（减少内存使用）
    device='cpu',             # 使用CPU
    project='neu_det_results', # 结果保存目录
    name='quick_test',        # 实验名称
    save=True,                # 保存检查点
    save_period=5,            # 每5个epoch保存一次
    val=True,                 # 训练时进行验证
    plots=True,               # 生成训练图表
    verbose=True,             # 显示详细信息
    fraction=0.0146          # 只使用约20张训练图片 (20/1368 ≈ 0.0146)
)

print("快速测试训练完成！")
print(f"结果保存在: {results.save_dir}")

