from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os

# ------------------ 1. 参数设置 ------------------
# 模型权重路径（使用 quick_test 训练得到的 best.pt，可根据需要修改）
MODEL_PATH = Path('neu_det_results/quick_test/weights/best.pt')

# 待预测图片路径（示例：验证集中的一张图片，可替换为自己的图片）
IMAGE_PATH = Path(r'C:\Users\刘嘉淯\Desktop\NEU-DET\images\val\crazing_114.jpg')

# 预测结果保存路径
OUTPUT_IMAGE_PATH = Path('prediction_result.jpg')

# 类别名称（NEU-DET 数据集）
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# ------------------ 2. 加载模型并进行预测 ------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f'模型文件不存在: {MODEL_PATH}')

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f'待预测图片不存在: {IMAGE_PATH}')

model = YOLO(str(MODEL_PATH))

# 进行预测（save=False 避免 YOLO 自动保存一份）
results = model.predict(source=str(IMAGE_PATH), save=False, conf=0.25)
result = results[0]

# ------------------ 3. 绘制并显示结果 ------------------
annotated_frame = result.plot()  # 带框的图像，BGR 格式

print('正在显示预测结果，按任意键关闭窗口...')
cv2.imshow('NEU-DET Prediction', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存预测结果图片
cv2.imwrite(str(OUTPUT_IMAGE_PATH), annotated_frame)
print(f'预测结果已保存至: {OUTPUT_IMAGE_PATH.resolve()}')

# ------------------ 4. 打印检测到的缺陷信息 ------------------
if result.boxes is not None and len(result.boxes) > 0:
    boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
    scores = result.boxes.conf.cpu().numpy()  # [N]
    classes = result.boxes.cls.cpu().numpy().astype(int)  # [N]

    print('\n检测到的缺陷:')
    for idx, (box, score, cls_id) in enumerate(zip(boxes, scores, classes), start=1):
        x1, y1, x2, y2 = box.astype(int)
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
        print(f'  目标 {idx}: {class_name} | 置信度 {score:.2f} | 位置 [{x1}, {y1}, {x2}, {y2}]')
else:
    print('未检测到任何缺陷。')

# ------------------ 5. 提示无分割结果 ------------------
if result.masks is None:
    print('\n提示: 当前模型为检测模型，不包含分割 mask。')
