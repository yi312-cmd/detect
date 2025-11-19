# NEU-DET 表面缺陷检测项目

基于YOLOv8的金属表面缺陷检测系统，使用NEU-DET数据集进行训练。

## 📋 项目简介

本项目实现了对金属表面6种缺陷的自动检测：
- **crazing** (裂纹)
- **inclusion** (夹杂)
- **patches** (斑块)
- **pitted_surface** (点蚀表面)
- **rolled-in_scale** (轧制氧化皮)
- **scratches** (划痕)

## 🗂️ 项目结构

```
NEU-DET-Defect-Detection/
├── scripts/              # 主要脚本
│   ├── convert_voc_to_yolo.py    # VOC格式转YOLO格式
│   ├── train_neu_det.py          # 完整训练脚本
│   ├── quick_train_neu_det.py    # 快速测试训练
│   ├── predict_neu_det.py        # 预测脚本
│   └── predict_neu_det_cv2.py    # OpenCV预测脚本
├── tools/                # 工具脚本
│   ├── package_results.py        # 结果打包工具
│   └── compare_prediction.py     # 预测对比工具
├── configs/              # 配置文件
│   └── neu_det.yaml              # 数据集配置文件
├── docs/                 # 文档
│   └── 预测模型说明.md           # 模型说明文档
├── dataset_info/         # 数据集信息
│   └── README.md                 # 数据集说明
└── examples/             # 示例
```

## 🚀 快速开始

### 1. 环境要求

```bash
pip install ultralytics opencv-python numpy
```

### 2. 数据集准备

1. 下载NEU-DET数据集
2. 运行格式转换脚本：
```bash
python scripts/convert_voc_to_yolo.py
```

### 3. 训练模型

```bash
# 快速测试训练（5 epochs）
python scripts/quick_train_neu_det.py

# 完整训练（100 epochs）
python scripts/train_neu_det.py
```

### 4. 使用模型预测

```bash
# 使用YOLO默认方式
python scripts/predict_neu_det.py

# 使用OpenCV显示
python scripts/predict_neu_det_cv2.py
```

## 📊 数据集信息

- **数据集名称**: NEU-DET (东北大学表面缺陷检测数据集)
- **数据格式**: Pascal VOC (XML) → YOLO (TXT)
- **类别数量**: 6类
- **图片尺寸**: 200x200
- **数据集路径**: 请参考 `configs/neu_det.yaml`

> **注意**: 数据集文件较大，请从官方渠道下载，不包含在本仓库中。

## 🔧 配置文件说明

### `configs/neu_det.yaml`

```yaml
path: /path/to/NEU-DET          # 数据集根目录
train: images/train            # 训练集路径
val: images/val                # 验证集路径
test: images/test              # 测试集路径（可选）

nc: 6                          # 类别数量
names: ['crazing', 'inclusion', 'patches', 
        'pitted_surface', 'rolled-in_scale', 'scratches']

task: detect                   # 任务类型：detect（检测）
```

## 📈 训练结果

训练完成后，结果保存在 `neu_det_results/` 目录下：
- `weights/best.pt`: 最佳模型权重
- `weights/last.pt`: 最后一个epoch的权重
- `results.png`: 训练曲线图
- `confusion_matrix.png`: 混淆矩阵
- 其他评估图表...

## 🎯 使用示例

### 单张图片预测

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('neu_det_results/quick_test/weights/best.pt')

# 预测
results = model.predict('path/to/image.jpg', conf=0.25)
results[0].show()  # 显示结果
```

### 批量预测

```python
# 预测整个文件夹
results = model.predict('path/to/images/', conf=0.25, save=True)
```

## 📝 注意事项

1. **数据集路径**: 请根据实际情况修改 `configs/neu_det.yaml` 中的路径
2. **模型性能**: 快速测试训练（5 epochs）仅用于验证流程，完整训练需要更多epochs
3. **GPU支持**: 建议使用GPU加速训练，CPU训练会非常慢

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目仅供学习和研究使用。

## 📧 联系方式

如有问题，请提交Issue。

---

**最后更新**: 2025-11-17
