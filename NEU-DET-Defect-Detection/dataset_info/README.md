# NEU-DET 数据集说明

## 数据集介绍

NEU-DET是东北大学提供的金属表面缺陷检测数据集，包含6种常见的表面缺陷类型。

## 数据集结构

```
NEU-DET/
├── Annotations/          # XML标注文件（Pascal VOC格式）
├── ImageSets/
│   └── Main/
│       ├── train.txt     # 训练集图片列表
│       ├── val.txt       # 验证集图片列表
│       └── test.txt      # 测试集图片列表
└── JPEGImages/          # 原始图片文件
```

## 数据集下载

请从官方渠道下载NEU-DET数据集：
- 数据集官网
- 相关论文引用

## 数据格式转换

使用 `scripts/convert_voc_to_yolo.py` 将Pascal VOC格式转换为YOLO格式：

转换后的结构：
```
NEU-DET/
├── images/
│   ├── train/           # 训练图片
│   ├── val/             # 验证图片
│   └── test/            # 测试图片
└── labels/
    ├── train/           # 训练标签（TXT格式）
    ├── val/             # 验证标签
    └── test/            # 测试标签
```

## 类别说明

1. **crazing** - 裂纹
2. **inclusion** - 夹杂
3. **patches** - 斑块
4. **pitted_surface** - 点蚀表面
5. **rolled-in_scale** - 轧制氧化皮
6. **scratches** - 划痕

## 注意事项

- 数据集文件较大，请确保有足够的存储空间
- 转换脚本会自动处理图片格式（支持.jpg和.bmp）
- 请根据实际情况修改配置文件中的路径
