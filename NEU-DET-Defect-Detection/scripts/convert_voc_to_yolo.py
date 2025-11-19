#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将NEU-DET数据集的Pascal VOC格式（XML）转换为YOLO格式（TXT）
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# 数据集路径
dataset_path = r"C:\Users\刘嘉淯\Desktop\NEU-DET"
annotations_dir = os.path.join(dataset_path, "Annotations")
images_dir = os.path.join(dataset_path, "JPEGImages")
imagesets_main = os.path.join(dataset_path, "ImageSets", "Main")

# 读取类别文件
classes_file = os.path.join(dataset_path, "classes.txt")
if os.path.exists(classes_file):
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
else:
    classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# 创建类别到索引的映射
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
print(f"类别映射: {class_to_idx}")

def convert_bbox_to_yolo(xml_file, img_width, img_height):
    """将Pascal VOC的bbox转换为YOLO格式"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_to_idx:
            print(f"警告: 未知类别 {class_name}，跳过")
            continue
        
        class_id = class_to_idx[class_name]
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
        center_x = (xmin + xmax) / 2.0 / img_width
        center_y = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations

def get_image_size(image_path):
    """获取图片尺寸"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        return img.width, img.height
    except:
        # 如果PIL不可用，尝试使用其他方法
        print(f"警告: 无法读取图片尺寸 {image_path}，使用默认尺寸")
        return 200, 200  # 默认尺寸，可能需要根据实际情况调整

def convert_dataset(split='train'):
    """转换指定数据集划分"""
    split_file = os.path.join(imagesets_main, f"{split}.txt")
    if not os.path.exists(split_file):
        print(f"未找到 {split}.txt 文件")
        return
    
    # 读取图片列表
    with open(split_file, 'r', encoding='utf-8') as f:
        image_names = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"\n处理 {split} 集，共 {len(image_names)} 张图片")
    
    # 创建输出目录
    output_images_dir = os.path.join(dataset_path, "images", split)
    output_labels_dir = os.path.join(dataset_path, "labels", split)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    converted = 0
    skipped = 0
    
    for img_name in image_names:
        # 查找图片文件（可能没有扩展名或不同扩展名）
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            test_path = os.path.join(images_dir, img_name + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if not img_path:
            print(f"未找到图片: {img_name}")
            skipped += 1
            continue
        
        # 查找对应的XML文件
        xml_path = os.path.join(annotations_dir, img_name + ".xml")
        if not os.path.exists(xml_path):
            print(f"未找到标注文件: {img_name}.xml")
            skipped += 1
            continue
        
        # 获取图片尺寸
        img_width, img_height = get_image_size(img_path)
        
        # 转换标注
        yolo_annotations = convert_bbox_to_yolo(xml_path, img_width, img_height)
        
        if not yolo_annotations:
            print(f"警告: {img_name} 没有有效标注")
            skipped += 1
            continue
        
        # 保存YOLO格式标注
        label_file = os.path.join(output_labels_dir, img_name + ".txt")
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_annotations))
        
        # 复制图片到images目录
        img_ext = os.path.splitext(img_path)[1]
        output_img_path = os.path.join(output_images_dir, img_name + img_ext)
        shutil.copy2(img_path, output_img_path)
        
        converted += 1
        if converted % 100 == 0:
            print(f"已转换 {converted} 张图片...")
    
    print(f"\n{split} 集转换完成:")
    print(f"  成功: {converted} 张")
    print(f"  跳过: {skipped} 张")

if __name__ == "__main__":
    print("开始转换NEU-DET数据集...")
    print(f"数据集路径: {dataset_path}")
    print(f"类别: {classes}")
    
    # 转换训练集和验证集
    convert_dataset('train')
    convert_dataset('val')
    
    # 如果存在测试集，也转换
    if os.path.exists(os.path.join(imagesets_main, "test.txt")):
        convert_dataset('test')
    
    print("\n转换完成！")
    print(f"转换后的数据集结构:")
    print(f"  {dataset_path}/")
    print(f"    images/")
    print(f"      train/")
    print(f"      val/")
    print(f"    labels/")
    print(f"      train/")
    print(f"      val/")

