import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import cv2
import random
import glob

dataset_path = 'C:/Users/admin/Desktop/homework8/dataset/data1a'
def prepare_classification_dataset(input_path, output_path):

    for split in ['train', 'val']:
        for class_name in ['damage', 'whole']:
            os.makedirs(os.path.join(output_path, split, class_name), exist_ok=True)
    
    for class_idx, class_name in enumerate(['00-damage', '01-whole']):
        class_path = os.path.join(input_path, 'training', class_name)
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
            images.extend(glob.glob(os.path.join(class_path, ext)))

        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        for img_path in train_imgs:
            filename = os.path.basename(img_path)
            dst = os.path.join(output_path, 'train', 'damage' if 'damage' in class_name else 'whole', filename)
            shutil.copy(img_path, dst)
        
        for img_path in val_imgs:
            filename = os.path.basename(img_path)
            dst = os.path.join(output_path, 'val', 'damage' if 'damage' in class_name else 'whole', filename)
            shutil.copy(img_path, dst)

prepare_classification_dataset(dataset_path, 'car_damage_classification')
yaml_content = """
# car_damage_classification.yaml
path: car_damage_classification/
train: train/
val: val/

# Количество классов
nc: 2

# Имена классов
names: ['damage', 'whole']
"""

with open('car_damage_classification.yaml', 'w') as f:
    f.write(yaml_content)