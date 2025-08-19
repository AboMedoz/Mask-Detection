import os

import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

from image_preprocessing import preprocess_img

# https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT, 'data', 'raw')
PROCESSED_DATA = os.path.join(ROOT, 'data', 'processed')


label_map = {'mask': 1, 'no_mask': 0}  # Since we have only 2 classes it doesn't make sense to store it in JSON
imgs = []
labels = []

for cat in os.listdir(DATA_PATH):
    category_path = os.path.join(DATA_PATH, cat)
    for img_str in os.listdir(category_path):
        img_path = os.path.join(category_path, img_str)
        img = cv2.imread(img_path)
        img = preprocess_img(img, -1)
        imgs.append(img)
        labels.append(label_map[cat])
    print(f"Finished Folder {cat}")  # Sanity Check

x = np.array(imgs, dtype=np.float32)
y = to_categorical(np.array(labels), num_classes=2)

x, y = shuffle(x, y, random_state=42)

np.savez_compressed(
    os.path.join(PROCESSED_DATA, 'processed_data.npz'),
    x=x,
    y=y,
)

