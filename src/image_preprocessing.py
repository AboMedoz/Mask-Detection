import cv2
import numpy as np


def preprocess_img(img, axis):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Under Test
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis)
    return img
