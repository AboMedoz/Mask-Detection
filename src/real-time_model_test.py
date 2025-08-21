import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from image_preprocessing import preprocess_img

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT, 'models')

model = load_model(os.path.join(MODEL_PATH, 'mask_detection_model.keras'))

label_map = {0: 'No Mask', 1: 'Mask'}  # Hardcoded the labels, The Project will not expand

cap = cv2.VideoCapture(0)
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_detection.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in face:
        roi = gray[y: y + h, x:x + w]
        roi = preprocess_img(roi, 0, False)

        prediction_proba = model.predict(roi)
        predicition = label_map[np.argmax(prediction_proba)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicition, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                    2)

    cv2.imshow('Mask Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()