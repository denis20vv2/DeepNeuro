# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Проверяем системы
ultralytics.checks()

# Загружаем предобученную модель
model = YOLO("yolov8s.pt")

# --- ДЕТЕКЦИЯ НА ИЗОБРАЖЕНИИ ---

results = model("image_1.jpg")   # ← укажи путь к любому изображению
result = results[0]

# Результат через OpenCV
cv2.imshow("YOLOv8", result.plot())

# Результат через matplotlib
plt.imshow(result.plot()[:, :, ::-1])
plt.show()

print("Обнаруженные рамки:", result.boxes)
print("Классы:", result.boxes.cls)
print("Конфиденсы:", result.boxes.conf)

# --- ДЕТЕКЦИЯ НА ВИДЕО ---

video_path = "masktrack.mp4"     # ← укажи путь к видео
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    vid_results = model(frame)
    cv2.imshow("YOLOv8-video", vid_results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
