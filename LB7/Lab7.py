# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from glob import glob
import ultralytics
ultralytics.checks()

model = YOLO(r"D:\PSUFINAL\DeepNeuro\LB7\runs\detect\train4\weights\best.pt")


val_folder = "dataset/images/val"
image_paths = glob(f"{val_folder}/*.jpg") + glob(f"{val_folder}/*.png")


for img_path in image_paths:
    results = model(img_path)
    result = results[0]
    print(f"\nИзображение: {img_path}")
    print("Обнаруженные рамки:", result.boxes.xyxy)
    print("Классы:", result.boxes.cls)
    print("Конфиденсы:", result.boxes.conf)

    
    img_cv = result.plot()  
    cv2.imshow("YOLOv8 Detection", img_cv)
    cv2.waitKey(0)  

   
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

cv2.destroyAllWindows()
