import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_random_image, map_to_aabb, draw_bounding_boxes

VAL_IMAGE_DIR = "../data/split1/val/images"
DATA_FILE = 'mydata.yml'
IMG_SIZE = 416
CLASSES = {0: "Plastic", 1: "Paper", 2: "Glass", 3: "Metal", 4: "Other"}
BEST_MODEL_PATH = "runs/detect/my_yolov8_model/weights/best.pt"

def test():

    model = YOLO(BEST_MODEL_PATH)

    img = load_random_image(VAL_IMAGE_DIR)
    img_np = np.array(img)[..., ::-1]

    results = model.predict(
        source=[img_np],
        imgsz=IMG_SIZE,
        conf=0.25,
        iou=0.45,
        verbose=False
    )

    for idx, r in enumerate(results):
        boxes   = r.boxes.xyxy.cpu().numpy()
        scores  = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.astype(int)
            label = f"{CLASSES[cls]} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Detection {idx}")
        plt.show()
