from ultralytics import YOLO
import os
import numpy as np
import torch
import cv2
import random


def map_to_yolo(cls, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return cls, x_center, y_center, width, height

def map_to_aabb(x_center, y_center, width, height):
    xmin = x_center - (width / 2)
    ymin = y_center - (height / 2)
    xmax = x_center + (width / 2)
    ymax = y_center + (height / 2)
    return xmin, ymin, xmax, ymax

def build_labels_dict(path):
    labels_dict = {}
    for filename in os.listdir(path):
        with open (os.path.join(path, filename), 'r') as f:
            lines = f.readlines()
            boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xmin, ymin, xmax, ymax = map(float, parts)
                    boxes.append(map_to_yolo(int(cls), xmin, ymin, xmax, ymax))
            labels_dict[filename] = boxes
    return labels_dict

def populate_labels_dir(aabb_path, labels_path):
    labels_dict = build_labels_dict(aabb_path)
    for filename, boxes in labels_dict.items():
        with open(os.path.join(labels_path, filename), 'w') as f:
            for box in boxes:
                cls, x_center, y_center, width, height = box
                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")
    
def load_model(path):
    model = YOLO(path)
    return model

def draw_bounding_boxes(image, boxes):
    height, width, _ = image.shape

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin = int(xmin * width)
        ymin = int(ymin * height)
        xmax = int(xmax * width)
        ymax = int(ymax * height)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_random_image(image_dir):
    image_files = os.listdir(image_dir)
    random_image_file = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image