import torch
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import utils

DATA_FILE = 'mydata.yml'
IMG_SIZE = 416
BATCH_SIZE = 32
EPOCHS = 1
SAVE_PATH = "yolo_trained.pt"
CLASSES = {0: "Plastic", 1: "Paper", 2: "Glass", 3: "Metal", 4: "Other"}

records = []

def train():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
    model = YOLO('yolov8n.pt')  
    model.to(device)

    def on_epoch_end(trainer):
        epoch = int(trainer.epoch) + 1

        out = model.val(
            data=DATA_FILE,
            split="train", 
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            verbose=False
        )

        metrics = {
            "train/mAP50": float(out.box.map50),
            "train/mAP": float(out.box.map),
            "train/precision": float(out.box.mp),
            "train/recall": float(out.box.mr),
            "epoch": epoch
        }
        records.append(metrics)
    
    model.add_callback("on_train_epoch_end", on_epoch_end)

    model.train(
        data=DATA_FILE,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='my_yolov8_model',
        save_period=1,
    )

    df = pd.DataFrame(records)
    df.to_csv("epoch_metrics_full.csv", index=False)
