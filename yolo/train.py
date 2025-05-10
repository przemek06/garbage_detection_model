import torch
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import utils
import os

DATA_FILE = 'mydata.yml'
IMG_SIZE = 416
BATCH_SIZE = 32
EPOCHS = 30
WEIGHTS_DIR = './epoch_weights'
METRICS_FILE = 'train_metrics.csv'

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
        weight_path = os.path.join(WEIGHTS_DIR, f'epoch_{epoch}.pt')
        model.save(weight_path)
    
    model.add_callback("on_train_epoch_end", on_epoch_end)

    model.train(
        data=DATA_FILE,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='my_yolov8_model'
    )

    evaluate_on_train(device)

def evaluate_on_train(device):
    for fname in sorted(os.listdir(WEIGHTS_DIR)):
        if fname.endswith('.pt'):
            epoch = int(fname.split('_')[1].split('.')[0])

            ckpt_path = os.path.join(WEIGHTS_DIR, fname)
            print(f"Evaluating epoch {epoch} with checkpoint {ckpt_path}")
            model_ckpt = YOLO(ckpt_path)
            model_ckpt.to(device)

            out = model_ckpt.val(
                data=DATA_FILE,
                split='train',
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                verbose=False
            )

            metrics = {
                'epoch': epoch,
                'train/mAP50': float(out.box.map50),
                'train/mAP': float(out.box.map),
                'train/precision': float(out.box.mp),
                'train/recall': float(out.box.mr)
            }
            records.append(metrics)

    df = pd.DataFrame(records).sort_values('epoch').reset_index(drop=True)
    df.to_csv(METRICS_FILE, index=False)
