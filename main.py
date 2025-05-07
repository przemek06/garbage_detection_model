import tensorflow as tf
import cv2 
from visualize import draw_bounding_boxes
import os
import random
from selective_search import load_precomputed_rois
import numpy as np
from normalize_data import standardize_image
from load_data import load_training_data, load_validation_data
from model import FastRCNN

CLASSES = {0: "Plastic", 1: "Paper", 2: "Glass", 3: "Metal", 4: "Other", 5: "Background"}
NUM_CLASSES = len(CLASSES)
EPOCHS = 30
BATCH_SIZE = 8
CHECKPOINT_PATH = "best_fast_rcnn_model.ckpt"

def load_random_image(image_dir):
    image_files = os.listdir(image_dir)
    random_image_file = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image_file)
    image = cv2.imread(image_path)
    return image

def precompute_rois():
    precompute_rois("./data/split1/train/images", "./data/split1/train/rois")
    precompute_rois("./data/split1/val/images", "./data/split1/val/rois")
    precompute_rois("./data/split1/test/images", "./data/split1/test/rois")

def train():
    model = FastRCNN(CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        cls_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        bbox_loss_fn=tf.keras.losses.Huber()
    )

    train_dataset = load_training_data(NUM_CLASSES).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = load_validation_data(NUM_CLASSES).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=True,       
        monitor='val_cls_loss',           
        mode='min',                    
        save_best_only=True,           
        verbose=1
    )

    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[checkpoint_cb])

    with open('training_history.txt', 'w') as f:
        for key in history.history.keys():
            f.write(f"{key}: {history.history[key]}\n")

def test():
    model = FastRCNN(CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        cls_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        bbox_loss_fn=tf.keras.losses.Huber()
    )
    model.load_weights(CHECKPOINT_PATH)
    image = load_random_image("./data/split1/test/images")
    final_classes, final_boxes = model.predict(image)
    for cls in final_classes:
        cls_name = CLASSES[int(cls.numpy()[0])]
        print(f"Predicted class: {cls_name}")
    draw_bounding_boxes(image, final_boxes)

def main():
    train()

if __name__ == "__main__":
    main()