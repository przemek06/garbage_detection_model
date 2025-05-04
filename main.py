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
EPOCHS = 1
BATCH_SIZE = 4

def load_random_image(image_dir):
    image_files = os.listdir(image_dir)
    random_image_file = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image_file)
    image = cv2.imread(image_path)
    return image, random_image_file


def main():
    model = FastRCNN(CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        cls_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        bbox_loss_fn=tf.keras.losses.Huber()
    )

    train_dataset = load_training_data(NUM_CLASSES).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = load_validation_data(NUM_CLASSES).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

    model.save('fast_rcnn_model.h5')

    with open('training_history.txt', 'w') as f:
        for key in history.history.keys():
            f.write(f"{key}: {history.history[key]}\n")

    

if __name__ == "__main__":
    main()