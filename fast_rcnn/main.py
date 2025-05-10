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
EPOCHS = 20
BATCH_SIZE = 8
CHECKPOINT_PATH = "best_fast_rcnn_model.ckpt"

class ResetMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_metrics()

def load_random_image(image_dir):
    image_files = os.listdir(image_dir)
    random_image_file = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[checkpoint_cb, ResetMetricsCallback()], shuffle=True)

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

def load_history():
    with open('training_history.txt', 'r') as f:
        history = {}
        for line in f:
            key, value = line.strip().split(': ')
            history[key] = eval(value)

    return history

def plot_training_history(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history['cls_loss'], label='Training Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['cls_accuracy'], label='Training Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['bbox_loss'], label='Training BBox Loss')
    plt.title('Bounding Box Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_vlaidation_history(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history['val_cls_loss'], label='Validation Loss')
    plt.title('Validation Classification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['val_cls_accuracy'], label='Validation Accuracy')
    plt.title('Validation Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['val_bbox_loss'], label='Validation BBox Loss')
    plt.title('Validation Bounding Box Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_training_history():
    history = load_history()
    plot_training_history(history)
    plot_vlaidation_history(history)

def main():
    visualize_training_history()
if __name__ == "__main__":
    main()