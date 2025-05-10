import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Configuration
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5
BATCH_SIZE = 1
MAX_ROIS = 2000
NMS_THRESHOLD = 0.5


def load_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    gt_boxes = []
    gt_classes = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls = int(parts[0])
        gt_boxes.append(parts[1:5])
        gt_classes.append(cls)
    return np.array(gt_boxes), np.array(gt_classes)

def compute_iou(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
    
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def get_targets(rois, gt_boxes, gt_classes):
    ious = compute_iou(rois, gt_boxes)
    max_iou_idx = np.argmax(ious, axis=1)
    max_ious = np.max(ious, axis=1)
    
    class_targets = gt_classes[max_iou_idx]
    pos_indices = max_ious >= 0.5
    
    # Compute regression targets
    pos_rois = rois[pos_indices]
    pos_gt = gt_boxes[max_iou_idx[pos_indices]]
    
    roi_cx = (pos_rois[:, 0] + pos_rois[:, 2]) / 2
    roi_cy = (pos_rois[:, 1] + pos_rois[:, 3]) / 2
    roi_w = pos_rois[:, 2] - pos_rois[:, 0]
    roi_h = pos_rois[:, 3] - pos_rois[:, 1]
    
    gt_cx = (pos_gt[:, 0] + pos_gt[:, 2]) / 2
    gt_cy = (pos_gt[:, 1] + pos_gt[:, 3]) / 2
    gt_w = pos_gt[:, 2] - pos_gt[:, 0]
    gt_h = pos_gt[:, 3] - pos_gt[:, 1]
    
    dx = (gt_cx - roi_cx) / roi_w
    dy = (gt_cy - roi_cy) / roi_h
    dw = np.log(gt_w / roi_w)
    dh = np.log(gt_h / roi_h)
    
    regression_targets = np.zeros((len(rois), 4), dtype=np.float32)
    regression_targets[pos_indices] = np.stack([dx, dy, dw, dh], axis=1)
    
    return class_targets, regression_targets, pos_indices

def create_dataset(data_dir):
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    roi_dir = os.path.join(data_dir, 'precomputed')
    
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    for image_path in image_paths:
        # Precompute ROIs
        precompute_rois(image_path, roi_dir)
        
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = image / 255.0
        
        # Load ROIs
        roi_file = os.path.splitext(os.path.basename(image_path))[0] + '_rois.npy'
        rois = np.load(os.path.join(roi_dir, roi_file))
        
        # Load labels
        label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
        gt_boxes, gt_classes = load_labels(label_path)
        
        # Compute targets
        class_targets, reg_targets, pos_indices = get_targets(rois, gt_boxes, gt_classes)
        
        yield (image, rois), (class_targets, reg_targets, pos_indices)

def build_fast_rcnn():
    # Feature extractor
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3))
    
    # Inputs
    image_input = layers.Input(shape=(*IMAGE_SIZE, 3))
    roi_input = layers.Input(shape=(MAX_ROIS, 4))
    
    # Feature maps
    features = base_model(image_input)
    
    # ROI Pooling
    def roi_pool(feats, rois):
        batch_size = tf.shape(feats)[0]
        num_rois = tf.shape(rois)[1]
        
        # Convert normalized coordinates to absolute
        h = tf.cast(tf.shape(feats)[1], tf.float32)
        w = tf.cast(tf.shape(feats)[2], tf.float32)
        boxes = tf.reshape(rois, [-1, 4])
        boxes = tf.stack([
            boxes[:, 1] * h,  # y1
            boxes[:, 0] * w,  # x1
            boxes[:, 3] * h,  # y2
            boxes[:, 2] * w,  # x2
        ], axis=1)
        
        box_indices = tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32)
        pooled = tf.image.crop_and_resize(
            feats, boxes, box_indices, (14, 14))
        return pooled
    
    pooled_features = roi_pool(features, roi_input)
    
    # Network heads
    x = layers.TimeDistributed(layers.Flatten())(pooled_features)
    x = layers.TimeDistributed(layers.Dense(1024, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    x = layers.TimeDistributed(layers.Dense(1024, activation='relu'))(x)
    
    # Outputs
    cls_output = layers.TimeDistributed(
        layers.Dense(NUM_CLASSES, activation='softmax'), name='cls')(x)
    reg_output = layers.TimeDistributed(
        layers.Dense(4, activation='linear'), name='reg')(x)
    
    return Model(inputs=[image_input, roi_input], outputs=[cls_output, reg_output])

class FastRCNNTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(1e-5)
    
    def compute_loss(self, cls_out, reg_out, cls_true, reg_true, pos_mask):
        # Classification loss
        cls_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(cls_true, cls_out))
        
        # Regression loss (only for positive ROIs)
        reg_loss = tf.reduce_mean(tf.abs(
            tf.boolean_mask(reg_out, pos_mask) - 
            tf.boolean_mask(reg_true, pos_mask)))
        
        return cls_loss + reg_loss, cls_loss, reg_loss
    
    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            cls_pred, reg_pred = self.model(inputs, training=True)
            total_loss, cls_loss, reg_loss = self.compute_loss(
                cls_pred, reg_pred, *targets)
        
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss, cls_loss, reg_loss
    
    @tf.function
    def val_step(self, inputs, targets):
        cls_pred, reg_pred = self.model(inputs, training=False)
        return self.compute_loss(cls_pred, reg_pred, *targets)

def train():
    # Prepare datasets
    train_dataset = tf.data.Dataset.from_generator(
        lambda: create_dataset('train'),
        output_signature=(
            (tf.TensorSpec(shape=(*IMAGE_SIZE, 3)), tf.TensorSpec(shape=(MAX_ROIS, 4))),
            (tf.TensorSpec(shape=(MAX_ROIS,)), tf.TensorSpec(shape=(MAX_ROIS, 4)), 
             tf.TensorSpec(shape=(MAX_ROIS,)))
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: create_dataset('val'),
        output_signature=(
            (tf.TensorSpec(shape=(*IMAGE_SIZE, 3)), tf.TensorSpec(shape=(MAX_ROIS, 4))),
            (tf.TensorSpec(shape=(MAX_ROIS,)), tf.TensorSpec(shape=(MAX_ROIS, 4)), 
             tf.TensorSpec(shape=(MAX_ROIS,)))
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    
    # Initialize model and trainer
    model = build_fast_rcnn()
    trainer = FastRCNNTrainer(model)
    
    # Training loop
    for epoch in range(10):
        # Training
        train_loss = []
        for inputs, targets in train_dataset:
            loss = trainer.train_step(inputs, targets)
            train_loss.append(loss)
            print(f'Epoch {epoch} Train Loss: {loss[0].numpy()}')
        
        # Validation
        val_loss = []
        for inputs, targets in val_dataset:
            loss = trainer.val_step(inputs, targets)
            val_loss.append(loss)
            print(f'Epoch {epoch} Val Loss: {loss[0].numpy()}')
        
        # Save weights
        model.save_weights(f'fast_rcnn_epoch_{epoch}.h5')

class FastRCNNPredictor:
    def __init__(self, model_weights):
        self.model = build_fast_rcnn()
        self.model.load_weights(model_weights)
    
    def predict(self, image_path):        
        # Load and preprocess image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        original_shape = tf.shape(image).numpy()
        image = tf.image.resize(image, IMAGE_SIZE)
        image = image / 255.0
        image = tf.expand_dims(image, 0)
        
        # Load ROIs
        roi_file = os.path.splitext(os.path.basename(image_path))[0] + '_rois.npy'
        rois = np.load(os.path.join('test/rois', roi_file))
        rois = tf.expand_dims(rois, 0)
        
        # Predict
        cls_pred, reg_pred = self.model([image, rois])
        cls_pred = tf.nn.softmax(cls_pred).numpy()[0]
        reg_pred = reg_pred.numpy()[0]
        
        # Decode boxes
        boxes = []
        scores = []
        classes = []
        for i in range(MAX_ROIS):
            if np.all(rois[0][i] == 0):  # Skip padded ROIs
                continue
            
            # Get class and score
            class_id = np.argmax(cls_pred[i])
            score = cls_pred[i][class_id]
            
            # Apply regression
            dx, dy, dw, dh = reg_pred[i]
            x1, y1, x2, y2 = rois[0][i]
            
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w/2 + dx*w
            cy = y1 + h/2 + dy*h
            nw = w * np.exp(dw)
            nh = h * np.exp(dh)
            
            nx1 = cx - nw/2
            ny1 = cy - nh/2
            nx2 = cx + nw/2
            ny2 = cy + nh/2
            
            boxes.append([nx1, ny1, nx2, ny2])
            scores.append(score)
            classes.append(class_id)
        
        # Apply NMS
        if len(boxes) > 0:
            selected = tf.image.non_max_suppression(
                boxes, scores, 100, NMS_THRESHOLD)
            return np.array(classes)[selected.numpy()], np.array(boxes)[selected.numpy()]
        return [], []