from normalize_data import standardize_image
import os
import cv2
import numpy as np
import tensorflow as tf
from selective_search import load_precomputed_rois
import tensorflow as tf

def load_data_sample(image_path, label_dir, rois_dir, num_classes):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    filename = os.path.basename(image_path)
    label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
    objects = []
    rois = load_precomputed_rois(image_path, rois_dir)

    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            labels = file.readlines()
        for label in labels:
            values = label.strip().split()
            if len(values) != 5:
                continue
            cl, xmin, ymin, xmax, ymax = map(float, values)
            objects.append({'class': int(cl), 'bbox': [xmin, ymin, xmax, ymax]})
    
    sample = {'image': image, 'objects': objects, 'rois': rois}
    match_by_iou(sample, num_classes)
    sample = filter_background(sample, num_classes - 1)

    if sample['objects']:
        obj_classes = np.array([obj['class'] for obj in sample['objects']], dtype=np.int32)
        obj_bboxes = np.array([obj['bbox'] for obj in sample['objects']], dtype=np.float32)
    else:
        obj_classes = np.empty((0,), dtype=np.int32)
        obj_bboxes = np.empty((0, 4), dtype=np.float32)
    sample['objects'] = {'class': obj_classes, 'bbox': obj_bboxes}

    if sample['rois']:
        roi_rois = np.array([r['roi'] for r in sample['rois']], dtype=np.float32)
        roi_classes = np.array([r['class'] for r in sample['rois']], dtype=np.int32)
        roi_bboxes = np.array([r['bbox'] for r in sample['rois']], dtype=np.float32)
    else:
        roi_rois = np.empty((0, 4), dtype=np.float32)
        roi_classes = np.empty((0,), dtype=np.int32)
        roi_bboxes = np.empty((0, 4), dtype=np.float32)
    sample['rois'] = {'roi': roi_rois, 'class': roi_classes, 'bbox': roi_bboxes}
    
    return {
        'rois': sample['rois'],
        'image': sample['image'],
        'objects': sample['objects']
    }


def match_by_iou(instance, num_classes):
    for roi in instance['rois']:
        best_iou = 0.0
        best_obj = None
        for obj in instance['objects']:
            iou = calculate_iou(roi['roi'], obj['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_obj = obj

        if best_obj is not None and best_iou > 0.5:
            roi['class'] = best_obj['class']
            roi['bbox'] = best_obj['bbox']
            roi['is_positive'] = True
        else:
            roi['class'] = num_classes - 1
            roi['bbox'] = roi['roi']
            roi['is_positive'] = False

def filter_background(sample, background_class):
    filtered_rois = []
    for roi in sample['rois']:
        if roi['class'] != background_class:
            filtered_rois.append(roi)
        elif np.random.rand() < 0.03:
            filtered_rois.append(roi)
    sample['rois'] = filtered_rois
    return sample


def calculate_iou(roi, bbox):
    xA = max(roi[0], bbox[0])
    yA = max(roi[1], bbox[1])
    xB = min(roi[2], bbox[2])
    yB = min(roi[3], bbox[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    union_area = roi_area + bbox_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area 

    return iou


def data_generator(image_dir, label_dir, rois_dir, num_classes):
    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)
        yield load_data_sample(path, label_dir, rois_dir, num_classes)

def format_sample(sample, max_rois=32):
    img = sample['image']
    img = tf.cast(img, tf.float32)
    
    roi_coords = sample['rois']['roi']  
    roi_classes = sample['rois']['class']  
    roi_bboxes = sample['rois']['bbox'] 
    num_rois = tf.shape(roi_coords)[0]
    shuffled_indices = tf.random.shuffle(tf.range(num_rois))
    selected_indices = shuffled_indices[:tf.minimum(max_rois, num_rois)]

    truncated_coords = tf.gather(roi_coords, selected_indices)
    truncated_classes = tf.gather(roi_classes, selected_indices)
    truncated_bboxes = tf.gather(roi_bboxes, selected_indices)
    
    num_rois = tf.shape(truncated_coords)[0]

    valid_mask = tf.concat([
        tf.ones(num_rois,  dtype=tf.bool),
        tf.zeros(max_rois - num_rois, dtype=tf.bool)
    ], axis=0)
    
    padded_coords = tf.pad(
        truncated_coords,
        paddings=[[0, max_rois - num_rois], [0, 0]],
        constant_values=0.0
    )
    padded_classes = tf.pad(
        truncated_classes,
        paddings=[[0, max_rois - num_rois]],
        constant_values=5
    )
    padded_bboxes = tf.pad(
        truncated_bboxes,
        paddings=[[0, max_rois - num_rois], [0, 0]],
        constant_values=0.0
    )
    
    return (img, padded_coords, padded_classes, padded_bboxes, valid_mask)

def change_bbox_targets_to_offsets(sample):
    img, padded_coords, padded_classes, padded_bboxes, valid_mask = sample

    def compute_offsets(inputs):
        coords, bbox = inputs
        xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
        xmin_b, ymin_b, xmax_b, ymax_b = bbox[0], bbox[1], bbox[2], bbox[3]

        w = xmax - xmin
        h = ymax - ymin

        if w == 0 or h == 0:
            return tf.zeros(4, dtype=tf.float32)

        dx = (xmin_b - xmin) / w
        dy = (ymin_b - ymin) / h
        dw = (xmax_b - xmin_b) / w
        dh = (ymax_b - ymin_b) / h

        return tf.stack([dx, dy, dw, dh])

    offsets = tf.map_fn(
        compute_offsets,
        (padded_coords, padded_bboxes),
        dtype=tf.float32
    )

    return (img, padded_coords), (padded_classes, offsets, valid_mask)


def build_tf_dataset(image_dir, label_dir, rois_dir, num_classes, max_rois=32):
    output_signature = {
        'image': tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        'rois': {
            'roi': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            'class': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'bbox': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        },
        'objects': {
            'class': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'bbox': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        },
    }
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(image_dir, label_dir, rois_dir, num_classes),
        output_signature=output_signature
    )
    dataset = dataset.map(lambda x: format_sample(x, max_rois), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda img, coords, classes, bboxes, valid_mask: change_bbox_targets_to_offsets((img, coords, classes, bboxes, valid_mask)), num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

def load_training_data(num_classes):
    training_data_dir = "./data/split1/train/"
    training_images_dir = training_data_dir + "images"
    training_labels_dir = training_data_dir + "labels"
    training_rois_dir = training_data_dir + "rois"
    return build_tf_dataset(training_images_dir, training_labels_dir, training_rois_dir, num_classes)

def load_validation_data(num_classes):
    validation_data_dir = "./data/split1/val/"
    validation_images_dir = validation_data_dir + "images"
    validation_labels_dir = validation_data_dir + "labels"
    validation_rois_dir = validation_data_dir + "rois"
    return build_tf_dataset(validation_images_dir, validation_labels_dir, validation_rois_dir, num_classes)

def load_test_data(num_classes):
    test_data_dir = "./data/split1/test/"
    test_images_dir = test_data_dir + "images"
    test_labels_dir = test_data_dir + "labels"
    test_rois_dir = test_data_dir + "rois"
    return build_tf_dataset(test_images_dir, test_labels_dir, test_rois_dir, num_classes)