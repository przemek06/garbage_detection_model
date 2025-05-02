import cv2
import os
import numpy as np

def get_rois(image):
    image = cv2.resize(image, (224, 224))
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    rectangles = [convert_to_rectangle(rect, 224, 224) for rect in rects]
    return rectangles

def convert_to_rectangle(rect, width, height):
    x, y, w, h = rect
    rectangle = (x/width, y/height, (x + w)/width, (y + h)/height)
    return rectangle

def precompute_rois(image_dir, target_dir):
    image_files = [f for f in os.listdir(image_dir)]
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        rois = get_rois(image)
        rois_path = os.path.join(target_dir, image_file.replace('.jpg', '.npy'))
        np.save(rois_path, rois)
        print(f"Saved ROIs for {image_file} to {rois_path}")

def load_precomputed_rois(image_path, rois_dir):
    image_name = os.path.basename(image_path).replace('.jpg', '.npy')
    rois_path = os.path.join(rois_dir, image_name)
    rois = np.load(rois_path, allow_pickle=True)
    rois = [{'roi': roi} for roi in rois]
    return rois