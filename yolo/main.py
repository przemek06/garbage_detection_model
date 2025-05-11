from train import train, evaluate_on_train
import utils
from predict import test
from summarize import summarize

AABB_TRAIN_LABELS_DIR = "../data/split2/train/aabb"
AABB_VAL_LABELS_DIR = "../data/split2/val/aabb"
AABB_TEST_LABELS_DIR = "../data/split2/test/aabb"
YOLO_TRAIN_LABELS_DIR = "../data/split2/train/labels"
YOLO_VAL_LABELS_DIR = "../data/split2/val/labels"
YOLO_TEST_LABELS_DIR = "../data/split1/test/labels"

def main():
    # utils.populate_labels_dir(AABB_TRAIN_LABELS_DIR, YOLO_TRAIN_LABELS_DIR)
    # utils.populate_labels_dir(AABB_VAL_LABELS_DIR, YOLO_VAL_LABELS_DIR)
    # utils.populate_labels_dir(AABB_TEST_LABELS_DIR, YOLO_TEST_LABELS_DIR)

    # evaluate_on_train()

    summarize()

if __name__ == "__main__":
    main()