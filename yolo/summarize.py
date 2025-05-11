import csv
import matplotlib.pyplot as plt
import os

RESULTS_FILE = "./runs/detect/my_yolov8_model/results.csv"
TRAIN_METRICS_FILE = "train_metrics.csv"
GRAPHS_DIR = "./graphs"

code_name_dict = {
    'time': 'Time',
    'train/box_loss': 'Train Box Loss',
    'train/cls_loss': 'Train Classification Loss',
    'train/dfl_loss': 'Train Distribution Focal Loss',
    'metrics/precision(B)': 'Val Precision (B)',
    'metrics/recall(B)': 'Val Recall (B)',
    'metrics/mAP50(B)': 'Val mAP50 (B)',
    'metrics/mAP50-95(B)': 'Val mAP50-95 (B)',
    'val/box_loss': 'Val Box Loss',
    'val/cls_loss': 'Val Classification Loss',
    'val/dfl_loss': 'Val Distribution Focal Loss',
    'train/mAP50': 'Train mAP50',
    'train/precision': 'Train Precision',
    'train/recall': 'Train Recall',
}

def summarize():
    results = csv_to_dict(RESULTS_FILE)
    train_metrics = csv_to_dict(TRAIN_METRICS_FILE)

    for code_key, nice_name in code_name_dict.items():
        plt.figure()
        plotted = False

        if code_key in train_metrics:
            y_train = [float(x) for x in train_metrics[code_key]]
            x_train = list(range(1, len(y_train) + 1))
            plt.plot(x_train, y_train, label=f"{nice_name}")
            plotted = True

        if code_key in results:
            y_val = [float(x) for x in (results[code_key])]
            x_val = list(range(1, len(y_val) + 1))
            plt.plot(x_val, y_val, '--', label=f"{nice_name}")
            plotted = True

        if plotted:
            plt.title(nice_name)
            plt.xlabel('Epoch')
            plt.ylabel(nice_name)
            plt.legend()
            plt.grid(True)
            filename = nice_name
            filepath = os.path.join(GRAPHS_DIR, f"{filename}.png")
            plt.savefig(filepath)
            plt.close()



def csv_to_dict(file_path):
    result = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key, value in row.items():
                result.setdefault(key, []).append(value)
    return result