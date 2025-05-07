import cv2
from pathlib import Path

def load_and_draw_bounding_boxes(image_path, txt_path):
    image = cv2.imread(image_path)
    if image is None:
        return
    
    height, width, _ = image.shape
    
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        values = line.strip().split()
        if len(values) != 5:
            continue
        
        _, xmin, ymin, xmax, ymax = map(float, values)
        
        xmin, ymin, xmax, ymax = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_bounding_boxes(image, boxes):
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
