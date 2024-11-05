import os
from ultralytics import YOLO

# Define the directory where images and annotations are located
base_dir = '/home/majid/PycharmProjects/E5/data/obj_train_data_RGB'  # Path to the directory containing images and annotations

# Define file paths to train, val, and test text files
train_file = '/home/majid/PycharmProjects/E5/data/train.txt'
val_file = '/home/majid/PycharmProjects/E5/data/val.txt'
test_file = '/home/majid/PycharmProjects/E5/data/test.txt'

# Load and train the YOLOv8 model
model = YOLO('yolo11x.pt')  # Load the YOLO model (adjust the model variant as needed)

# Use both GPUs (GPU 0 and GPU 1)
model.train(data='/home/majid/PycharmProjects/E5/data/data.yml', epochs=100, imgsz=640, batch=16, name='yolov11_experiment', device='0,1')

# Optionally, validate the model after training (this is done automatically during training, but you can do it separately)
results = model.val()
