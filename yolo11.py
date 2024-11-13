import os
from ultralytics import YOLO

# Define the directory where images and annotations are located
base_dir = '/home/majid/PycharmProjects/E5/data/obj_train_data_RGB'  # Path to the directory containing images and annotations

# Define file paths to train, val, and test text files
train_file = '/home/majid/PycharmProjects/E5/data/train.txt'
val_file = '/home/majid/PycharmProjects/E5/data/val.txt'
test_file = '/home/majid/PycharmProjects/E5/data/test.txt'

# Load and train the YOLOv11 model
model = YOLO('yolo11x.pt')  # Load the YOLOv11 'x' variant for more depth and capacity

# Set parameters optimized for small object detection
model.train(
    data='/home/majid/PycharmProjects/E5/data/data.yml',  # Path to data YAML
    epochs=200,                # Increased epochs for more training cycles
    imgsz=1024,                # Higher image size to capture small defects
    batch=4,                   # Smaller batch size for better focus on details
    name='yolov11_small_object_optimized',
    device='0,1',              # Use both GPUs
    optimizer='AdamW',         # Optimizer for stability
    lr0=0.0005,                # Lower learning rate for refined learning
    mosaic=1.0,                # Enable mosaic augmentation for varied data synthesis
    scale=0.5,                 # Scale augmentation from 0.5x to 1.5x
    flipud=0.5,                # Vertical flip augmentation
    fliplr=0.5,                # Horizontal flip augmentation
    augment=True,              # General augmentation parameter for training
    cos_lr=True,               # Cosine annealing learning rate schedule for stability
    patience=30                # Increased patience for convergence
)

# Validation to evaluate performance on test set
results = model.val(
    data='/home/majid/PycharmProjects/E5/data/data.yml',  # Path to data YAML
    imgsz=1024,                # Match validation image size to training size
    batch=8,                   # Use larger batch size for validation
    device='0,1'
)
