import os
from ultralytics import YOLO
from pathlib import Path
import yaml

# Load clean dataset configuration
data_yaml_path = 'data_clean/data_clean.yml'

# Verify data yaml exists
if not Path(data_yaml_path).exists():
    raise FileNotFoundError(f"Clean dataset YAML not found at {data_yaml_path}")

# Load and verify the configuration
with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)
    required_files = [data_config['train'], data_config['val'], data_config['test']]
    
    # Verify all required files exist
    for file_path in required_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Required file {file_path} not found")

# Load and train the YOLOv8 model with optimized parameters for small object detection
model = YOLO('yolov8x.pt')  # Load YOLOv8 'x' variant for more depth and capacity

# Train with parameters optimized for small object detection
model.train(
    data=data_yaml_path,        # Use clean dataset YAML
    epochs=150,                 # Increased epochs for more training cycles
    imgsz=1024,                # Higher image size to capture small defects
    batch=4,                    # Smaller batch size for better focus on details
    name='yolov8_clean_optimized',
    device='0,1',              # Use both GPUs
    optimizer='AdamW',         # Optimizer for stability
    lr0=0.0005,               # Lower learning rate for refined learning
    mosaic=1.0,               # Enable mosaic augmentation for varied data synthesis
    scale=0.5,                # Scale augmentation from 0.5x to 1.5x
    flipud=0.5,               # Vertical flip augmentation
    fliplr=0.5,               # Horizontal flip augmentation
    augment=True,             # General augmentation parameter
    cos_lr=True,              # Cosine annealing learning rate schedule for stability
    patience=30               # Increased patience for convergence
)

# Validation to evaluate performance on test set after training
results = model.val(
    data=data_yaml_path,       # Use clean dataset YAML
    imgsz=1024,               # Match validation image size to training size
    batch=8,                  # Use larger batch size for validation
    device='0,1'
)
