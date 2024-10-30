import os
from ultralytics import YOLO

# Base directory and dataset information
base_dir = '/home/majid/PycharmProjects/E5/data/obj_train_data_RGB'
data_yaml_path = '/home/majid/PycharmProjects/E5/data/data.yml'

# List of YOLO models
models = ['yolov8x.pt', 'yolov9e.pt', 'yolov10x.pt', 'yolo11x.pt']

# Define the range of epochs
epoch_range = range(100, 1001, 200)

# Directory to save evaluation metrics
metrics_dir = os.path.join(base_dir, 'metrics')
os.makedirs(metrics_dir, exist_ok=True)

# Iterate over each model and train with different numbers of epochs
for model_name in models:
    for epochs in epoch_range:
        # Load the YOLO model
        model = YOLO(model_name)
        
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name=f'{model_name[:-3]}_epoch_{epochs}',
            device='0,1'
        )
        
        # Validate the model
        validation = model.val()
        
        # Save evaluation metrics
        metrics_file = os.path.join(metrics_dir, f'{model_name[:-3]}_epochs_{epochs}_metrics.txt')
        with open(metrics_file, 'w') as f:
            for key, value in validation.items():
                f.write(f'{key}: {value}\n')
        
        # Save figures
        figures_dir = os.path.join(metrics_dir, f'{model_name[:-3]}_epoch_{epochs}_figures')
        os.makedirs(figures_dir, exist_ok=True)
        for figure in validation.get('plots', []):
            figure_path = os.path.join(figures_dir, f'{figure.name}.jpg')
            figure.save(figure_path)