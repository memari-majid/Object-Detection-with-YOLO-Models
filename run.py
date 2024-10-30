import os
import wandb
from ultralytics import YOLO
from pathlib import Path
import time
import torch
import numpy as np
from collections import deque
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()

class EarlyStoppingCallback:
    def __init__(self, 
                 patience=20,
                 min_epochs=50,
                 max_epochs=1000,
                 map_threshold=0.85,
                 loss_threshold=0.01,
                 smoothing_window=5):
        self.patience = patience
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.map_threshold = map_threshold
        self.loss_threshold = loss_threshold
        self.smoothing_window = smoothing_window
        
        self.val_losses = deque(maxlen=smoothing_window)
        self.map_scores = deque(maxlen=smoothing_window)
        self.best_map = 0
        self.best_epoch = 0
        self.stagnant_epochs = 0
        self.best_weights = None

    def check_stop_criteria(self, epoch, metrics, model_weights=None):
        current_map = metrics.get('mAP50-95', 0)
        current_loss = metrics.get('train/box_loss', float('inf'))
        
        self.map_scores.append(current_map)
        self.val_losses.append(current_loss)
        
        if current_map > self.best_map:
            self.best_map = current_map
            self.best_epoch = epoch
            self.stagnant_epochs = 0
            self.best_weights = model_weights
        else:
            self.stagnant_epochs += 1
        
        if epoch < self.min_epochs:
            return False, "Minimum epochs not reached"
            
        if epoch >= self.max_epochs:
            return True, "Maximum epochs reached"
            
        if current_map >= self.map_threshold:
            return True, f"mAP threshold {self.map_threshold} reached"
            
        if len(self.val_losses) == self.smoothing_window:
            loss_change = abs(np.mean(list(self.val_losses)[-self.smoothing_window:]) - 
                            np.mean(list(self.val_losses)[:-1]))
            if loss_change < self.loss_threshold:
                return True, f"Loss plateaued (change: {loss_change:.4f})"
                
        if self.stagnant_epochs >= self.patience:
            return True, f"No improvement for {self.patience} epochs"
            
        return False, "Training continuing"

def get_training_args(model_version):
    """Get version-specific training arguments"""
    common_args = {
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.3,
        'degrees': 10.0,
        'translate': 0.2,
        'scale': 0.9,
        'shear': 2.0,
        'perspective': 0.5,
        'flipud': 0.3,
        'fliplr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'warmup_epochs': 3,
        'close_mosaic': 10,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.1,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.2,
    }
    
    version_specific = {
        '8': {'mixup': 0.15, 'copy_paste': 0.3, 'perspective': 0.5},
        '9': {'mixup': 0.20, 'copy_paste': 0.4, 'perspective': 0.6},
        '10': {'mixup': 0.25, 'copy_paste': 0.5, 'perspective': 0.7},
        '11': {'mixup': 0.30, 'copy_paste': 0.6, 'perspective': 0.8}
    }
    
    args = common_args.copy()
    args.update(version_specific.get(model_version, version_specific['8']))
    return args

def train_and_evaluate(model_name, data_yaml_path, epochs, batch_size=16, image_size=640):
    try:
        # Initialize W&B run
        setup_wandb_logging(model_name, epochs, batch_size, image_size)
        
        # Load model
        model = YOLO(model_name)
        
        # Initialize early stopping
        early_stopping = EarlyStoppingCallback(
            patience=20,
            min_epochs=50,
            max_epochs=epochs,
            map_threshold=0.85,
            loss_threshold=0.01
        )
        
        # Training configuration
        train_args = {
            'data': data_yaml_path,
            'epochs': epochs,
            'imgsz': image_size,
            'batch': batch_size,
            'device': '0,1',
            'amp': True,
            'workers': 8,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.001,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'save': True,
            'plots': True,
            'callbacks': [early_stopping]  # Add the early stopping callback
        }
        
        # Train model
        results = model.train(**train_args)
        
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        wandb.finish()

def main():
    # Configuration
    base_dir = '/home/majid/PycharmProjects/E5/data/obj_train_data_RGB'
    data_yaml_path = '/home/majid/PycharmProjects/E5/data/data.yml'
    
    models = ['yolov8x.pt', 'yolov9e.pt', 'yolov10x.pt', 'yolo11x.pt']
    
    # CUDA settings
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    # Results tracking
    results_summary = []
    
    # Train models sequentially
    for model_name in models:
        console.print(f"\n[bold blue]Starting training for {model_name}")
        
        start_time = time.time()
        success, error = train_and_evaluate(
            model_name=model_name,
            data_yaml_path=data_yaml_path
        )
        
        duration = time.time() - start_time
        
        # Store results
        results_summary.append({
            'model': model_name,
            'success': success,
            'duration': duration,
            'error': error if not success else None
        })
    
    # Print summary
    console.print("\n[bold yellow]Training Summary:")
    for result in results_summary:
        status = "[green]Success" if result['success'] else f"[red]Failed: {result['error']}"
        console.print(f"Model: {result['model']}")
        console.print(f"Status: {status}")
        console.print(f"Duration: {result['duration']/3600:.2f} hours\n")

if __name__ == "__main__":
    main()