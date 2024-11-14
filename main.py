"""
Wind Turbine Blade Defect Detection Training Script
================================================

This script trains and evaluates YOLO models for detecting defects in wind turbine blades,
with a specific focus on small defect detection. It supports multiple dataset configurations
and parameter presets optimized for small object detection.

Key Features:
------------
1. Multiple dataset support (clean and full datasets)
2. Optimized parameter presets for small object detection
3. Comprehensive logging with Weights & Biases
4. Detailed misclassification analysis
5. Performance visualization and comparison

Usage:
------
Basic usage:
    python main.py

With specific configuration:
    python main.py --dataset clean --preset yolo11_small_focused --models yolo11x

Available arguments:
    --dataset: clean, full
    --preset: yolo11_small_objects, yolo11_small_focused
    --models: yolo11x

Parameter Presets:
----------------
1. yolo11_small_objects:
   - Base configuration optimized for small defect detection
   - Higher resolution (1536px)
   - Balanced augmentation strategy
   - Extended training duration (300 epochs)

2. yolo11_small_focused:
   - Aggressive configuration for very small defects
   - Maximum resolution (1920px)
   - Intensive augmentation
   - Extended training duration (400 epochs)

3. yolo11_high_res:
   - Enhanced configuration for high-resolution processing
   - Increased resolution (2560px)
   - Reduced batch size for higher resolution
   - Multi-GPU utilization
   - Automatic mixed precision
   - Gradient accumulation steps
   - Memory optimization
   - Advanced training features
   - Augmentation strategy optimized for high-resolution
   - Training stability
   - Loss functions
   - Advanced augmentation

Dataset Structure:
----------------
Expected directory structure:
    data_clean/
        ├── data_clean.yml
        ├── train_clean.txt
        ├── val_clean.txt
        └── test_clean.txt

    data/
        ├── data.yml
        ├── train.txt
        ├── val.txt
        └── test.txt

Requirements:
------------
- Python 3.8+
- PyTorch 1.7+
- Ultralytics YOLO
- Weights & Biases
- OpenCV
- Rich (for console output)
- Pandas
- Matplotlib

Environment Variables:
-------------------
- WANDB_API_KEY: Weights & Biases API key for logging

Notes:
-----
- Ensure sufficient GPU memory for high-resolution training
- Monitor GPU temperature during extended training sessions
- Backup trained models periodically
- Check wandb.ai for detailed training metrics and visualizations

Author: [Majid Memari]
Date: [2024-11-13]
Version: 1.0
"""

import os
import wandb
from ultralytics import YOLO
from pathlib import Path
import time
import torch
import numpy as np
import logging
from rich.console import Console
import pandas as pd
from datetime import datetime
import cv2
import json
import yaml
import argparse

console = Console()

# Dataset Configuration
DATASET_CONFIG = {
    'clean': {
        'yaml_path': 'data_clean/data_clean.yml',
        'description': 'Clean dataset with filtered images',
        'project_name': 'WTB_Results_Clean'
    },
    'full': {
        'yaml_path': 'data/data.yml',
        'description': 'Full dataset with all images',
        'project_name': 'WTB_Results_Full'
    }
}

# Model Configuration
MODELS = {
    'yolov8x': {
        'path': 'yolov8x.pt',
        'description': 'YOLOv8 extra large'
    },
    'yolov9e': {
        'path': 'yolov9e.pt',
        'description': 'YOLOv9 efficient'
    },
    'yolov10x': {
        'path': 'yolov10x.pt',
        'description': 'YOLOv10 extra large'
    },
    'yolo11x': {
        'path': 'yolo11x.pt',
        'description': 'YOLOv11 extra large'
    }
}

# Training Parameter Presets - Updated with valid YOLO arguments
PARAMETER_PRESETS = {
    'yolo11_small_focused': {  # Even more focused on tiny objects
        'epochs': 400,         # More epochs
        'imgsz': 1920,         # Maximum resolution
        'batch': 2,            # Changed from 1 to 2 to be divisible by 2 GPUs
        'optimizer': 'AdamW',
        'lr0': 0.0001,        # Very low learning rate
        'lrf': 0.000001,      # Very low final learning rate
        # Augmentation
        'mosaic': 1.0,
        'scale': 0.2,         # More aggressive scaling (0.2-1.8)
        'flipud': 0.7,
        'fliplr': 0.7,
        'augment': True,
        'degrees': 10.0,
        'translate': 0.3,
        'perspective': 0.001,
        'shear': 3.0,
        # Training stability
        'cos_lr': True,
        'patience': 100,      # Very high patience
        'workers': 10,
        'label_smoothing': 0.15,
        'overlap_mask': True,
        'warmup_epochs': 25,  # Extended warmup
        'weight_decay': 0.001,
        'momentum': 0.937,
        # Loss weights
        'box': 10.0,         # Increased box loss gain
        'cls': 0.3,          # Class loss gain
        'dfl': 2.0,          # DFL loss gain
        # Additional settings
        'close_mosaic': 15,
        'mixup': 0.2,
        'copy_paste': 0.4,
        'hsv_h': 0.015,
        'hsv_s': 0.8,
        'hsv_v': 0.5,
        'device': '0,1',
        'exist_ok': True,
        'project': 'WTB_Results'
    },
    'yolo11_high_res': {
        # Basic Training Parameters
        'epochs': 400,
        'imgsz': 1280,        # Reduced from 2560 to 1280 for tiles
        'batch': 2,           # Batch size per GPU
        'device': '0,1',
        'amp': True,
        'cache': True,
        
        # Memory Optimization
        'overlap_mask': False,
        'workers': 4,
        'batch_gpu': 1,
        
        # Tiling Parameters
        'tile_size': 1280,
        'tile_overlap': 0.2,
        'merge_iou_thresh': 0.5,
        
        # Optimizer Configuration
        'optimizer': 'AdamW',
        'lr0': 0.0001,        # Initial learning rate
        'lrf': 0.000001,      # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.001,
        
        # Training Features
        'rect': True,         # Rectangular training
        
        # Augmentation Strategy
        'mosaic': 1.0,
        'scale': 0.2,
        'flipud': 0.7,
        'fliplr': 0.7,
        'augment': True,
        'degrees': 10.0,
        'translate': 0.3,
        'perspective': 0.001,
        'shear': 3.0,
        
        # Training Stability
        'cos_lr': True,       # Cosine learning rate
        'patience': 100,
        'workers': 8,
        'label_smoothing': 0.15,
        'overlap_mask': True,
        'warmup_epochs': 25,
        
        # Loss Functions
        'box': 10.0,          # Box loss gain
        'cls': 0.3,           # Class loss gain
        'dfl': 2.0,           # DFL loss gain
        
        # Advanced Augmentation
        'close_mosaic': 15,
        'mixup': 0.2,
        'copy_paste': 0.4,
        'hsv_h': 0.015,
        'hsv_s': 0.8,
        'hsv_v': 0.5,
        
        # Project Settings
        'exist_ok': True,
        'project': 'WTB_Results'
    }
}

# Validation Parameter Presets - Updated for small objects
VAL_PARAMS = {
    'yolo11_small_objects': {
        'imgsz': 1536,
        'batch': 1,
        'device': '0,1',
        'conf': 0.15,        # Lower confidence threshold
        'iou': 0.45         # Lower IoU threshold
    },
    'yolo11_small_focused': {
        'imgsz': 1920,
        'batch': 2,          # Changed from 1 to 2 to be divisible by 2 GPUs
        'device': '0,1',
        'conf': 0.1,         # Very low confidence threshold
        'iou': 0.4          # Lower IoU threshold
    },
    'yolo11_high_res': {
        'imgsz': 2560,
        'batch': 1,
        'device': '0,1',
        'conf': 0.1,
        'iou': 0.4,
        'amp': True,
        'rect': True
    }
}

# Default Configuration - Set specific defaults
DEFAULT_CONFIG = {
    'dataset': 'clean',
    'parameter_preset': 'yolo11_high_res',  # Changed from yolo11_small_focused to yolo11_high_res
    'models': ['yolo11x'],  # Changed from all models to just yolo11x
    'device': '0,1'
}

# Update all parameter presets with common settings
for preset in PARAMETER_PRESETS.values():
    preset.update({
        'device': DEFAULT_CONFIG['device'],
        'exist_ok': True,
        'project': 'WTB_Results'  # Will be updated based on dataset
    })

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

def setup_wandb_logging(model_name, run_type="training"):
    """Initialize Weights & Biases logging with improved console output."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_base_name = model_name.replace('.pt', '')
    
    console.print(f"\n[bold cyan]Setting up logging for {model_base_name}...")
    
    if run_type == "training":
        project_name = "WTB_Defect_Detection_Clean"
        run_name = f"{model_base_name}/clean_training_{timestamp}"
    else:  # evaluation
        project_name = "WTB_Model_Evaluation_Clean"
        run_name = f"{model_base_name}/clean_evaluation_{timestamp}"
    
    console.print(f"[cyan]Project: {project_name}")
    console.print(f"[cyan]Run Name: {run_name}")
    
    return wandb.init(
        project=project_name,
        name=run_name,
        group=model_base_name,
        job_type=run_type,
        config={
            "model": model_name,
            **PARAMETER_PRESETS['yolo11_small_focused'],
            "dataset": "wind_turbine_blades_clean",
            "defect_types": ["defect"]
        },
        reinit=True
    )

def analyze_misclassifications(model, data_yaml_path, model_name, results_dir):
    """
    Analyze and visualize model misclassifications on the test set.

    Parameters:
    -----------
    model : YOLO
        Trained YOLO model
    data_yaml_path : str
        Path to dataset YAML configuration
    model_name : str
        Name of the model being analyzed
    results_dir : str
        Directory to save analysis results

    Returns:
    --------
    dict or None
        Statistics about misclassifications if successful, None if failed

    Saves:
    ------
    - Annotated images of misclassified examples
    - CSV summary of misclassifications
    - JSON file with statistics
    """
    try:
        # Create directory for misclassified examples
        misc_dir = Path(results_dir) / f"{model_name.replace('.pt', '')}_misclassified"
        misc_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test set path from yaml
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            test_path = data_config['test']
        
        # Run validation on test set
        results = model.val(data=data_yaml_path, split='test')
        
        misclassified_info = []
        
        # Process each image in the test set
        with open(test_path, 'r') as f:
            test_images = f.readlines()
        
        for idx, img_path in enumerate(test_images):
            img_path = img_path.strip()
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Get model predictions for this image
            results = model.predict(img_path, conf=0.25)
            
            # Load ground truth labels if they exist
            label_path = img_path.replace('.JPG', '.txt').replace('.jpg', '.txt')
            has_ground_truth = False
            ground_truth = []
            
            if Path(label_path).exists():
                with open(label_path, 'r') as f:
                    ground_truth = f.readlines()
                has_ground_truth = len(ground_truth) > 0
            
            # Get predictions
            predictions = results[0].boxes
            has_predictions = len(predictions) > 0
            
            # Determine if this is a misclassification
            is_misclassified = (has_ground_truth and not has_predictions) or \
                             (not has_ground_truth and has_predictions)
            
            if is_misclassified:
                # Create annotated image
                img_annotated = img.copy()
                
                # Add ground truth boxes in green
                if has_ground_truth:
                    for gt in ground_truth:
                        cls, x, y, w, h = map(float, gt.strip().split())
                        img_h, img_w = img.shape[:2]
                        x1 = int((x - w/2) * img_w)
                        y1 = int((y - h/2) * img_h)
                        x2 = int((x + w/2) * img_w)
                        y2 = int((y + h/2) * img_h)
                        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_annotated, 'Ground Truth', (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add prediction boxes in red
                if has_predictions:
                    for box in predictions:
                        box = box.xyxy[0].cpu().numpy()
                        cv2.rectangle(img_annotated, 
                                    (int(box[0]), int(box[1])), 
                                    (int(box[2]), int(box[3])), 
                                    (0, 0, 255), 2)
                        cv2.putText(img_annotated, 'Prediction', 
                                  (int(box[0]), int(box[1])-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Add text description
                text_lines = [
                    f"Image: {Path(img_path).name}",
                    f"Ground Truth Objects: {'Yes' if has_ground_truth else 'No'}",
                    f"Predicted Objects: {'Yes' if has_predictions else 'No'}",
                    "Type: " + ("False Positive" if not has_ground_truth and has_predictions 
                               else "False Negative" if has_ground_truth and not has_predictions 
                               else "Unknown")
                ]
                
                y_pos = 30
                for line in text_lines:
                    cv2.putText(img_annotated, line, (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_pos += 30
                
                # Save annotated image
                save_path = misc_dir / f"misclassified_{idx}.jpg"
                cv2.imwrite(str(save_path), img_annotated)
                
                # Store misclassification info
                misclassified_info.append({
                    'image_path': img_path,
                    'has_ground_truth': has_ground_truth,
                    'has_predictions': has_predictions,
                    'type': "False Positive" if not has_ground_truth and has_predictions 
                           else "False Negative" if has_ground_truth and not has_predictions 
                           else "Unknown"
                })
        
        # Save misclassification summary
        summary_df = pd.DataFrame(misclassified_info)
        summary_df.to_csv(misc_dir / 'misclassification_summary.csv', index=False)
        
        # Create and save statistics
        stats = {
            'total_misclassified': len(misclassified_info),
            'false_positives': len(summary_df[summary_df['type'] == 'False Positive']),
            'false_negatives': len(summary_df[summary_df['type'] == 'False Negative'])
        }
        
        with open(misc_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        console.print(f"[green]Saved {len(misclassified_info)} misclassified examples to {misc_dir}")
        console.print("Statistics:", stats)
        
        return stats
        
    except Exception as e:
        console.print(f"[bold red]Error analyzing misclassifications: {str(e)}")
        return None

def setup_training_optimizations(model):
    """Setup advanced training optimizations."""
    try:
        # Enable automatic mixed precision
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        
        # Enable activation checkpointing
        if hasattr(model, 'model'):
            from torch.utils.checkpoint import checkpoint_sequential
            model.model = checkpoint_sequential(model.model, 3)
        
        # Set memory efficient attention
        if hasattr(model, 'set_efficient_attention'):
            model.set_efficient_attention(True)
        
        return scaler
    except Exception as e:
        console.print(f"[yellow]Warning: Could not setup all optimizations: {str(e)}")
        return None

def train_with_gradient_accumulation(model, dataloader, optimizer, scaler, accumulation_steps=4):
    """Training step with gradient accumulation."""
    optimizer.zero_grad()
    accumulated_loss = 0
    
    for i, batch in enumerate(dataloader):
        with torch.cuda.amp.autocast():
            loss = model(batch)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        accumulated_loss += loss.item()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    return accumulated_loss * accumulation_steps

def train_and_evaluate(model_name, data_yaml_path, training_params, val_params):
    """Train and evaluate with tiled processing for high-resolution images."""
    try:
        # Initialize training run
        train_run = setup_wandb_logging(model_name, "training_clean")
        
        # Print training setup
        console.print("\n[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        console.print(f"[bold blue]Starting Training: {model_name}")
        console.print("[bold blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Print key parameters
        console.print("\n[yellow]Key Training Parameters:")
        important_params = ['epochs', 'imgsz', 'batch', 'device', 'lr0']
        for param in important_params:
            if param in training_params:
                console.print(f"[yellow]• {param}: {training_params[param]}")
        
        # Load model
        console.print("\n[cyan]Loading model...")
        model = YOLO(model_name)
        
        # Process training images in tiles
        console.print("\n[cyan]Processing training images in tiles...")
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Create temporary directories for tiles
        tile_dir = Path('tiles_temp')
        tile_dir.mkdir(exist_ok=True)
        
        # Process each image and create tiles
        for split in ['train', 'val']:
            split_file = data_config[split]
            with open(split_file, 'r') as f:
                image_paths = f.readlines()
                
            for img_path in image_paths:
                img_path = img_path.strip()
                tiles, positions, orig_size = process_high_res_image(
                    img_path,
                    target_size=training_params['imgsz'],
                    tile_size=training_params['tile_size'],
                    overlap=training_params['tile_overlap']
                )
                
                if tiles is not None:
                    # Save tiles and their metadata
                    for i, (tile, pos) in enumerate(zip(tiles, positions)):
                        tile_path = tile_dir / f"{Path(img_path).stem}_tile_{i}.jpg"
                        cv2.imwrite(str(tile_path), tile)
                        
        # Update data configuration to use tiles
        training_params['data'] = str(tile_dir)
        
        # Train the model
        console.print("\n[bold green]Starting training...")
        results = model.train(**training_params)
        
        # Validate using tiled approach
        console.print("\n[bold cyan]Starting validation...")
        val_results = []
        
        for img_path in val_images:
            tiles, positions, orig_size = process_high_res_image(
                img_path,
                target_size=val_params['imgsz'],
                tile_size=training_params['tile_size'],
                overlap=training_params['tile_overlap']
            )
            
            if tiles is not None:
                # Get predictions for each tile
                tile_predictions = []
                for tile in tiles:
                    pred = model.predict(tile, **val_params)
                    tile_predictions.append(pred)
                
                # Merge predictions
                final_predictions = merge_predictions(
                    tile_predictions,
                    positions,
                    orig_size,
                    training_params['merge_iou_thresh']
                )
                
                val_results.append(final_predictions)
        
        # Close training run
        train_run.finish()
        
        # Start evaluation run
        console.print("\n[bold magenta]Starting evaluation...")
        eval_run = setup_wandb_logging(model_name, "evaluation")
        
        # Extract and format metrics
        metrics = {
            'model': model_name,
            'training_time': results.t_total if hasattr(results, 't_total') else None
        }
        
        if hasattr(results, 'results_dict'):
            metrics.update({
                'precision': results.results_dict.get('metrics/precision(B)', None),
                'recall': results.results_dict.get('metrics/recall(B)', None),
                'f1-score': results.results_dict.get('metrics/F1(B)', None),
            })
        
        if hasattr(results, 'maps') and results.maps is not None:
            metrics.update({
                'mAP50': results.maps[0] if len(results.maps) > 0 else None,
                'mAP50-95': results.maps[1] if len(results.maps) > 1 else None,
            })
        
        # Print results summary
        console.print("\n[bold green]Training Complete!")
        console.print("[bold yellow]Final Metrics:")
        for metric, value in metrics.items():
            if value is not None:
                console.print(f"[yellow]• {metric}: {value:.4f}" if isinstance(value, float) else f"[yellow]• {metric}: {value}")
        
        return True, metrics
        
    except Exception as e:
        console.print(f"\n[bold red]Error during training: {str(e)}")
        return False, str(e)
    finally:
        if 'train_run' in locals():
            train_run.finish()
        if 'eval_run' in locals():
            eval_run.finish()

def create_base_data_yaml():
    """Create base data.yml if it doesn't exist."""
    data_yaml_path = Path('data/data.yml')
    if not data_yaml_path.exists():
        base_config = {
            'path': 'data/obj_train_data_RGB',  # Path to images directory
            'train': 'data/train.txt',
            'val': 'data/val.txt',
            'test': 'data/test.txt',
            'nc': 1,  # Number of classes
            'names': ['defect']  # Class names
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        
        console.print(f"[green]Created base data.yml at {data_yaml_path}")
    return str(data_yaml_path)

def organize_annotated_images(grass_files, output_base_dir):
    """Organize images into folders based on whether they contain objects or not."""
    try:
        # Create output directories
        output_base = Path(output_base_dir)
        object_dir = output_base / "with_objects"
        no_object_dir = output_base / "without_objects"
        
        for dir_path in [object_dir, no_object_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Process each image in the dataset
        with open(grass_files['train'], 'r') as f:
            image_paths = f.readlines()
            
        for img_path in image_paths:
            img_path = img_path.strip()
            img = cv2.imread(img_path)
            
            if img is None:
                console.print(f"[yellow]Warning: Could not read image {img_path}")
                continue
                
            # Get corresponding label path
            label_path = img_path.replace('.JPG', '.txt').replace('.jpg', '.txt')
            has_objects = False
            
            # Create annotated image
            img_annotated = img.copy()
            
            if Path(label_path).exists():
                with open(label_path, 'r') as f:
                    annotations = f.readlines()
                    if annotations:
                        has_objects = True
                        # Draw boxes for each annotation
                        for ann in annotations:
                            cls, x, y, w, h = map(float, ann.strip().split())
                            img_h, img_w = img.shape[:2]
                            x1 = int((x - w/2) * img_w)
                            y1 = int((y - h/2) * img_h)
                            x2 = int((x + w/2) * img_w)
                            y2 = int((y + h/2) * img_h)
                            
                            # Draw rectangle and label
                            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_annotated, 'Defect', (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save to appropriate directory
            save_dir = object_dir if has_objects else no_object_dir
            save_path = save_dir / Path(img_path).name
            
            # Add text overlay indicating presence/absence of objects
            status_text = "Contains Objects" if has_objects else "No Objects"
            cv2.putText(img_annotated, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if has_objects else (0, 0, 255), 2)
            
            cv2.imwrite(str(save_path), img_annotated)
        
        # Count images in each directory
        object_count = len(list(object_dir.glob('*.jpg'))) + len(list(object_dir.glob('*.JPG')))
        no_object_count = len(list(no_object_dir.glob('*.jpg'))) + len(list(no_object_dir.glob('*.JPG')))
        
        console.print(f"[green]Successfully organized images:")
        console.print(f"Images with objects: {object_count}")
        console.print(f"Images without objects: {no_object_count}")
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]Error organizing images: {str(e)}")
        return False

def check_requirements():
    """
    Check if all required packages are installed and GPU is available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            console.print("[bold red]Warning: CUDA is not available. Training will be slow on CPU.")
            return False
        
        required_packages = {
            'ultralytics': 'ultralytics',
            'wandb': 'wandb',
            'opencv-python': 'cv2',
            'rich': 'rich',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'pyyaml': 'yaml'
        }
        
        missing_packages = []
        for package, import_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            console.print(f"[bold red]Missing required packages: {', '.join(missing_packages)}")
            console.print("Install using: pip install " + " ".join(missing_packages))
            return False
            
        return True
    except Exception as e:
        console.print(f"[bold red]Error checking requirements: {str(e)}")
        return False

def verify_dataset_structure(data_yaml_path):
    """
    Verify that all required dataset files and directories exist.
    """
    try:
        if not Path(data_yaml_path).exists():
            console.print(f"[bold red]Dataset YAML not found at {data_yaml_path}")
            return False
            
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        required_keys = ['train', 'val', 'test', 'nc', 'names']
        missing_keys = [key for key in required_keys if key not in data_config]
        if missing_keys:
            console.print(f"[bold red]Missing required keys in {data_yaml_path}: {missing_keys}")
            return False
            
        for split in ['train', 'val', 'test']:
            file_path = data_config[split]
            if not Path(file_path).exists():
                console.print(f"[bold red]{split} file not found at {file_path}")
                return False
                
            # Verify that image paths in the file exist
            with open(file_path, 'r') as f:
                image_paths = f.readlines()
                for img_path in image_paths[:5]:  # Check first 5 images as sample
                    if not Path(img_path.strip()).exists():
                        console.print(f"[bold red]Image not found: {img_path.strip()}")
                        return False
        
        return True
    except Exception as e:
        console.print(f"[bold red]Error verifying dataset structure: {str(e)}")
        return False

def verify_model_file(model_path):
    """
    Verify that the model file exists and is valid.
    """
    try:
        if not Path(model_path).exists():
            console.print(f"[bold red]Model file not found: {model_path}")
            return False
            
        # Try to load model to verify it's valid
        model = YOLO(model_path)
        return True
    except Exception as e:
        console.print(f"[bold red]Error verifying model file {model_path}: {str(e)}")
        return False

def setup_gpu_environment():
    """
    Setup optimal GPU environment for training.
    """
    try:
        import torch
        
        # Set device
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus < 2:
                console.print("[yellow]Warning: Less than 2 GPUs available. Adjusting configuration...")
                # Update device settings in presets
                for preset in PARAMETER_PRESETS.values():
                    preset['device'] = '0'
                for preset in VAL_PARAMS.values():
                    preset['device'] = '0'
                    
            # Set CUDA device
            torch.cuda.set_device(0)
            
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Print GPU info
            for i in range(n_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # Convert to GB
                console.print(f"[green]GPU {i}: {gpu_name} ({memory:.1f} GB)")
                
        return True
    except Exception as e:
        console.print(f"[bold red]Error setting up GPU environment: {str(e)}")
        return False

def process_high_res_image(image_path, target_size=2560, tile_size=1280, overlap=0.2):
    """
    Process high-resolution images using tiling strategy.
    
    Parameters:
    -----------
    image_path : str
        Path to the high-resolution image
    target_size : int
        Target size for the longest dimension
    tile_size : int
        Size of each tile
    overlap : float
        Overlap percentage between tiles (0.0 to 1.0)
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Calculate overlap in pixels
        overlap_px = int(tile_size * overlap)
        
        # Get original dimensions
        h, w = image.shape[:2]
        
        # Calculate number of tiles needed
        n_tiles_h = max(1, (h - overlap_px) // (tile_size - overlap_px))
        n_tiles_w = max(1, (w - overlap_px) // (tile_size - overlap_px))
        
        tiles = []
        tile_positions = []
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile coordinates
                x1 = j * (tile_size - overlap_px)
                y1 = i * (tile_size - overlap_px)
                x2 = min(x1 + tile_size, w)
                y2 = min(y1 + tile_size, h)
                
                # Extract tile
                tile = image[y1:y2, x1:x2]
                
                # Pad if necessary
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                tiles.append(tile)
                tile_positions.append((x1, y1, x2, y2))
        
        return tiles, tile_positions, (h, w)
        
    except Exception as e:
        console.print(f"[bold red]Error processing image {image_path}: {str(e)}")
        return None, None, None

def merge_predictions(predictions, tile_positions, original_size, iou_threshold=0.5):
    """
    Merge predictions from multiple tiles into a single prediction.
    
    Parameters:
    -----------
    predictions : list
        List of predictions from each tile
    tile_positions : list
        List of tile positions (x1, y1, x2, y2)
    original_size : tuple
        Original image size (height, width)
    iou_threshold : float
        IoU threshold for merging overlapping predictions
    """
    merged_predictions = []
    
    # Convert tile predictions to global coordinates
    for pred, (tx1, ty1, _, _) in zip(predictions, tile_positions):
        if pred is not None and len(pred) > 0:
            # Adjust coordinates to global image space
            pred[:, [0, 2]] += tx1  # adjust x coordinates
            pred[:, [1, 3]] += ty1  # adjust y coordinates
            merged_predictions.extend(pred)
    
    if not merged_predictions:
        return np.array([])
    
    # Convert to numpy array
    merged_predictions = np.vstack(merged_predictions)
    
    # Apply NMS to remove overlapping predictions
    from ultralytics.utils.ops import non_max_suppression
    final_predictions = non_max_suppression(
        torch.from_numpy(merged_predictions).float(),
        iou_thres=iou_threshold,
        conf_thres=0.1
    )[0].numpy()
    
    return final_predictions

def main():
    """
    Main execution function for training and evaluation pipeline.
    """
    # Check requirements first
    if not check_requirements():
        return
        
    # Setup GPU environment
    if not setup_gpu_environment():
        return
        
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train YOLO models on WTB dataset')
    parser.add_argument('--dataset', choices=list(DATASET_CONFIG.keys()), 
                       default=DEFAULT_CONFIG['dataset'],
                       help='Dataset to use for training')
    parser.add_argument('--preset', choices=list(PARAMETER_PRESETS.keys()),
                       default=DEFAULT_CONFIG['parameter_preset'],
                       help='Parameter preset to use')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                       default=DEFAULT_CONFIG['models'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    # Print configuration
    console.print("\n[bold blue]Starting training with configuration:")
    console.print(f"Dataset: {args.dataset}")
    console.print(f"Preset: {args.preset}")
    console.print(f"Models: {args.models}")
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIG[args.dataset]
    data_yaml_path = dataset_config['yaml_path']
    
    # Verify dataset structure
    if not verify_dataset_structure(data_yaml_path):
        return
    
    # Get training parameters
    training_params = PARAMETER_PRESETS[args.preset].copy()
    training_params['project'] = dataset_config['project_name']
    val_params = VAL_PARAMS[args.preset].copy()
    
    all_results = []
    
    # Train models
    for model_name in args.models:
        model_info = MODELS[model_name]
        
        # Verify model file
        if not verify_model_file(model_info['path']):
            continue
            
        console.print(f"\n[bold blue]Training {model_info['description']}")
        
        try:
            success, result = train_and_evaluate(model_info['path'], data_yaml_path, 
                                               training_params, val_params)
            if success:
                all_results.append(result)
        except Exception as e:
            console.print(f"[bold red]Error training {model_name}: {str(e)}")
            continue
    
    # Generate results
    if all_results:
        console.print("\n[bold yellow]Final Model Comparison (Clean Dataset):")
        comparison_df = pd.DataFrame(all_results)
        console.print(comparison_df.to_string())
        
        # Save results
        results_dir = Path('evaluation_results_clean')
        results_dir.mkdir(exist_ok=True)
        
        # Save comparison DataFrame
        comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
        
        # Create comparison plot
        try:
            import matplotlib.pyplot as plt
            
            metrics_to_plot = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1-score']
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(comparison_df))
            width = 0.15
            multiplier = 0
            
            for metric in metrics_to_plot:
                if metric in comparison_df.columns:
                    offset = width * multiplier
                    plt.bar(x + offset, comparison_df[metric], width, label=metric)
                    multiplier += 1
            
            plt.xlabel('Models')
            plt.ylabel('Scores')
            plt.title('YOLO Models Performance Comparison (Clean Dataset)')
            plt.xticks(x + width * 2, comparison_df['model'], rotation=45)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            
            plt.savefig(results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            console.print(f"[bold red]Error creating comparison plot: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Training interrupted by user")
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {str(e)}")
    finally:
        # Cleanup
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()