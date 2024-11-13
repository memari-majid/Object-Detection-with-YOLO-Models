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
    python run.py

With specific configuration:
    python run.py --dataset clean --preset yolo11_small_focused --models yolo11x

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

# Training Parameter Presets - Updated with valid batch size for 2 GPUs
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
        'workers': 8,
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
    }
}

# Default Configuration - Allow all models
DEFAULT_CONFIG = {
    'dataset': 'clean',
    'parameter_preset': 'yolo11_small_focused',  # Default to small object configuration
    'models': ['yolov8x', 'yolov9e', 'yolov10x', 'yolo11x'],  # All models available
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
    """
    Initialize Weights & Biases logging for training or evaluation runs.

    Parameters:
    -----------
    model_name : str
        Name of the YOLO model being trained/evaluated
    run_type : str, optional
        Type of run ('training' or 'evaluation'), default is 'training'

    Returns:
    --------
    wandb.Run
        Initialized Weights & Biases run object

    Notes:
    ------
    - Creates unique run names using timestamps
    - Includes all training parameters in configuration
    - Groups runs by model name for easier comparison
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_base_name = model_name.replace('.pt', '')
    
    if run_type == "training":
        project_name = "WTB_Defect_Detection_Clean"
        run_name = f"{model_base_name}/clean_training_{timestamp}"
    else:  # evaluation
        project_name = "WTB_Model_Evaluation_Clean"
        run_name = f"{model_base_name}/clean_evaluation_{timestamp}"
    
    return wandb.init(
        project=project_name,
        name=run_name,
        group=model_base_name,
        job_type=run_type,
        config={
            "model": model_name,
            **PARAMETER_PRESETS['yolo11_small_focused'],  # Use the specific preset
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

def train_and_evaluate(model_name, data_yaml_path, training_params, val_params):
    """
    Train and evaluate a YOLO model with specified parameters.

    Parameters:
    -----------
    model_name : str
        Path to the YOLO model weights
    data_yaml_path : str
        Path to dataset configuration YAML
    training_params : dict
        Training parameters and hyperparameters
    val_params : dict
        Validation parameters

    Returns:
    --------
    tuple
        (success: bool, metrics: dict)
        success: Whether training completed successfully
        metrics: Dictionary of evaluation metrics

    Features:
    ---------
    - Wandb logging for both training and evaluation
    - Comprehensive metric collection
    - Misclassification analysis
    - Result visualization
    """
    try:
        # Initialize training run
        train_run = setup_wandb_logging(model_name, "training_clean")
        console.print(f"\n[bold blue]Starting training for {model_name} with clean dataset")
        
        # Load model
        model = YOLO(model_name)
        
        # Train using defined parameters
        results = model.train(
            data=data_yaml_path,
            **training_params
        )
        
        # Validate the model using validation parameters
        val_results = model.val(
            data=data_yaml_path,
            split='test',
            **val_params
        )
        
        # Close training run
        train_run.finish()
        
        # Start evaluation run
        eval_run = setup_wandb_logging(model_name, "evaluation")
        
        # Extract and log metrics
        try:
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
            
            valid_metrics = {k: v for k, v in metrics.items() if v is not None}
            if valid_metrics:
                eval_run.log(valid_metrics)
                
                save_dir = Path('evaluation_results_clean') / model_name.replace('.pt', '')
                save_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([valid_metrics]).to_csv(
                    save_dir / 'metrics.csv',
                    index=False
                )
                
                console.print(f"[bold green]Evaluation completed for {model_name}")
                console.print("Metrics:", valid_metrics)
            else:
                console.print(f"[bold yellow]Warning: No valid metrics found for {model_name}")
            
            # After training, analyze misclassifications
            console.print("\n[bold blue]Analyzing misclassifications...")
            misclassification_stats = analyze_misclassifications(
                model, 
                data_yaml_path,
                model_name,
                f"WTB_Results_Clean/{model_name.replace('.pt', '')}_clean_results"
            )
            
            if misclassification_stats:
                metrics.update({
                    'total_misclassified': misclassification_stats['total_misclassified'],
                    'false_positives': misclassification_stats['false_positives'],
                    'false_negatives': misclassification_stats['false_negatives']
                })
            
            return True, valid_metrics
            
        except Exception as metric_error:
            console.print(f"[bold yellow]Warning: Error extracting metrics: {str(metric_error)}")
            return True, {'model': model_name, 'error': str(metric_error)}
            
    except Exception as e:
        console.print(f"[bold red]Error processing {model_name}: {str(e)}")
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

def main():
    """
    Main execution function for training and evaluation pipeline.

    Features:
    ---------
    - Command line argument parsing
    - Dataset validation
    - Model availability checking
    - Training execution
    - Results comparison and visualization
    - Error handling and logging

    Command Line Arguments:
    ---------------------
    --dataset : str
        Dataset configuration to use
    --preset : str
        Parameter preset for training
    --models : list
        Models to train and evaluate

    Outputs:
    --------
    - Training logs
    - Evaluation metrics
    - Comparison plots
    - Misclassification analysis
    """
    # Allow command line arguments or use defaults
    import argparse
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
    
    # Parse arguments only once
    args = parser.parse_args()
    
    console.print("\n[bold blue]Starting training with configuration:")
    console.print(f"Dataset: {args.dataset}")
    console.print(f"Preset: {args.preset}")
    console.print(f"Models: {args.models}")
    
    # Print the actual parameters being used
    console.print("\n[bold yellow]Training Parameters:")
    for key, value in PARAMETER_PRESETS[args.preset].items():
        console.print(f"{key}: {value}")
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIG[args.dataset]
    data_yaml_path = dataset_config['yaml_path']
    
    if not Path(data_yaml_path).exists():
        console.print(f"[bold red]Dataset not found at {data_yaml_path}")
        return
    
    # Verify the data files exist
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
        required_files = [data_config['train'], data_config['val'], data_config['test']]
        
    for file_path in required_files:
        if not Path(file_path).exists():
            console.print(f"[bold red]Error: Required file {file_path} not found")
            return
    
    # Get training parameters
    training_params = PARAMETER_PRESETS[args.preset].copy()
    training_params['project'] = dataset_config['project_name']
    val_params = VAL_PARAMS[args.preset].copy()
    
    # Print configuration
    console.print("\n[bold blue]Training Configuration:")
    console.print(f"Dataset: {dataset_config['description']}")
    console.print(f"Parameter Preset: {args.preset}")
    console.print(f"Models to train: {args.models}")
    
    all_results = []
    
    # Check for available models
    console.print("\n[bold blue]Available model files in directory:")
    for model_name in args.models:
        model_info = MODELS[model_name]
        model_path = Path(model_info['path'])
        if model_path.exists():
            console.print(f"[green]Found: {model_info['path']} - {model_info['description']}")
        else:
            console.print(f"[red]Missing: {model_info['path']} - {model_info['description']}")
    
    # Train models
    for model_name in args.models:
        model_info = MODELS[model_name]
        model_path = Path(model_info['path'])
        if not model_path.exists():
            console.print(f"[bold red]Error: Model file {model_info['path']} not found. Skipping...")
            continue
        
        console.print(f"[bold blue]Training {model_info['description']}")
        success, result = train_and_evaluate(model_info['path'], data_yaml_path, 
                                           training_params, val_params)
        if success:
            all_results.append(result)
    
    if all_results:
        console.print("\n[bold yellow]Final Model Comparison (Clean Dataset):")
        comparison_df = pd.DataFrame(all_results)
        console.print(comparison_df.to_string())
        
        # Create comparison plot
        try:
            import matplotlib.pyplot as plt
            
            metrics_to_plot = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1-score']
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(MODELS))
            width = 0.15
            multiplier = 0
            
            for metric in metrics_to_plot:
                offset = width * multiplier
                plt.bar(x + offset, comparison_df[metric], width, label=metric)
                multiplier += 1
            
            plt.xlabel('Models')
            plt.ylabel('Scores')
            plt.title('YOLO Models Performance Comparison (Clean Dataset)')
            plt.xticks(x + width * 2, MODELS.keys(), rotation=45)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            
            plt.savefig('evaluation_results_clean/model_comparison.png')
            plt.close()
            
        except Exception as e:
            console.print(f"[bold red]Error creating comparison plot: {str(e)}")

if __name__ == "__main__":
    main()