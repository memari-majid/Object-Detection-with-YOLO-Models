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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

def filter_grass_images(file_path):
    """Filter and keep only Grass images from the given file."""
    try:
        # Read the original file
        with open(file_path, 'r') as f:
            image_paths = f.readlines()
        
        # Filter only Grass images
        grass_images = [path.strip() for path in image_paths if 'Grass' in path]
        
        # Create a new file for Grass images
        grass_file = str(Path(file_path)).replace('.txt', '_grass.txt')
        with open(grass_file, 'w') as f:
            f.write('\n'.join(grass_images))
        
        console.print(f"[green]Filtered {len(grass_images)} Grass images from {len(image_paths)} total images in {file_path}")
        return grass_file
        
    except Exception as e:
        console.print(f"[bold red]Error filtering Grass images from {file_path}: {str(e)}")
        return None

def prepare_grass_dataset():
    """Prepare train, val, and test datasets with only Grass images."""
    try:
        dataset_files = {
            'train': Path('data/train.txt'),
            'val': Path('data/val.txt'),
            'test': Path('data/test.txt')
        }
        
        grass_files = {}
        
        for dataset_type, file_path in dataset_files.items():
            if not file_path.exists():
                console.print(f"[bold red]Error: {file_path} not found")
                return None
                
            grass_file = filter_grass_images(file_path)
            if not grass_file:
                return None
            grass_files[dataset_type] = grass_file
            
        return grass_files
        
    except Exception as e:
        console.print(f"[bold red]Error preparing Grass dataset: {str(e)}")
        return None

def setup_wandb_logging(model_name, run_type="training"):
    """Initialize Weights & Biases logging with better organization."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_base_name = model_name.replace('.pt', '')
    
    if run_type == "training":
        project_name = "WTB_Defect_Detection_Grass"  # Updated project name
        run_name = f"{model_base_name}/grass_training_{timestamp}"
    else:  # evaluation
        project_name = "WTB_Model_Evaluation_Grass"
        run_name = f"{model_base_name}/grass_evaluation_{timestamp}"
    
    return wandb.init(
        project=project_name,
        name=run_name,
        group=model_base_name,
        job_type=run_type,
        config={
            "model": model_name,
            "epochs": 10,
            "batch_size": 16,
            "image_size": 640,
            "dataset": "wind_turbine_blades_grass_only",
            "defect_types": ["crack", "erosion"]
        },
        reinit=True
    )

def analyze_misclassifications(model, grass_files, model_name, results_dir):
    """Analyze and save misclassified examples with detailed annotations."""
    try:
        # Create directory for misclassified examples
        misc_dir = Path(results_dir) / f"{model_name.replace('.pt', '')}_misclassified"
        misc_dir.mkdir(parents=True, exist_ok=True)
        
        # Run validation on test set to get predictions
        results = model.val(data=grass_files['test'])
        
        misclassified_info = []
        
        # Process each image in the test set
        with open(grass_files['test'], 'r') as f:
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

def train_and_evaluate(model_name, data_yaml_path):
    """Train and evaluate a single YOLO model using clean Grass images."""
    try:
        # Initialize training run
        train_run = setup_wandb_logging(model_name, "training_clean")
        console.print(f"\n[bold blue]Starting training for {model_name} with clean Grass images")
        
        # Load model
        model = YOLO(model_name)
        
        # Train with modified parameters for clean dataset
        results = model.train(
            data=data_yaml_path,
            epochs=150,  # Increased epochs for better learning
            imgsz=640,
            batch=8,    # Reduced batch size for smaller dataset
            device='0,1',
            workers=8,
            optimizer='AdamW',
            lr0=0.001,  # Reduced learning rate for stability
            lrf=0.0001,
            momentum=0.937,
            weight_decay=0.0005,
            save=True,
            plots=True,
            name=f"{model_name.replace('.pt', '')}_clean_results",
            project='WTB_Results_Clean',
            exist_ok=True,
            
            # Disable augmentations
            augment=False,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            
            # Training parameters
            cos_lr=True,
            warmup_epochs=5,  # Increased warmup
            label_smoothing=0.1,
            overlap_mask=True,
            patience=20  # Increased patience for better convergence
        )
        
        # Validate the model
        val_results = model.val(
            data=data_yaml_path,
            split='test',  # Use test split for final validation
            imgsz=640,
            batch=16,
            device='0,1'
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
                
                save_dir = Path('evaluation_results_grass') / model_name.replace('.pt', '')
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
                grass_files,
                model_name,
                f"WTB_Results_Grass/{model_name.replace('.pt', '')}_grass_results"
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

def create_grass_data_yaml(original_yaml_path):
    """Create a new data.yml file specifically for Grass dataset."""
    try:
        # Read original data.yml
        with open(original_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Get the absolute path to the project root
        project_root = Path.cwd().absolute()
        
        # Update paths for Grass-only files with absolute paths
        grass_config = {
            'path': str(project_root / 'data/obj_train_data_RGB'),  # Base dataset directory
            'train': str(project_root / 'data/train_grass.txt'),    # Train file
            'val': str(project_root / 'data/val_grass.txt'),        # Val file
            'test': str(project_root / 'data/test_grass.txt'),      # Test file
            'nc': data_config['nc'],                                # Keep original number of classes
            'names': data_config['names']                           # Keep original class names
        }
        
        # Save new Grass-specific data.yml
        grass_yaml_path = str(project_root / 'data/data_grass.yml')
        with open(grass_yaml_path, 'w') as f:
            yaml.dump(grass_config, f, default_flow_style=False)
            
        console.print(f"[green]Created Grass data YAML at {grass_yaml_path}")
        console.print(f"[blue]Config contents: {grass_config}")
            
        return grass_yaml_path
        
    except Exception as e:
        console.print(f"[bold red]Error creating Grass data.yml: {str(e)}")
        return None

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
    # Change this line to match the actual path from create_clean_dataset.py
    data_yaml_path = 'data_clean/data_clean.yml'  # This is where the file was actually created
    if not Path(data_yaml_path).exists():
        console.print("[bold red]Clean dataset not found. Run prepare_clean_dataset.py first.")
        return
    
    models = [
        'yolo8x.pt',
        'yolo9e.pt',
        'yolo10x.pt',
        'yolo11x.pt'
    ]
    
    all_results = []
    
    console.print("\n[bold blue]Available model files in directory:")
    for file in Path('.').glob('*.pt'):
        console.print(f"Found: {file}")
    
    for model_name in models:
        model_path = Path(model_name)
        if not model_path.exists():
            console.print(f"[bold red]Error: Model file {model_name} not found. Skipping...")
            continue
            
        if not Path(data_yaml_path).exists():
            console.print(f"[bold red]Error: {data_yaml_path} not found")
            continue
            
        console.print(f"[bold green]Found model file: {model_name}")
        success, result = train_and_evaluate(model_name, data_yaml_path)
        if success:
            all_results.append(result)
    
    if all_results:
        console.print("\n[bold yellow]Final Model Comparison (Grass Images Only):")
        comparison_df = pd.DataFrame(all_results)
        console.print(comparison_df.to_string())
        
        try:
            import matplotlib.pyplot as plt
            
            metrics_to_plot = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1-score']
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(models))
            width = 0.15
            multiplier = 0
            
            for metric in metrics_to_plot:
                offset = width * multiplier
                plt.bar(x + offset, comparison_df[metric], width, label=metric)
                multiplier += 1
            
            plt.xlabel('Models')
            plt.ylabel('Scores')
            plt.title('YOLO Models Performance Comparison (Grass Images Only)')
            plt.xticks(x + width * 2, models, rotation=45)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            
            plt.savefig('evaluation_results_grass/model_comparison.png')
            plt.close()
            
        except Exception as e:
            console.print(f"[bold red]Error creating comparison plot: {str(e)}")
    
    # Add this after preparing grass dataset
    grass_files = prepare_grass_dataset()
    if grass_files:
        console.print("\n[bold blue]Organizing and annotating images...")
        organize_annotated_images(grass_files, "annotated_grass_images")

if __name__ == "__main__":
    main()