"""
Difficult Image Detection Tool

This module identifies challenging or problematic images in a dataset by analyzing predictions
from multiple YOLO models. It helps improve dataset quality by finding images that are:
- Inconsistently detected across different models
- Have ground truth labels but are frequently missed
- Generate disagreement between models

Key Features:
- Multi-model analysis for robust difficulty assessment
- Ground truth verification
- Visual result generation with annotations
- Comprehensive analysis report generation
- Support for multiple dataset splits (train/val/test)

Dependencies:
    - opencv-python (cv2)
    - ultralytics (YOLO)
    - pandas
    - rich
    - pathlib
"""

import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import pandas as pd
from rich.console import Console
from rich.progress import track
import shutil

console = Console()

def load_all_images():
    """
    Load image paths from all dataset splits (train, val, test).
    
    Returns:
        list: A list of all unique image paths found in the dataset.
        
    Note:
        Expects image paths to be stored in text files named:
        - data/train_grass.txt
        - data/val_grass.txt
        - data/test_grass.txt
    """
    image_paths = set()
    for split in ['train', 'val', 'test']:
        file_path = f'data/{split}_grass.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                paths = [line.strip() for line in f.readlines()]
                image_paths.update(paths)
    return list(image_paths)

def analyze_image_quality(model_paths, confidence_threshold=0.25):
    """
    Analyze images using multiple models to identify difficult cases.
    
    This function processes each image through multiple YOLO models and identifies
    cases where:
    1. Models disagree significantly on detections
    2. Ground truth exists but models fail to detect
    3. Detection confidence varies significantly between models
    
    Args:
        model_paths (list): List of paths to YOLO model weights
        confidence_threshold (float): Minimum confidence score for detections (default: 0.25)
    
    Returns:
        list: List of dictionaries containing information about difficult images
    
    Directory Structure Created:
    difficult_images_analysis/
    ├── difficult_images_analysis.csv
    ├── original_images/
    │   └── [copied difficult images]
    └── annotated_images/
        └── [annotated difficult images]
    
    The analysis creates:
    - CSV report with difficulty metrics
    - Copies of difficult images
    - Annotated versions showing:
        - Ground truth boxes (green)
        - Model agreement scores
        - Difficulty reasons
    """
    
    # Load all image paths
    image_paths = load_all_images()
    console.print(f"[green]Found {len(image_paths)} total images to analyze")
    
    # Load models
    models = [YOLO(model_path) for model_path in model_paths]
    
    results_dir = Path('difficult_images_analysis')
    results_dir.mkdir(exist_ok=True)
    
    difficult_images = []
    
    for img_path in track(image_paths, description="Analyzing images..."):
        try:
            img = cv2.imread(img_path)
            if img is None:
                console.print(f"[red]Could not read image: {img_path}")
                continue
                
            # Get ground truth labels
            label_path = img_path.replace('.JPG', '.txt').replace('.jpg', '.txt')
            has_ground_truth = False
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    ground_truth = f.readlines()
                has_ground_truth = len(ground_truth) > 0
            
            # Check predictions from all models
            model_detections = []
            for model in models:
                results = model.predict(img_path, conf=confidence_threshold)
                model_detections.append(len(results[0].boxes) > 0)
            
            # Calculate agreement between models
            detection_agreement = sum(model_detections) / len(models)
            
            # Identify difficult cases based on defined criteria
            is_difficult = False
            difficulty_reason = []
            
            # Case 1: Has ground truth but models fail to detect
            if has_ground_truth and detection_agreement < 0.5:
                is_difficult = True
                difficulty_reason.append("Models failed to detect existing defects")
            
            # Case 2: Models disagree significantly
            if 0.2 < detection_agreement < 0.8:
                is_difficult = True
                difficulty_reason.append("Models disagree on detections")
            
            if is_difficult:
                # Save information about difficult image
                difficult_images.append({
                    'image_path': img_path,
                    'has_ground_truth': has_ground_truth,
                    'model_agreement': detection_agreement,
                    'reason': ' & '.join(difficulty_reason)
                })
                
                # Copy image to results directory
                dest_path = results_dir / Path(img_path).name
                shutil.copy2(img_path, dest_path)
                
                # If has ground truth, copy label file too
                if has_ground_truth:
                    label_dest = str(dest_path).replace('.JPG', '.txt').replace('.jpg', '.txt')
                    shutil.copy2(label_path, label_dest)
                
                # Create visualization with annotations
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
                
                # Add text description
                text_lines = [
                    f"Model Agreement: {detection_agreement:.2f}",
                    f"Ground Truth Objects: {'Yes' if has_ground_truth else 'No'}",
                    f"Difficulty: {' & '.join(difficulty_reason)}"
                ]
                
                y_pos = 30
                for line in text_lines:
                    cv2.putText(img_annotated, line, (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_pos += 30
                
                # Save annotated image
                cv2.imwrite(str(results_dir / f"annotated_{Path(img_path).name}"), img_annotated)
        
        except Exception as e:
            console.print(f"[red]Error processing {img_path}: {str(e)}")
    
    # Save analysis results
    if difficult_images:
        df = pd.DataFrame(difficult_images)
        df.to_csv(results_dir / 'difficult_images_analysis.csv', index=False)
        
        # Print summary
        console.print("\n[yellow]Analysis Summary:")
        console.print(f"Total images analyzed: {len(image_paths)}")
        console.print(f"Difficult images found: {len(difficult_images)}")
        console.print("\nDifficulty reasons distribution:")
        console.print(df['reason'].value_counts())
    
    return difficult_images

if __name__ == "__main__":
    # List of model paths to use for analysis
    model_paths = [
        'yolo8x.pt',
        'yolo9e.pt',
        'yolo10x.pt',
        'yolo11x.pt'
    ]
    
    # Run analysis
    difficult_images = analyze_image_quality(model_paths)