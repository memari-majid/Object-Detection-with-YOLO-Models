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

def setup_wandb_logging(model_name, run_type="training"):
    """Initialize Weights & Biases logging with better organization."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_base_name = model_name.replace('.pt', '')
    
    if run_type == "training":
        project_name = "WTB_Defect_Detection"  # More descriptive project name
        run_name = f"{model_base_name}/training_{timestamp}"
    else:  # evaluation
        project_name = "WTB_Model_Evaluation"
        run_name = f"{model_base_name}/evaluation_{timestamp}"
    
    return wandb.init(
        project=project_name,
        name=run_name,
        group=model_base_name,  # Group runs by model
        job_type=run_type,      # Distinguish between training and evaluation
        config={
            "model": model_name,
            "epochs": 10,
            "batch_size": 16,
            "image_size": 640,
            "dataset": "wind_turbine_blades",
            "defect_types": ["crack", "erosion"]
        },
        reinit=True
    )

def train_and_evaluate(model_name, data_yaml_path):
    """Train and evaluate a single YOLO model with better organized logging."""
    try:
        # Initialize training run
        train_run = setup_wandb_logging(model_name, "training")
        console.print(f"\n[bold blue]Starting training for {model_name}")
        
        # Load model
        model = YOLO(model_name)
        
        # Training configuration with valid parameters only
        train_args = {
            'data': data_yaml_path,
            'epochs': 100,           # Increased from 10 to 100
            'imgsz': 640,
            'batch': 16,
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
            'name': f"{model_name.replace('.pt', '')}_results",
            'project': 'WTB_Results',
            'exist_ok': True,
            
            # Valid training improvements
            'cos_lr': True,          # Cosine LR scheduler
            'warmup_epochs': 3,      # Warmup epochs
            'close_mosaic': 10,      # Disable mosaic augmentation in final epochs
            'label_smoothing': 0.1,  # Label smoothing
            'overlap_mask': True,    # Mask overlap
            'val': True              # Run validation
        }
        
        # Train model and get results
        results = model.train(**train_args)
        
        # Close training run
        train_run.finish()
        
        # Start evaluation run
        eval_run = setup_wandb_logging(model_name, "evaluation")
        
        # Extract and log metrics
        metrics = {
            'model': model_name,
            'mAP50': results.maps[0],
            'mAP50-95': results.maps[1],
            'precision': results.results_dict['metrics/precision(B)'],
            'recall': results.results_dict['metrics/recall(B)'],
            'f1-score': results.results_dict['metrics/F1(B)'],
            'training_time': results.t_total
        }
        
        # Log metrics and artifacts
        eval_run.log(metrics)
        
        # Save model artifacts
        model_artifact = wandb.Artifact(
            name=f"{model_name.replace('.pt', '')}_model",
            type="model",
            description=f"Trained {model_name} for wind turbine blade defect detection"
        )
        model_artifact.add_file(f"WTB_Results/{model_name.replace('.pt', '')}_results/weights/best.pt")
        eval_run.log_artifact(model_artifact)
        
        # Save results locally
        save_dir = Path('evaluation_results') / model_name.replace('.pt', '')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed metrics
        pd.DataFrame([metrics]).to_csv(
            save_dir / 'metrics.csv',
            index=False
        )
        
        # Save training plots
        if hasattr(results, 'plot_metrics'):
            results.plot_metrics()
            plt.savefig(save_dir / 'training_metrics.png')
            plt.close()
        
        console.print(f"[bold green]Evaluation completed for {model_name}")
        console.print("Metrics:", metrics)
        
        eval_run.finish()
        return True, metrics
        
    except Exception as e:
        console.print(f"[bold red]Error processing {model_name}: {str(e)}")
        return False, str(e)

def main():
    # Configuration
    data_yaml_path = 'data/data.yml'
    
    # Updated model names to match exactly what's in the directory
    models = [
        'yolo8x.pt',     # matches yolo8x.pt
        'yolo9e.pt',     # matches yolo9e.pt
        'yolo10x.pt',    # matches yolo10x.pt
        'yolo11x.pt'     # matches yolo11x.pt
    ]
    
    # Results tracking
    all_results = []
    
    # Print available models
    console.print("\n[bold blue]Available model files in directory:")
    for file in Path('.').glob('*.pt'):
        console.print(f"Found: {file}")
    
    # Verify model files exist before training
    for model_name in models:
        model_path = Path(model_name)
        if not model_path.exists():
            console.print(f"[bold red]Error: Model file {model_name} not found. Skipping...")
            continue
            
        # Verify data.yml exists
        if not Path(data_yaml_path).exists():
            console.print(f"[bold red]Error: {data_yaml_path} not found")
            continue
            
        console.print(f"[bold green]Found model file: {model_name}")
        success, result = train_and_evaluate(model_name, data_yaml_path)
        if success:
            all_results.append(result)
    
    # Print final comparison
    if all_results:
        console.print("\n[bold yellow]Final Model Comparison:")
        comparison_df = pd.DataFrame(all_results)
        console.print(comparison_df.to_string())
        
        # Save final comparison plot
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
            plt.title('YOLO Models Performance Comparison')
            plt.xticks(x + width * 2, models, rotation=45)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('evaluation_results/model_comparison.png')
            plt.close()
            
        except Exception as e:
            console.print(f"[bold red]Error creating comparison plot: {str(e)}")

if __name__ == "__main__":
    main()