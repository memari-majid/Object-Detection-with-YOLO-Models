"""
Wind Turbine Blade (WTB) Analysis Tool

This module provides functionality to analyze classification results from machine learning models
that detect defects in wind turbine blades. It processes saved prediction results, identifies 
misclassified examples, and generates comprehensive analysis reports.

Key Features:
- Analyzes prediction results from multiple models
- Identifies and saves misclassified examples with high confidence
- Generates confusion matrices and statistical summaries
- Creates annotated visualizations of misclassified examples
- Saves both original and annotated images for comparison

Dependencies:
    - opencv-python (cv2)
    - pandas
    - rich
    - pathlib
    - json
    - shutil
"""

import os
from pathlib import Path
import cv2
import json
import pandas as pd
from rich.console import Console
import shutil

console = Console()

def analyze_saved_results(results_dir="WTB_Results", confidence_threshold=0.5, max_examples=50):
    """
    Analyze classification results and extract misclassified examples from multiple models.

    This function processes prediction results from different models, identifies cases where
    the model made high-confidence mistakes, and generates comprehensive analysis reports
    including visualizations and statistics.

    Args:
        results_dir (str): Directory containing the model results. Default is "WTB_Results".
        confidence_threshold (float): Minimum confidence score to consider a prediction.
                                    Range: 0.0 to 1.0. Default is 0.5.
        max_examples (int): Maximum number of misclassified examples to save per model.
                          Default is 50.

    Directory Structure Created:
    misclassification_analysis/
    ├── model_1_name/
    │   ├── misc_1.jpg           # Annotated misclassified image
    │   ├── misc_1_original.jpg  # Original image
    │   ├── misclassified_summary.csv
    │   └── confusion_matrix.csv
    ├── model_2_name/
    │   └── ...
    ├── overall_analysis.csv
    └── analysis_stats.json

    Returns:
        None. Results are saved to disk in the 'misclassification_analysis' directory.

    Raises:
        FileNotFoundError: If the results directory doesn't exist
        JSONDecodeError: If prediction files are corrupted
        Exception: For other processing errors, with detailed error messages
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        console.print("[bold red]Results directory not found!")
        return

    # Create a new directory for the analysis
    analysis_dir = Path("misclassification_analysis")
    analysis_dir.mkdir(exist_ok=True)

    all_misclassified = []

    # Iterate through all model results
    for model_dir in results_path.glob("*_results"):
        model_name = model_dir.name.replace("_results", "")
        console.print(f"\n[bold blue]Analyzing results for model: {model_name}")

        val_dir = model_dir / "val"
        if not val_dir.exists():
            console.print(f"[yellow]No validation results found for {model_name}")
            continue

        pred_file = val_dir / "predictions.json"
        if not pred_file.exists():
            console.print(f"[yellow]No predictions file found for {model_name}")
            continue

        try:
            # Load and process predictions
            with open(pred_file, 'r') as f:
                predictions = json.load(f)

            # Extract misclassified examples with high confidence
            model_misclassified = []
            for pred in predictions:
                if pred.get('confidence', 0) > confidence_threshold and not pred.get('correct', True):
                    model_misclassified.append({
                        'model': model_name,
                        'image_path': pred['image_path'],
                        'predicted': pred.get('predicted_class', 'unknown'),
                        'actual': pred.get('true_class', 'unknown'),
                        'confidence': pred.get('confidence', 0)
                    })

            # Sort by confidence and limit examples
            model_misclassified.sort(key=lambda x: x['confidence'], reverse=True)
            model_misclassified = model_misclassified[:max_examples]
            all_misclassified.extend(model_misclassified)

            # Create model-specific directory
            model_analysis_dir = analysis_dir / model_name
            model_analysis_dir.mkdir(exist_ok=True)

            # Process and save misclassified examples
            for idx, misc in enumerate(model_misclassified):
                try:
                    img_path = misc['image_path']
                    if not Path(img_path).exists():
                        console.print(f"[yellow]Image not found: {img_path}")
                        continue

                    # Read and annotate image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    # Add informative annotations
                    text = [
                        f"Model: {misc['model']}",
                        f"Predicted: {misc['predicted']}",
                        f"Actual: {misc['actual']}",
                        f"Confidence: {misc['confidence']:.2f}"
                    ]

                    y0, dy = 30, 30
                    for i, line in enumerate(text):
                        y = y0 + i*dy
                        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 0, 255), 2)

                    # Save annotated and original images
                    save_path = model_analysis_dir / f"misc_{idx+1}.jpg"
                    cv2.imwrite(str(save_path), img)
                    orig_save_path = model_analysis_dir / f"misc_{idx+1}_original.jpg"
                    shutil.copy2(img_path, orig_save_path)

                except Exception as e:
                    console.print(f"[yellow]Error processing image {idx}: {e}")

            # Generate and save analysis reports
            df = pd.DataFrame(model_misclassified)
            df.to_csv(model_analysis_dir / "misclassified_summary.csv", index=False)

            # Create and save confusion matrix
            confusion = pd.crosstab(
                pd.Series([m['actual'] for m in model_misclassified], name='Actual'),
                pd.Series([m['predicted'] for m in model_misclassified], name='Predicted')
            )
            console.print(f"\n[bold green]Confusion Matrix for {model_name}:")
            console.print(confusion)
            confusion.to_csv(model_analysis_dir / "confusion_matrix.csv")

        except Exception as e:
            console.print(f"[bold red]Error analyzing {model_name}: {e}")

    # Generate overall analysis if there are misclassified examples
    if all_misclassified:
        overall_df = pd.DataFrame(all_misclassified)
        overall_df.to_csv(analysis_dir / "overall_analysis.csv", index=False)

        # Calculate and save statistics
        stats = {
            'total_misclassified': len(all_misclassified),
            'misclassified_by_model': overall_df.groupby('model').size().to_dict(),
            'most_common_mistakes': overall_df.groupby(['actual', 'predicted']).size().sort_values(ascending=False).head(10).to_dict()
        }

        with open(analysis_dir / "analysis_stats.json", 'w') as f:
            json.dump(stats, f, indent=4)

        # Print summary statistics
        console.print("\n[bold green]Analysis Complete!")
        console.print(f"Results saved to: {analysis_dir}")
        console.print("\n[bold blue]Summary Statistics:")
        console.print(f"Total misclassified examples: {stats['total_misclassified']}")
        console.print("\nMisclassified by model:")
        for model, count in stats['misclassified_by_model'].items():
            console.print(f"{model}: {count}")

if __name__ == "__main__":
    analyze_saved_results(
        results_dir="WTB_Results",
        confidence_threshold=0.5,  # Minimum confidence to consider
        max_examples=50  # Maximum number of examples per model
    ) 