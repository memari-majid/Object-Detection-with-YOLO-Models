# Wind Turbine Blade Defect Detection with YOLO Models

## Project Overview

This project implements advanced object detection for wind turbine blade defects using YOLO models (v8-v11), with a specific focus on small defect detection. The implementation includes optimized configurations for high-resolution images and multi-GPU support.

### Key Features

- **Multiple YOLO Model Support**: 
  - YOLOv8x (extra large)
  - YOLOv9e (efficient)
  - YOLOv10x (extra large)
  - YOLOv11x (extra large)

- **Small Object Detection Optimization**:
  - High-resolution image processing (up to 1920px)
  - Specialized parameter presets for small defects
  - Advanced augmentation techniques

- **Multi-GPU Training**:
  - Distributed training across available GPUs
  - Optimized batch sizes for multi-GPU setup
  - Efficient resource utilization

## Quick Start

### Basic Usage

Train a specific model with optimized small object detection parameters:
```bash
python run.py --dataset clean --preset yolo11_small_focused --models yolo11x
```

### Available Parameters

#### Dataset Options
- `clean`: Filtered, high-quality dataset
- `full`: Complete dataset

#### Parameter Presets
- `yolo11_small_focused`: Optimized for very small defects
  - Resolution: 1920px
  - Batch size: 2 (optimized for 2 GPUs)
  - Extended training: 400 epochs
  - Advanced augmentation strategy

#### Model Options
- `yolov8x`: YOLOv8 extra large variant
- `yolov9e`: YOLOv9 efficient variant
- `yolov10x`: YOLOv10 extra large variant
- `yolo11x`: YOLOv11 extra large variant

### Configuration Details

#### Small Object Detection Parameters
```python
PARAMETER_PRESETS = {
    'yolo11_small_focused': {
        'epochs': 400,         # Extended training
        'imgsz': 1920,        # Maximum resolution
        'batch': 2,           # Optimized for 2 GPUs
        'optimizer': 'AdamW',
        'lr0': 0.0001,        # Fine-tuned learning rate
        'augment': True,      # Enhanced augmentation
        # ... additional parameters
    }
}
```

## Project Structure

```
.
├── run.py                 # Main training script
├── data_clean/           # Clean dataset directory
│   ├── data_clean.yml    # Dataset configuration
│   ├── train_clean.txt   # Training image paths
│   ├── val_clean.txt     # Validation image paths
│   └── test_clean.txt    # Test image paths
├── models/               # YOLO model weights
└── results/             # Training results and logs
```

## Training Process

1. **Dataset Preparation**:
   - Clean dataset configuration in `data_clean/data_clean.yml`
   - Verification of all required files
   - Automatic path validation

2. **Model Training**:
   - Multi-GPU initialization
   - Parameter preset application
   - Progress tracking with Weights & Biases
   - Comprehensive metric logging

3. **Evaluation**:
   - Automated validation on test set
   - Misclassification analysis
   - Performance visualization
   - Metric comparison across models

## Results and Visualization

Training results are automatically logged and include:
- Training metrics and learning curves
- Validation performance
- Misclassification analysis
- Model comparison visualizations

Access detailed results through:
- Weights & Biases dashboard
- Generated CSV reports
- Comparison plots in `evaluation_results_clean/`

## Requirements

- Python 3.8+
- PyTorch 1.7+
- CUDA-capable GPUs (minimum 2)
- 32GB+ GPU memory (total)
- Ultralytics YOLO package
- Weights & Biases account

## Installation

```bash
git clone <repository-url>
cd <project-directory>
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Dr. Majid Memari
Utah Valley University
mmemari@uvu.edu
