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
python main.py --dataset clean --preset yolo11_small_focused --models yolo11x
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
# Training Parameters for YOLOv11 Small Object Detection
PARAMETER_PRESETS = {
    'yolo11_small_focused': {
        # Basic Training Parameters
        'epochs': 400,         # Extended training duration
        'imgsz': 1920,        # Maximum resolution for small object detection
        'batch': 2,           # Optimized for 2 GPUs
        'device': '0,1',      # Multi-GPU utilization
        
        # Optimizer Configuration
        'optimizer': 'AdamW',  # Advanced optimizer
        'lr0': 0.0001,        # Very low learning rate for fine details
        'lrf': 0.000001,      # Final learning rate
        'momentum': 0.937,    # Optimizer momentum
        'weight_decay': 0.001,# Weight decay for regularization
        
        # Augmentation Strategy
        'mosaic': 1.0,        # Maximum mosaic augmentation
        'scale': 0.2,         # Aggressive scaling (0.2-1.8)
        'flipud': 0.7,        # Vertical flip probability
        'fliplr': 0.7,        # Horizontal flip probability
        'augment': True,      # Enable augmentation
        'degrees': 10.0,      # Rotation range
        'translate': 0.3,     # Translation range
        'perspective': 0.001, # Perspective transformation
        'shear': 3.0,        # Shearing range
        
        # Training Stability
        'cos_lr': True,       # Cosine learning rate schedule
        'patience': 100,      # High patience for convergence
        'workers': 10,         # Number of worker threads
        'label_smoothing': 0.15, # Label smoothing factor
        'overlap_mask': True, # Mask overlap handling
        'warmup_epochs': 25,  # Extended warmup period
        
        # Loss Functions
        'box': 10.0,         # Box loss weight
        'cls': 0.3,          # Classification loss weight
        'dfl': 2.0,          # DFL loss weight
        
        # Advanced Augmentation
        'close_mosaic': 15,  # Disable mosaic in final epochs
        'mixup': 0.2,        # Mixup augmentation
        'copy_paste': 0.4,   # Copy-paste augmentation
        'hsv_h': 0.015,      # HSV hue augmentation
        'hsv_s': 0.8,        # HSV saturation augmentation
        'hsv_v': 0.5,        # HSV value augmentation
    }
}

# Validation Parameters
VAL_PARAMS = {
    'yolo11_small_focused': {
        'imgsz': 1920,       # Match training resolution
        'batch': 2,          # Validation batch size
        'device': '0,1',     # Use both GPUs
        'conf': 0.1,         # Low confidence threshold for small objects
        'iou': 0.4          # IoU threshold
    }
}
```

### Hardware Requirements

#### GPU Configuration
- Number of GPUs: 2
- Minimum GPU Memory: 16GB per GPU
- Recommended GPU: NVIDIA RTX 3090 or better
- CUDA Version: 11.7 or higher

#### System Requirements
- RAM: 64GB minimum
- Storage: 500GB SSD recommended
- CPU: 8+ cores recommended
- OS: Ubuntu 20.04 or higher

### Dataset Specifications

#### Clean Dataset Structure
```yaml
path: data_clean/obj_train_data_RGB
train: data_clean/train_clean.txt
val: data_clean/val_clean.txt
test: data_clean/test_clean.txt
nc: 1  # Number of classes
names: ['defect']
```

#### Image Specifications
- Resolution: Up to 1920x1920
- Format: JPG/JPEG
- Color Space: RGB
- Annotation Format: YOLO txt format

### Training Process Details

#### Initialization
- Multi-GPU synchronization
- Automatic batch size adjustment
- Memory cache clearing
- Gradient synchronization

#### Training Phases
1. **Warmup Phase** (25 epochs):
   - Gradually increasing learning rate
   - Full augmentation enabled
   - All GPUs synchronized

2. **Main Training Phase** (350 epochs):
   - Cosine learning rate scheduling
   - Advanced augmentation pipeline
   - Regular validation checks

3. **Fine-tuning Phase** (25 epochs):
   - Reduced learning rate
   - Disabled mosaic augmentation
   - Final model optimization

#### Monitoring and Logging
- Real-time metric tracking
- GPU memory usage monitoring
- Training progress visualization
- Automatic checkpoint saving

### Performance Metrics

#### Training Metrics
- Loss components (box, classification, DFL)
- Learning rate progression
- GPU utilization
- Memory usage

#### Validation Metrics
- mAP50-95
- Precision
- Recall
- F1-Score
- Confusion matrix

## Project Structure

```
.
├── main.py                # Main training script (previously run.py)
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

2. **Model Training** (via main.py):
   - Multi-GPU initialization
   - Parameter preset application
   - Progress tracking with Weights & Biases
   - Comprehensive metric logging

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

MIT License

Copyright (c) 2024 Majid Memari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

Dr. Majid Memari
Utah Valley University
mmemari@uvu.edu

## Citation

If you use this code in your research, please cite:

```bibtex
@software{memari2024wtb,
    author = {Memari, Majid},
    title = {Wind Turbine Blade Defect Detection with YOLO Models},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/memari-majid/WTB_Defect_Detection}},
    institution = {Utah Valley University}
}
```
