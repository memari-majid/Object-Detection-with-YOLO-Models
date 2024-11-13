# Wind Turbine Blade Defect Detection with YOLO Models

## Table of Contents
- [Educational Overview](#educational-overview)
- [Project Overview](#project-overview)
- [Small Defect Detection Challenges](#small-defect-detection-challenges)
- [Training Pipeline](#training-pipeline)
- [Parameter Configuration](#parameter-configuration)
- [Model Validation](#model-validation)
- [Testing and Analysis](#testing-and-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Results](#results)
- [License](#license)
- [Author](#author)
- [Citation](#citation)

## Educational Overview

### Learning Objectives
1. Understanding small object detection challenges
2. Mastering YOLO model configurations
3. Learning advanced data augmentation techniques
4. Implementing multi-GPU training strategies
5. Analyzing and improving model performance

### Key Concepts

#### 1. Small Object Detection Fundamentals
- **Definition**: Objects occupying <1% of image area
- **Challenges**:
  - Limited feature information
  - Easy to confuse with noise
  - Requires high-resolution processing
  - Computationally intensive

#### 2. Resolution vs. Performance Trade-offs
```python
# Resolution considerations
{
    'standard_resolution': 640,    # Traditional YOLO input
    'medium_resolution': 1280,     # Better for small objects
    'high_resolution': 1920,       # Optimal for tiny defects
    'computational_cost': {
        640: 'Base memory usage',
        1280: '4x base memory',
        1920: '9x base memory'
    }
}
```

#### 3. Feature Extraction Optimization
- Multi-scale feature maps
- Feature pyramid networks
- Attention mechanisms
- Anchor optimization

## Project Overview
This project implements advanced object detection for wind turbine blade defects using YOLO models (v8-v11), with a specific focus on small defect detection. The implementation includes optimized configurations for high-resolution images and multi-GPU support.

## Small Defect Detection Challenges

### 1. Scale Challenges
```python
# Typical defect scales
{
    'large_defects': '>64x64 pixels',
    'medium_defects': '32x32 to 64x64 pixels',
    'small_defects': '16x16 to 32x32 pixels',
    'tiny_defects': '<16x16 pixels'
}
```

### 2. Technical Solutions
1. **Resolution Enhancement**
   - Input size increase
   - Feature map upsampling
   - Multi-scale training

2. **Feature Enhancement**
   ```python
   # Feature enhancement techniques
   {
       'feature_pyramid': True,    # Enhanced feature hierarchy
       'attention_modules': True,  # Focus on relevant features
       'deep_supervision': True,   # Multi-level supervision
       'adaptive_anchors': True    # Size-specific anchors
   }
   ```

3. **Loss Function Optimization**
   ```python
   # Loss components
   {
       'box_loss_weight': 10.0,    # Stronger localization
       'cls_loss_weight': 0.3,     # Reduced classification emphasis
       'dfl_loss_weight': 2.0,     # Better boundary definition
       'iou_loss_type': 'giou'     # Improved overlap calculation
   }
   ```

## Training Pipeline

### Advanced Training Strategy

#### 1. Progressive Resolution Training
```python
# Resolution progression
{
    'warmup_phase': {
        'epochs': 1-25,
        'resolution': 640
    },
    'intermediate_phase': {
        'epochs': 26-300,
        'resolution': 1280
    },
    'final_phase': {
        'epochs': 301-400,
        'resolution': 1920
    }
}
```

#### 2. Learning Rate Strategy
```python
# Learning rate configuration
{
    'initial_lr': 0.0001,
    'warmup_lr': {
        'start': 0.00001,
        'end': 0.0001,
        'epochs': 25
    },
    'main_training': {
        'scheduler': 'cosine',
        'final_lr': 0.000001,
        'epochs': 350
    },
    'fine_tuning': {
        'lr': 0.00001,
        'epochs': 25
    }
}
```

#### 3. Advanced Augmentation Pipeline
```python
# Comprehensive augmentation strategy
{
    'geometric_transforms': {
        'mosaic': {'prob': 1.0, 'disable_epoch': 375},
        'scale': {'range': [0.2, 1.8]},
        'rotate': {'max_degrees': 10.0},
        'shear': {'max_degrees': 3.0}
    },
    'appearance_transforms': {
        'hsv_h': {'range': [-0.015, 0.015]},
        'hsv_s': {'range': [-0.8, 0.8]},
        'hsv_v': {'range': [-0.5, 0.5]},
        'blur': {'types': ['gaussian', 'motion']}
    },
    'mixing_strategies': {
        'mixup': {'prob': 0.2},
        'copy_paste': {'prob': 0.4},
        'cutout': {'prob': 0.3}
    }
}
```

### Training Monitoring and Debugging

#### 1. Performance Metrics
```python
# Comprehensive metrics tracking
{
    'training_metrics': {
        'loss_components': ['box', 'cls', 'dfl'],
        'learning_rate': 'current_lr',
        'gradient_norm': 'grad_norm',
        'memory_usage': 'gpu_mem'
    },
    'validation_metrics': {
        'map': ['mAP50', 'mAP50-95'],
        'precision': 'precision_score',
        'recall': 'recall_score',
        'f1': 'f1_score'
    }
}
```

#### 2. Debug Features
```python
# Debugging tools
{
    'gradient_checking': True,
    'nan_detection': True,
    'memory_profiling': True,
    'batch_sampling': {
        'difficult_examples': True,
        'failed_detections': True
    }
}
```

## Model Validation

### Advanced Validation Strategy

#### 1. Multi-Scale Validation
```python
# Multi-scale testing
{
    'scales': [
        640,   # Base scale
        1280,  # Medium scale
        1920   # Full scale
    ],
    'flip_augmentation': True,
    'ensemble_predictions': True
}
```

#### 2. Confidence Calibration
```python
# Confidence thresholds
{
    'base_threshold': 0.1,
    'dynamic_adjustment': True,
    'size_specific_thresholds': {
        'large': 0.3,
        'medium': 0.2,
        'small': 0.1,
        'tiny': 0.05
    }
}
```

## Testing and Analysis

### Misclassification Analysis
```python
# Analysis Configuration
{
    'conf_threshold': 0.1,    # Detection confidence threshold
    'iou_threshold': 0.4,     # IoU threshold for matching
    'save_images': True,      # Save annotated images
    'analyze_errors': True    # Detailed error analysis
}
```

### Analysis Features
1. **Error Categories**
   - False positives
   - False negatives
   - Classification errors
   - Localization errors

2. **Visualization**
   - Annotated images with predictions
   - Ground truth comparisons
   - Error distribution plots
   - Performance metrics graphs

### Performance Metrics
```python
# Metrics Configuration
{
    'metrics': {
        'mAP50': True,        # Mean Average Precision at IoU=0.5
        'mAP50-95': True,     # Mean Average Precision at IoU=0.5:0.95
        'precision': True,     # Precision score
        'recall': True,       # Recall score
        'f1': True           # F1 score
    }
}
```

## Hardware Requirements

### GPU Configuration
```python
# Hardware Requirements
{
    'gpu_count': 2,           # Number of GPUs required
    'min_gpu_memory': '16GB', # Minimum GPU memory per device
    'cuda_version': '11.7+',  # CUDA version requirement
    'total_gpu_memory': '32GB+' # Total GPU memory recommended
}
```

## Results

### Training Metrics
- Learning rate progression
- Loss components tracking
- GPU utilization
- Memory usage patterns

### Validation Results
- Model performance metrics
- Confusion matrix
- Error analysis
- Performance comparisons

### Visualization Tools
1. **Weights & Biases Integration**
   - Real-time metric tracking
   - Experiment comparison
   - Resource monitoring
   - Result visualization

2. **Local Visualization**
   - CSV reports generation
   - Comparison plots
   - Error analysis graphs
   - Performance curves

## Installation
```bash
git clone <repository-url>
cd <project-directory>
pip install -r requirements.txt
```

## Usage

### Basic Usage
Train a specific model with optimized small object detection parameters:
```bash
python main.py --dataset clean --preset yolo11_small_focused --models yolo11x
```

### Available Parameters
- **Dataset Options**
  - `clean`: Filtered, high-quality dataset
  - `full`: Complete dataset

## Requirements
- Python 3.8+
- PyTorch 1.7+
- CUDA-capable GPUs (minimum 2)
- 32GB+ GPU memory (total)
- Ultralytics YOLO package
- Weights & Biases account

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
Majid Memari  
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
