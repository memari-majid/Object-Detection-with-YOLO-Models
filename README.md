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
