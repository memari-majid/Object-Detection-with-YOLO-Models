# Non-Destructive Fault Detection in Small Wind Turbine Blades Using UAV Infrared and Thermal Imaging

## Project Overview

This project advances non-destructive fault detection in small wind turbine blades through the utilization of UAV-captured infrared and thermal images. By integrating advanced image processing and machine learning techniques, the project focuses on identifying subsurface defects that can compromise wind turbine blade performance and lifespan. These defects include delaminations, voids, and cracks, which can lead to significant structural weaknesses in the blades.

The UAV-mounted thermal imaging system enables maintenance teams to detect faults without requiring turbine downtime, thereby enhancing operational efficiency. A specialized, high-resolution thermal dataset was developed to accurately represent real operational conditions, collected under various environmental settings to allow detailed inspection and analysis of blade defects. Additionally, a comparative analysis of recent object detection models, including advanced YOLO versions, assesses the adaptability of these models to wind turbine blade fault detection.

## UAV and Thermal Imaging Equipment

The project employs the following UAVs for data collection:

- **DJI Matrice 300 RTK**
  - **Features**: High-precision RTK capabilities, multiple payload options, extended flight time.
  - **Purpose**: Captures high-resolution thermal and RGB images optimized for detailed blade inspections.

- **DJI Mavic 3T**
  - **Features**: Compact design, dual-camera system (thermal and RGB), high flight stability.
  - **Purpose**: Provides flexible imaging options suitable for various inspection scenarios.

These drones capture detailed images in both thermal and RGB modes, offering essential insights into the blade's structural integrity under diverse conditions.

## Fluid Dynamics and Stress Analysis

Complementing the UAV thermal imaging, the project incorporates fluid dynamics and stress analysis on Ryse Energy's E5 wind turbine blade. Utilizing ANSYS Fluent and static structural analysis tools, the project simulates airflow, analyzes stress distribution, and evaluates blade deformation under operational conditions. This analysis provides crucial data on how aerodynamic forces impact blade stress and deformation, contributing to a better understanding of potential failure points.

## Key Contributions

1. **New High-Resolution Thermal Dataset**
   - Developed a custom thermal dataset to capture specific small defects unique to wind turbine blades.
   - Optimized for defect detectability and analysis accuracy under various environmental conditions.

2. **Comparative Analysis of Object Detection Models**
   - Evaluated state-of-the-art object detection models, including YOLOv8, YOLOv9, YOLOv10, and YOLOv11.
   - Assessed the suitability of these models for detailed defect localization within thermal images.

3. **Drone-Compatible Camera Testing**
   - Conducted thermal camera testing in both handheld and UAV setups.
   - Confirmed image quality suitable for comprehensive blade inspections via UAV deployment.

4. **CFD and Stress Analysis**
   - Performed Fluid-Structure Interaction (FSI) simulations and stress analysis.
   - Provided insights into aerodynamic effects on blade stability and contributed to predictions on blade fatigue life.

## Features

- **Sequential Training of Multiple YOLO Versions**: Supports YOLOv8, YOLOv9, YOLOv10, and YOLOv11 for comprehensive object detection capabilities.
- **Smart Early Stopping**: Implements multiple criteria for early stopping to optimize training efficiency:
  - mAP threshold (0.85)
  - Loss plateau detection
  - Patience-based stopping (20 epochs)
  - Minimum (50 epochs) and maximum (1000 epochs) epoch constraints
- **Multi-GPU Support**: Leverages multiple GPUs for accelerated training.
- **Weights & Biases Integration**: Facilitates experiment tracking and performance monitoring.
- **Rich Console Output**: Provides real-time training progress and metrics visualization.
- **Version-Specific Hyperparameter Optimization**: Tailors augmentation and training parameters based on YOLO version.

## Project Structure

```
.
├── run.py              # Main training pipeline
├── yolov8.py           # YOLOv8 specific training
├── yolov9.py           # YOLOv9 specific training
├── yolov10.py          # YOLOv10 specific training
├── yolov11.py          # YOLOv11 specific training
└── data/
    ├── obj_train_data_RGB/  # Training images
    ├── data.yml            # Dataset configuration
    ├── train.txt           # Training set paths
    ├── val.txt             # Validation set paths
    └── test.txt            # Test set paths
```

## Models Overview

### Supported YOLO Versions

1. **YOLOv8x**
   - **Description**: Latest stable version from Ultralytics.
   - **Features**: Enhanced backbone and neck architecture, improved anchor-free detection.
   - **Use Case**: Best for general object detection tasks.

2. **YOLOv9e**
   - **Description**: Efficient architecture with improved feature extraction.
   - **Features**: Enhanced computational efficiency, better performance on small objects.
   - **Use Case**: Suitable for scenarios requiring high speed and accuracy on small defect detection.

3. **YOLOv10x**
   - **Description**: Advanced architecture with improved accuracy.
   - **Features**: Better handling of complex scenes, enhanced feature pyramid network.
   - **Use Case**: Ideal for detailed and complex defect localization tasks.

4. **YOLOv11x**
   - **Description**: Latest experimental version.
   - **Features**: Advanced augmentation techniques, improved loss functions.
   - **Use Case**: Experimental setups requiring cutting-edge features and optimizations.

## Training Pipeline

### Early Stopping Mechanism

The training pipeline incorporates a sophisticated early stopping system to optimize training efficiency and prevent overfitting. The parameters are as follows:

```python
early_stopping = EarlyStoppingCallback(
    patience=20,          # Epochs to wait before early stopping
    min_epochs=50,        # Minimum epochs before allowing early stop
    max_epochs=1000,      # Maximum epochs to train
    map_threshold=0.85,   # Stop if mAP reaches this value
    loss_threshold=0.01,  # Minimum change in loss to continue
    smoothing_window=5    # Window size for smoothing metrics
)
```

**Stopping Criteria:**
- **mAP Threshold**: Training stops if the mean Average Precision (mAP) reaches or exceeds 0.85.
- **Loss Plateau Detection**: Stops if the change in loss over a smoothing window of 5 epochs is less than 0.01.
- **Patience-Based Stopping**: Stops if there is no improvement in mAP for 20 consecutive epochs.
- **Epoch Constraints**: Ensures a minimum of 50 epochs and allows training to run up to 1000 epochs if necessary.

### Version-Specific Augmentation

Each YOLO version utilizes optimized augmentation parameters to enhance model performance:

```python
version_specific = {
    '8': {'mixup': 0.15, 'copy_paste': 0.3, 'perspective': 0.5},
    '9': {'mixup': 0.20, 'copy_paste': 0.4, 'perspective': 0.6},
    '10': {'mixup': 0.25, 'copy_paste': 0.5, 'perspective': 0.7},
    '11': {'mixup': 0.30, 'copy_paste': 0.6, 'perspective': 0.8}
}
```

### Common Training Parameters

All models share the following base training configurations:

```python
common_args = {
    'mosaic': 1.0,
    'degrees': 10.0,
    'translate': 0.2,
    'scale': 0.9,
    'shear': 2.0,
    'flipud': 0.3,
    'fliplr': 0.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'warmup_epochs': 3,
    'close_mosaic': 10,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'label_smoothing': 0.1,
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.2,
}
```

### Training Configuration

The training setup integrates early stopping and version-specific augmentations:

```python
def train_and_evaluate(model_name, data_yaml_path, epochs, batch_size=16, image_size=640):
    try:
        # Initialize W&B run
        setup_wandb_logging(model_name, epochs, batch_size, image_size)
        
        # Load model
        model = YOLO(model_name)
        
        # Initialize early stopping
        early_stopping = EarlyStoppingCallback(
            patience=20,
            min_epochs=50,
            max_epochs=epochs,
            map_threshold=0.85,
            loss_threshold=0.01
        )
        
        # Training configuration
        train_args = {
            'data': data_yaml_path,
            'epochs': epochs,
            'imgsz': image_size,
            'batch': batch_size,
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
            'callbacks': [early_stopping]  # Add the early stopping callback
        }
        
        # Train model
        results = model.train(**train_args)
        
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        wandb.finish()
```

### Training Process

The `run.py` script orchestrates the training of multiple YOLO versions sequentially:

```python
def main():
    # Configuration
    base_dir = '/home/majid/PycharmProjects/E5/data/obj_train_data_RGB'
    data_yaml_path = '/home/majid/PycharmProjects/E5/data/data.yml'
    
    models = ['yolov8x.pt', 'yolov9e.pt', 'yolov10x.pt', 'yolo11x.pt']
    
    # CUDA settings
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    # Results tracking
    results_summary = []
    
    # Train models sequentially
    for model_name in models:
        console.print(f"\n[bold blue]Starting training for {model_name}")
        
        start_time = time.time()
        success, error = train_and_evaluate(
            model_name=model_name,
            data_yaml_path=data_yaml_path
        )
        
        duration = time.time() - start_time
        
        # Store results
        results_summary.append({
            'model': model_name,
            'success': success,
            'duration': duration,
            'error': error if not success else None
        })
    
    # Print summary
    console.print("\n[bold yellow]Training Summary:")
    for result in results_summary:
        status = "[green]Success" if result['success'] else f"[red]Failed: {result['error']}"
        console.print(f"Model: {result['model']}")
        console.print(f"Status: {status}")
        console.print(f"Duration: {result['duration']/3600:.2f} hours\n")
```

**Key Steps:**

1. **Initialization**: Clears GPU cache and benchmarks CUDA operations for optimized performance.
2. **Sequential Training**: Iterates through each YOLO version, initiating training with the specified configurations.
3. **Result Tracking**: Records the success status, duration, and any errors encountered during training.
4. **Summary Reporting**: Provides a comprehensive summary of the training outcomes for each model.

## GPU Utilization

The training pipeline is optimized for multi-GPU environments:

- **Devices**: Utilizes CUDA devices 0 and 1.
- **Automatic Mixed Precision (AMP)**: Enhances training speed and reduces memory usage.
- **CUDNN Benchmark Mode**: Optimizes CUDA operations for the specific hardware.

GPU memory management is handled automatically to ensure efficient resource utilization:

```python
torch.cuda.empty_cache()  # Clear GPU memory
torch.backends.cudnn.benchmark = True  # Optimize CUDA operations
```

## Monitoring and Logging

### Real-time Monitoring

- **Rich Console Output**: Provides dynamic and visually appealing progress bars and metric displays.
- **Training Metrics**: Displays current epoch, loss, mAP, and other relevant metrics in real-time.
- **Early Stopping Status**: Indicates if and when early stopping criteria are met.

### Weights & Biases Integration

- **Experiment Tracking**: Logs all training runs, including hyperparameters, metrics, and model checkpoints.
- **Performance Metrics**: Visualizes mAP, loss curves, and other key indicators.
- **Model Artifacts**: Saves and versions trained models for future reference and deployment.

### Training Summary

After training completion, a summary report is generated:

```
Training Summary:
Model: yolov8x.pt
Status: Success
Duration: 10.50 hours

Model: yolov9e.pt
Status: Failed: [Error Message]
Duration: 8.75 hours

...
```

## Requirements

- **Python**: 3.8+
- **PyTorch**: >= 1.7.0
- **Ultralytics YOLO**
- **Weights & Biases**
- **Rich** (for enhanced console output)
- **NumPy**
- **CUDA-capable GPU(s)**: Recommended with 16GB+ GPU memory

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/memari-majid/Fault-Detection-Using-UAV-Thermal-Imaging.git
    cd Fault-Detection-Using-UAV-Thermal-Imaging
    ```

2. **Create and Activate Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or
    .\venv\Scripts\activate     # Windows
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    *Alternatively, install individually:*
    ```bash
    pip install ultralytics wandb rich torch torchvision numpy
    ```

4. **Configure Weights & Biases**
    ```bash
    wandb login
    ```

## Configuration

### Dataset Preparation

Ensure your dataset is organized in YOLO format with appropriate annotations. Update the `data.yml` file with the correct paths:

```yaml
train: /path/to/train/images
val: /path/to/val/images
test: /path/to/test/images
nc: <number_of_classes>
names: ['class1', 'class2', ...]
```

### Early Stopping Parameters

Customize early stopping criteria as needed:

```python
early_stopping = EarlyStoppingCallback(
    patience=20,          # Epochs to wait before early stopping
    min_epochs=50,        # Minimum epochs before allowing early stop
    max_epochs=1000,      # Maximum epochs to train
    map_threshold=0.85,   # Stop if mAP reaches this value
    loss_threshold=0.01,  # Minimum change in loss to continue
    smoothing_window=5    # Window size for smoothing metrics
)
```

### Training Parameters

Adjust training parameters in the `train_args` dictionary as necessary:

```python
train_args = {
    'imgsz': 640,            # Input image size
    'batch': 16,             # Batch size
    'device': '0,1',         # Multi-GPU training
    'amp': True,             # Automatic Mixed Precision
    'workers': 8,            # Number of worker threads
    'optimizer': 'AdamW',    # Optimizer choice
    'lr0': 0.01,             # Initial learning rate
    'lrf': 0.001,            # Final learning rate
    'momentum': 0.937,       # Optimizer momentum
    'weight_decay': 0.0005,  # Weight decay coefficient
    'save': True,            # Save checkpoints
    'plots': True,           # Generate training plots
    'callbacks': [early_stopping]  # Early stopping callback
}
```

### Version-Specific Augmentation

Set augmentation parameters based on the YOLO version:

```python
version_specific = {
    '8': {'mixup': 0.15, 'copy_paste': 0.3, 'perspective': 0.5},
    '9': {'mixup': 0.20, 'copy_paste': 0.4, 'perspective': 0.6},
    '10': {'mixup': 0.25, 'copy_paste': 0.5, 'perspective': 0.7},
    '11': {'mixup': 0.30, 'copy_paste': 0.6, 'perspective': 0.8}
}
```

## Usage

1. **Prepare Your Dataset**
   - Organize images and annotations in YOLO format.
   - Update the `data.yml` file with correct paths and class information.

2. **Initialize Weights & Biases**
    ```bash
    wandb login
    ```

3. **Run the Training Pipeline**
    ```bash
    python run.py
    ```

    This will sequentially train YOLOv8, YOLOv9, YOLOv10, and YOLOv11 models with the specified configurations and early stopping mechanisms.

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. **Fork the Repository**
2. **Create Your Feature Branch**
    ```bash
    git checkout -b feature/AmazingFeature
    ```
3. **Commit Your Changes**
    ```bash
    git commit -m 'Add some AmazingFeature'
    ```
4. **Push to the Branch**
    ```bash
    git push origin feature/AmazingFeature
    ```
5. **Open a Pull Request**

## License

![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Distributed under the MIT License. See `LICENSE` for more information.

### MIT License

The MIT License (MIT) 

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be  
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics) for the YOLO implementation
- [Joseph Redmon](https://pjreddie.com/) for creating the original YOLO algorithms
- [Darknet](https://github.com/pjreddie/darknet) for the original YOLO framework
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [ANSYS](https://www.ansys.com/) for providing fluid dynamics and stress analysis tools

## Contact

Majid Memari - [mmemari@uvu.edu](mailto:mmemari@uvu.edu)  
Project Link: [https://github.com/memari-majid/Fault-Detection-Using-UAV-Thermal-Imaging](https://github.com/memari-majid/Fault-Detection-Using-UAV-Thermal-Imaging)

## Environment Setup

### Using Conda Environment

1. **Create Environment from YAML**
   ```bash
   conda env create -f ultralytics.yml
   ```

2. **Activate Environment**
   ```bash
   conda activate ultralytics
   ```

3. **Verify Installation**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}')"
   python -c "import ultralytics; print(f'Ultralytics {ultralytics.__version__}')"
   ```

4. **Update Environment** (if needed)
   ```bash
   conda env update -f ultralytics.yml --prune
   ```

### Manual Environment Setup (Alternative)

If you prefer to set up the environment manually:

1. **Create a New Conda Environment**
   ```bash
   conda create -n    conda activate ultralytics python=3.8
   conda activate    conda activate ultralytics 

   ```

2. **Install Core Dependencies**
   ```bash
   pip install torch torchvision
   pip install ultralytics
   pip install wandb
   pip install rich
   ```

3. **Install Additional Dependencies**
   ```bash
   pip install numpy pandas opencv-python
   pip install matplotlib seaborn
   ```

### Environment Information

- Python version: 3.8.19
- Key packages:
  - PyTorch: 2.3.1
  - Ultralytics: 8.3.24
  - CUDA support: Yes (CUDA 12.1)
  - Weights & Biases: 0.18.5

### Troubleshooting

If you encounter CUDA-related issues:
1. Verify CUDA installation:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Check CUDA compatibility:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
   ```

3. Common solutions:
   - Ensure NVIDIA drivers are up to date
   - Match PyTorch version with CUDA version
   - Clear GPU memory: `torch.cuda.empty_cache()`