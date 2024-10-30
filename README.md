# Multi-GPU Object Detection with YOLO Family Models

## Project Overview

This project demonstrates advanced object detection using multiple YOLO model versions (v8-v11) with multi-GPU support. It serves as an educational template for implementing and comparing different YOLO architectures while leveraging parallel processing capabilities. The project showcases best practices in deep learning, including:

- Multi-GPU training optimization
- Early stopping mechanisms
- Experiment tracking
- Hyperparameter tuning
- Data augmentation strategies
- Model performance comparison

While the example use case focuses on defect detection in wind turbine blades using thermal imaging, the framework can be adapted for any object detection task.

## Key Features

- **Multi-Model Training Pipeline**: Sequential training of YOLOv8, YOLOv9, YOLOv10, and YOLOv11
- **GPU Optimization**: 
  - Multi-GPU support with automatic workload distribution
  - Automatic Mixed Precision (AMP) training
  - Efficient memory management
- **Advanced Training Features**:
  - Smart early stopping with multiple criteria
  - Version-specific hyperparameter optimization
  - Comprehensive logging and monitoring
  - Rich console output for real-time tracking
- **Experiment Management**:
  - Weights & Biases integration
  - Automated performance tracking
  - Model versioning and artifact management

## Example Use Case: Wind Turbine Blade Inspection

This implementation demonstrates practical application through wind turbine blade inspection:

1. **Data Collection**: 
   - UAV-captured thermal and RGB images
   - Multiple drone platforms (DJI Matrice 300 RTK, DJI Mavic 3T)
   - Various environmental conditions

2. **Pre-processing**: 
   - Image normalization and augmentation
   - Thermal image calibration
   - Dataset optimization for defect detection

3. **Model Training**: 
   - Multi-GPU training with different YOLO versions
   - Specialized augmentation for thermal imagery
   - Defect-specific optimization

4. **Evaluation**: 
   - Performance comparison across models
   - Defect detection accuracy analysis
   - Environmental condition impact assessment

5. **Deployment**: 
   - Model export and inference optimization
   - Real-time processing capabilities
   - UAV integration considerations

## Project Structure

```
.
├── run.py              # Main training pipeline
├── models/
│   ├── yolov8.py      # YOLOv8 implementation
│   ├── yolov9.py      # YOLOv9 implementation
│   ├── yolov10.py     # YOLOv10 implementation
│   └── yolov11.py     # YOLOv11 implementation
├── utils/
│   ├── augmentation.py # Data augmentation utilities
│   ├── early_stop.py  # Early stopping implementation
│   └── gpu_utils.py   # GPU management utilities
└── data/
    ├── train/         # Training data
    ├── val/           # Validation data
    ├── test/          # Test data
    └── config.yml     # Dataset configuration
```

## Technical Implementation

### Multi-GPU Setup

```python
def setup_gpu_environment():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    devices = ','.join([str(x) for x in range(torch.cuda.device_count())])
    return devices

train_args = {
    'device': setup_gpu_environment(),
    'amp': True,  # Automatic Mixed Precision
    'workers': 8  # Number of worker threads per GPU
}
```

### Early Stopping Implementation

```python
class EarlyStoppingCallback:
    def __init__(self, patience=20, min_epochs=50, max_epochs=1000):
        self.patience = patience
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.best_metric = 0
        self.counter = 0
    
    def __call__(self, metric):
        if metric > self.best_metric:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
```

### Version-Specific Augmentation

```python
augmentation_config = {
    'v8': {
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.3,
        'perspective': 0.5
    },
    'v9': {
        'mosaic': 1.0,
        'mixup': 0.20,
        'copy_paste': 0.4,
        'perspective': 0.6
    }
    # ... configurations for v10 and v11
}
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU(s) with 8GB+ memory
- NVIDIA drivers and CUDA toolkit installed

### Installation

1. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Prepare Your Dataset**
```yaml
# config.yml
train: path/to/train
val: path/to/val
test: path/to/test
nc: 5  # number of classes
names: ['class1', 'class2', 'class3', 'class4', 'class5']
```

2. **Run Training**
```bash
python run.py --config config.yml --models v8,v9,v10,v11
```

## Advanced Usage

### Custom Training Configuration

```python
training_config = {
    'batch_size': 16,
    'image_size': 640,
    'epochs': 100,
    'optimizer': 'AdamW',
    'lr0': 0.01,
    'lrf': 0.001,
    'momentum': 0.937,
    'weight_decay': 0.0005
}
```

### Custom Early Stopping

```python
early_stopping = EarlyStoppingCallback(
    patience=20,
    min_epochs=50,
    max_epochs=1000,
    map_threshold=0.85
)
```

## Performance Optimization Tips

1. **GPU Memory Management**
   - Use appropriate batch sizes per GPU
   - Enable AMP training
   - Clear cache between training runs

2. **Training Speed**
   - Optimize number of workers per GPU
   - Enable CUDNN benchmarking
   - Use appropriate image size

3. **Model Performance**
   - Implement proper augmentation strategy
   - Use learning rate scheduling
   - Monitor validation metrics

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Ultralytics for YOLO implementations
- NVIDIA for GPU computing resources
- Weights & Biases for experiment tracking
- DJI for drone platforms
- ANSYS for fluid dynamics and stress analysis tools

## Contact

For questions and feedback:
- Create an issue in the repository
- Email: mmemari@uvu.edu
- Author: Dr. Majid Memari, Assistant Professor
- Department: Computer Science
- Institution: Utah Valley University
- Project Link: [https://github.com/memari-majid/Multi-GPU-Object-Detection](https://github.com/memari-majid/Multi-GPU-Object-Detection)

## Citation

If you use this work in your research, please cite:

```bibtex
@software{memari2024multigpu,
  author = {Memari, Majid},
  title = {Multi-GPU Object Detection with YOLO Family Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/memari-majid/Multi-GPU-Object-Detection}
}
```