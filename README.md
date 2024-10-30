# Multi-GPU Object Detection with YOLO Family Models

## Project Overview

This project demonstrates advanced object detection using multiple YOLO model versions (YOLOv8 to YOLOv11) with multi-GPU support. It serves as an educational template for implementing and comparing different YOLO architectures while leveraging parallel processing capabilities. The project showcases best practices in deep learning, including:

- Multi-GPU training optimization
- Early stopping mechanisms
- Experiment tracking
- Hyperparameter tuning
- Data augmentation strategies
- Model performance comparison

While the example use case focuses on defect detection in wind turbine blades using thermal imaging, the framework can be adapted for any object detection task across various domains such as autonomous driving, medical imaging, and surveillance.

## Key Features

- **Multi-Model Training Pipeline**: Sequential training of YOLOv8, YOLOv9, YOLOv10, and YOLOv11.
- **GPU Optimization**: 
  - Multi-GPU support with automatic workload distribution.
  - Automatic Mixed Precision (AMP) training for faster computation.
  - Efficient memory management to maximize GPU utilization.
- **Advanced Training Features**:
  - Smart early stopping with multiple criteria to prevent overfitting.
  - Version-specific hyperparameter optimization tailored to each YOLO variant.
  - Comprehensive logging and monitoring for detailed training insights.
  - Rich console output for real-time tracking of training progress.
- **Experiment Management**:
  - Weights & Biases integration for robust experiment tracking.
  - Automated performance tracking to monitor model improvements.
  - Model versioning and artifact management to maintain organized records.

## Example Use Case: Wind Turbine Blade Inspection

This implementation demonstrates practical application through wind turbine blade inspection:

1. **Data Collection**: 
   - UAV-captured thermal and RGB images.
   - Utilizes multiple drone platforms (DJI Matrice 300 RTK, DJI Mavic 3T).
   - Collects data under various environmental conditions to ensure diverse dataset representation.

2. **Pre-processing**: 
   - Image normalization and augmentation to enhance dataset quality.
   - Thermal image calibration to ensure accurate temperature representation.
   - Dataset optimization tailored for defect detection in wind turbine blades.

3. **Model Training**: 
   - Multi-GPU training with different YOLO versions to leverage computational power.
   - Specialized augmentation techniques for thermal imagery to improve model robustness.
   - Defect-specific optimization to enhance detection accuracy of subsurface faults.

4. **Evaluation**: 
   - Performance comparison across different YOLO models to identify the most effective architecture.
   - Defect detection accuracy analysis to measure model precision and recall.
   - Assessment of environmental condition impacts on model performance.

5. **Deployment**: 
   - Model export and inference optimization for real-time applications.
   - Integration with UAV systems for on-the-fly fault detection.
   - Considerations for deploying models in edge devices to facilitate remote inspections.

## Project Structure

```
.
├── run.py              # Main training pipeline
├── models/
│   ├── yolov8.py       # YOLOv8 implementation
│   ├── yolov9.py       # YOLOv9 implementation
│   ├── yolov10.py      # YOLOv10 implementation
│   └── yolov11.py      # YOLOv11 implementation
├── utils/
│   ├── augmentation.py # Data augmentation utilities
│   ├── early_stop.py   # Early stopping implementation
│   └── gpu_utils.py    # GPU management utilities
├── data/
│   ├── train/          # Training data
│   ├── val/            # Validation data
│   ├── test/           # Test data
│   └── config.yml      # Dataset configuration
├── environment.yml     # Conda environment setup
├── requirements.txt    # Pip dependencies
└── README.md           # Project documentation
```

## Technical Implementation

### YOLO Models Overview

The YOLO (You Only Look Once) family of models are state-of-the-art real-time object detection systems known for their speed and accuracy. This project leverages multiple versions of YOLO to compare their performance and suitability for specific tasks.

#### YOLOv8

- **Description**: The latest stable version from Ultralytics.
- **Features**:
  - Enhanced backbone and neck architecture.
  - Improved anchor-free detection mechanism.
  - Faster inference times with comparable accuracy.
- **Use Case**: Ideal for general object detection tasks requiring high speed.

#### YOLOv9

- **Description**: An efficient architecture with optimized feature extraction.
- **Features**:
  - Enhanced computational efficiency.
  - Better performance on small and densely packed objects.
  - Reduced model size without sacrificing accuracy.
- **Use Case**: Suitable for scenarios requiring high speed and accuracy on small object detection.

#### YOLOv10

- **Description**: An advanced version with improved accuracy and robustness.
- **Features**:
  - Superior handling of complex scenes with multiple objects.
  - Enhanced feature pyramid network for better multi-scale detection.
  - Improved loss functions for more precise localization.
- **Use Case**: Ideal for detailed and complex defect localization tasks.

#### YOLOv11

- **Description**: The latest experimental version incorporating cutting-edge features.
- **Features**:
  - Advanced augmentation techniques.
  - Improved loss functions for better training dynamics.
  - Experimental modules for future YOLO advancements.
- **Use Case**: Experimental setups that require the latest features and optimizations in object detection.

### Multi-GPU Setup

Utilizing multiple GPUs significantly accelerates the training process and allows handling larger models and datasets. The setup ensures efficient distribution of workloads across available GPUs.

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

Early stopping is crucial to prevent overfitting and to ensure that training halts when the model ceases to improve.

```python
class EarlyStoppingCallback:
    def __init__(self, 
                 patience=20,
                 min_epochs=50,
                 max_epochs=1000,
                 map_threshold=0.85,
                 loss_threshold=0.01,
                 smoothing_window=5):
        self.patience = patience
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.map_threshold = map_threshold
        self.loss_threshold = loss_threshold
        self.smoothing_window = smoothing_window
        
        self.val_losses = deque(maxlen=smoothing_window)
        self.map_scores = deque(maxlen=smoothing_window)
        self.best_map = 0
        self.best_epoch = 0
        self.stagnant_epochs = 0
        self.best_weights = None

    def check_stop_criteria(self, epoch, metrics, model_weights=None):
        current_map = metrics.get('mAP50-95', 0)
        current_loss = metrics.get('train/box_loss', float('inf'))
        
        self.map_scores.append(current_map)
        self.val_losses.append(current_loss)
        
        if current_map > self.best_map:
            self.best_map = current_map
            self.best_epoch = epoch
            self.stagnant_epochs = 0
            self.best_weights = model_weights
        else:
            self.stagnant_epochs += 1
        
        if epoch < self.min_epochs:
            return False, "Minimum epochs not reached"
            
        if epoch >= self.max_epochs:
            return True, "Maximum epochs reached"
            
        if current_map >= self.map_threshold:
            return True, f"mAP threshold {self.map_threshold} reached"
            
        if len(self.val_losses) == self.smoothing_window:
            loss_change = abs(np.mean(list(self.val_losses)[-self.smoothing_window:]) - 
                            np.mean(list(self.val_losses)[:-1]))
            if loss_change < self.loss_threshold:
                return True, f"Loss plateaued (change: {loss_change:.4f})"
                
        if self.stagnant_epochs >= self.patience:
            return True, f"No improvement for {self.patience} epochs"
            
        return False, "Training continuing"
```

### Version-Specific Augmentation

Different YOLO versions may benefit from tailored data augmentation strategies to enhance model performance.

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
    },
    'v10': {
        'mosaic': 1.0,
        'mixup': 0.25,
        'copy_paste': 0.5,
        'perspective': 0.7
    },
    'v11': {
        'mosaic': 1.0,
        'mixup': 0.30,
        'copy_paste': 0.6,
        'perspective': 0.8
    }
}
```

## Getting Started

### Prerequisites

- **Python**: 3.8+
- **CUDA-capable GPU(s)**: Recommended with 16GB+ GPU memory.
- **NVIDIA Drivers and CUDA Toolkit**: Ensure compatibility with PyTorch and your GPU.
- **Anaconda or Miniconda**: For environment management.

### Installation

#### Option 1: Using Conda (Recommended)

1. **Create Conda Environment from YAML**

    ```bash
    conda env create -f ultralytics.yml
    conda activate ultralytics
    ```

2. **Verify Installation**

    ```bash
    python -c "import torch; print(f'PyTorch {torch.__version__}')"
    python -c "import ultralytics; print(f'Ultralytics {ultralytics.__version__}')"
    ```

    The `ultralytics.yml` file contains:

    ```yaml
    name: ultralytics
    channels:
      - pytorch
      - nvidia
      - conda-forge
      - defaults
    dependencies:
      - python=3.8
      - pytorch>=2.0.0
      - torchvision
      - pytorch-cuda=11.8
      - ultralytics
      - wandb
      - rich
      - numpy
      - pandas
      - opencv
      - matplotlib
      - seaborn
      - pip
      - pip:
        - albumentations>=1.3.1
    ```


### Basic Usage

1. **Prepare Your Dataset**

    Ensure your dataset is organized in YOLO format with appropriate annotations. Update the `config.yml` file with the correct paths and class information.

    ```yaml
    # config.yml
    train: path/to/train
    val: path/to/val
    test: path/to/test
    nc: 5  # number of classes
    names: ['class1', 'class2', 'class3', 'class4', 'class5']
    ```

2. **Run Training**

    Execute the main training pipeline to train all YOLO models sequentially.

    ```bash
    python run.py --config config.yml --models v8,v9,v10,v11
    ```

### Advanced Usage

#### Custom Training Configuration

Customize training parameters to suit your specific needs.

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

#### Custom Early Stopping

Adjust early stopping criteria to control training duration and prevent overfitting.

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
   - Use appropriate batch sizes per GPU to maximize memory utilization.
   - Enable AMP training to reduce memory footprint.
   - Clear GPU cache between training runs using `torch.cuda.empty_cache()`.

2. **Training Speed**
   - Optimize the number of worker threads per GPU based on your system's capabilities.
   - Enable CUDNN benchmarking (`torch.backends.cudnn.benchmark = True`) for optimized CUDA operations.
   - Choose an appropriate image size that balances detection accuracy and computational efficiency.

3. **Model Performance**
   - Implement a robust augmentation strategy to enhance model generalization.
   - Use learning rate scheduling to stabilize training and improve convergence.
   - Continuously monitor validation metrics to track model improvements and detect overfitting.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

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

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics) for YOLO implementations.
- [NVIDIA](https://www.nvidia.com/) for GPU computing resources.
- [Weights & Biases](https://wandb.ai/) for experiment tracking.
- [DJI](https://www.dji.com/) for drone platforms.
- [ANSYS](https://www.ansys.com/) for fluid dynamics and stress analysis tools.

## Contact

For questions and feedback:

- Create an issue in the repository.
- Email: [mmemari@uvu.edu](mailto:mmemari@uvu.edu)
- **Author**: Dr. Majid Memari, Assistant Professor
- **Department**: Computer Science
- **Institution**: Utah Valley University
- **Project Link**: [https://github.com/memari-majid/Multi-GPU-Object-Detection](https://github.com/memari-majid/Multi-GPU-Object-Detection)

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

## References

### Deep Learning Frameworks
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

### YOLO Models
- [YOLOv8 Paper](https://arxiv.org/abs/2304.00501)
- [YOLOv9 Paper](https://arxiv.org/abs/2402.13616)
- [YOLOv10 Paper](https://arxiv.org/abs/2403.00000)  <!-- Placeholder: Replace with actual link -->
- [YOLOv11 Paper](https://arxiv.org/abs/2404.00000)  <!-- Placeholder: Replace with actual link -->
- [Original YOLO Paper](https://arxiv.org/abs/1506.02640)

### Tools and Libraries
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Albumentations Documentation](https://albumentations.ai/docs/)
- [Rich Documentation](https://rich.readthedocs.io/)

### Multi-GPU Training
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NVIDIA Multi-GPU Best Practices](https://developer.nvidia.com/blog/scaling-deep-learning-training-with-pytorch-lightning/)

### Thermal Imaging
- [Thermal Image Processing Guide](https://www.flir.com/discover/rd-science/thermal-imaging-cameras-for-research-development/)
- [UAV Thermal Imaging Best Practices](https://www.dji.com/downloads/doc/M300_RTK_Thermal_Imaging_Best_Practices.pdf)

## Changelog

### [1.0.0] - 2024-03-15
- Initial release.
- Multi-GPU training support.
- YOLOv8-11 implementation.
- Thermal image processing pipeline.

### [1.0.1] - 2024-03-20
- Added `environment.yml` for Conda setup.
- Improved GPU memory management.
- Enhanced documentation.

## Future Work

1. **Model Improvements**
   - Integration of newer YOLO versions as they become available.
   - Custom architecture modifications to better suit specific detection tasks.
   - Model distillation for faster inference without significant loss in accuracy.

2. **Training Optimizations**
   - Advanced learning rate scheduling techniques.
   - Further improvements in mixed precision training.
   - Implementation of more sophisticated memory optimization techniques.

3. **Deployment**
   - Optimization for edge devices to enable on-device inference.
   - Development of real-time processing pipelines for live data streams.
   - Exploration of cloud deployment options for scalable object detection services.

### Environment Troubleshooting

1. **CUDA Issues**
    ```bash
    # Check CUDA availability
    nvidia-smi
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
    
    # Check GPU memory
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    ```
    
2. **Common CUDA Problems**
    - **Mismatch between PyTorch and CUDA versions**: Ensure that the installed PyTorch version is compatible with your CUDA toolkit.
    - **Insufficient GPU memory**: Reduce batch size or use gradient accumulation.
    - **Driver version incompatibility**: Update NVIDIA drivers to match CUDA toolkit requirements.
    
    See [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html) for more details.