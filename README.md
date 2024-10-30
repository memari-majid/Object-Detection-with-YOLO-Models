# Multi-GPU Object Detection with YOLO Family Models

## Project Overview

This project demonstrates advanced object detection using multiple YOLO model versions (YOLOv8 to YOLOv11) with multi-GPU support. It serves as an educational template for implementing and comparing different YOLO architectures while leveraging parallel processing capabilities.

### Research Context and Significance

The project focuses on developing non-destructive fault detection methods for wind turbine blades (WTBs) using thermal images captured by Unmanned Aerial Vehicles (UAVs). Wind energy is a critical renewable source, but turbine blades are susceptible to damage that can hinder performance. While thermal imaging is effective for identifying subsurface defects, existing methods are limited due to generic datasets and lack of high-fidelity environmental conditions in testing.

### Project Goals and Methodology

This research addresses these limitations through:
- Creation of a custom, high-resolution thermal dataset specifically for WTBs
- Implementation of state-of-the-art machine learning and image processing techniques
- Advanced object detection model comparison using the YOLO family
- Integration of UAV-mounted cameras (DJI Matrice 300 RTK and DJI Mavic 3T)
- Comprehensive environmental simulation and testing

### Technical Analysis

The methodology incorporates:
- **Computational Analysis**: 
  - Computational Fluid Dynamics (CFD)
  - Finite Element Analysis (FEA) on WTBs
  - Stress distribution studies
  - Deformation analysis under varying wind speeds
- **Practical Applications**:
  - Material fatigue assessment
  - Structural integrity evaluation
  - Early-stage fault detection
  - Maintenance optimization

### Expected Impact

The project aims to:
- Improve wind turbine reliability
- Optimize maintenance procedures
- Extend blade lifespan
- Reduce operational costs
- Enhance wind energy infrastructure sustainability

## Table of Contents

1. [Key Features](#key-features)
2. [Getting Started](#getting-started)
3. [Technical Implementation](#technical-implementation)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Performance Optimization](#performance-optimization)
6. [Example Use Case](#example-use-case-wind-turbine-blade-inspection)
7. [Contributing](#contributing)
8. [Documentation and Resources](#documentation-and-resources)
9. [Project Information](#project-information)

## Key Features

- **Multi-Model Training Pipeline**: Sequential training of YOLOv8, YOLOv9, YOLOv10, and YOLOv11.
- **GPU Optimization**: 
  - Multi-GPU support with automatic workload distribution.
  - Automatic Mixed Precision (AMP) training for faster computation.
  - Efficient memory management to maximize GPU utilization.
- **Advanced Training Features**:
  - Smart early stopping with multiple criteria to prevent overfitting.
  - Version-specific hyperparameter optimization.
  - Comprehensive logging and monitoring.
- **Experiment Management**:
  - Weights & Biases integration for experiment tracking.
  - Automated performance tracking.
  - Model versioning and artifact management.

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU(s) with 16GB+ memory
- NVIDIA Drivers and CUDA Toolkit
- Anaconda or Miniconda

### Installation

```bash
conda env create -f ultralytics.yml
conda activate ultralytics
```

### Basic Usage

1. **Prepare Dataset**
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

## Technical Implementation

### Project Structure
```
.
├── run.py              # Main training pipeline
├── models/            
├── utils/             
├── data/              
├── environment.yml    
├── requirements.txt   
└── README.md         
```

### YOLO Models Overview

#### YOLOv8
- **Description**: The latest stable version from Ultralytics.
- **Features**:
  - Enhanced backbone and neck architecture
  - Improved anchor-free detection mechanism
  - Faster inference times with comparable accuracy
- **Use Case**: Ideal for general object detection tasks requiring high speed

#### YOLOv9
- **Description**: An efficient architecture with optimized feature extraction.
- **Features**:
  - Enhanced computational efficiency
  - Better performance on small and densely packed objects
  - Reduced model size without sacrificing accuracy
- **Use Case**: Suitable for scenarios requiring high speed and accuracy on small object detection

#### YOLOv10
- **Description**: An advanced version with improved accuracy and robustness.
- **Features**:
  - Superior handling of complex scenes with multiple objects
  - Enhanced feature pyramid network for better multi-scale detection
  - Improved loss functions for more precise localization
- **Use Case**: Ideal for detailed and complex defect localization tasks

#### YOLOv11
- **Description**: The latest experimental version incorporating cutting-edge features.
- **Features**:
  - Advanced augmentation techniques
  - Improved loss functions for better training dynamics
  - Experimental modules for future YOLO advancements
- **Use Case**: Experimental setups that require the latest features and optimizations in object detection

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

### Evaluation Metrics

The project uses comprehensive metrics for model evaluation:

- **mAP50-95**: Mean Average Precision across IoU thresholds
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all relevant instances
- **F1-Score**: Harmonic mean of precision and recall
- **Box Loss**: Loss value for bounding box predictions
- **Classification Loss**: Loss value for class predictions

### Early Stopping Implementation

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
```

## Example Use Case: Wind Turbine Blade Inspection

### Data Collection
- **UAV Equipment**:
  - DJI Matrice 300 RTK with H20T camera
  - DJI Mavic 3T with thermal imaging capabilities
  - Custom-mounted FLIR thermal cameras
- **Imaging Protocols**:
  - Multiple angles (0°, 45°, 90°) for comprehensive coverage
  - Flight patterns optimized for blade inspection
  - Both daytime and nighttime thermal imaging
- **Environmental Monitoring**:
  - Weather conditions (temperature, humidity, wind speed)
  - Time of day tracking for thermal variations
  - Seasonal data collection for diverse conditions

### Pre-processing
- **Image Normalization**:
  - Thermal calibration using reference temperatures
  - Histogram equalization for enhanced contrast
  - Resolution standardization across different cameras
- **Data Augmentation**:
  - Temperature variation simulation
  - Defect intensity augmentation
  - Environmental condition simulation
- **Quality Control**:
  - Automated blur detection and removal
  - Thermal noise reduction
  - Registration of thermal and RGB image pairs

### Model Training
- **Multi-GPU Strategy**:
  - Distributed training across available GPUs
  - Batch size optimization per GPU
  - Gradient synchronization strategy
- **Thermal-Specific Features**:
  - Custom loss functions for thermal gradients
  - Temperature-aware detection thresholds
  - Thermal pattern recognition optimization
- **Defect Categories**:
  - Delamination detection
  - Crack identification
  - Moisture ingress mapping
  - Internal structure anomalies

### Deployment
- **Model Optimization**:
  - TensorRT conversion for faster inference
  - Model quantization for edge devices
  - Batch processing optimization
- **UAV Integration**:
  - Real-time inference on drone systems
  - Automated flight path adjustment
  - Dynamic focus on detected anomalies
- **Monitoring System**:
  - Web-based dashboard for results visualization
  - Automated report generation
  - Historical trend analysis
- **Edge Computing**:
  - On-device processing capabilities
  - Low-latency detection requirements
  - Power consumption optimization

### Performance Metrics
- **Detection Accuracy**:
  - mAP scores for different defect types
  - False positive/negative analysis
  - Detection confidence thresholds
- **Processing Speed**:
  - Inference time per image
  - Real-time processing capabilities
  - GPU vs. CPU performance comparison
- **Resource Utilization**:
  - Memory usage optimization
  - Battery life considerations
  - Storage requirements for different deployment scenarios

## Model Evaluation on Unlabeled Data

### Real-Time Performance Assessment

#### Confidence Score Analysis
- **Threshold Optimization**:
  - Dynamic confidence threshold adjustment based on lighting conditions
  - Time-of-day specific threshold calibration
  - Environmental condition-based threshold adaptation

- **Statistical Monitoring**:
  ```python
  def analyze_confidence_distribution(predictions, window_size=1000):
      confidence_scores = [pred.conf for pred in predictions]
      mean_conf = np.mean(confidence_scores)
      std_conf = np.std(confidence_scores)
      return {
          'mean_confidence': mean_conf,
          'std_confidence': std_conf,
          'confidence_histogram': np.histogram(confidence_scores, bins=10)
      }
  ```

#### Domain Shift Detection
- **Feature Distribution Monitoring**:
  - Track changes in feature space distributions
  - Alert on significant deviations from training distribution
  - Adaptive calibration for new environmental conditions

- **Implementation Example**:
  ```python
  def monitor_domain_shift(current_features, baseline_features):
      # Calculate distribution divergence
      kl_div = compute_kl_divergence(current_features, baseline_features)
      
      # Alert if shift detected
      if kl_div > SHIFT_THRESHOLD:
          trigger_recalibration()
  ```

### Deployment Validation Strategy

#### Cross-Validation with Expert Review
1. **Initial Deployment Phase**:
   - Parallel human expert validation
   - Confusion matrix building for unlabeled data
   - Refinement of detection thresholds

2. **Continuous Monitoring**:
   ```python
   def track_detection_stability(detections, time_window=3600):  # 1 hour
       return {
           'detection_count': len(detections),
           'mean_confidence': np.mean([d.conf for d in detections]),
           'detection_variance': np.var([d.bbox_area for d in detections]),
           'temporal_consistency': assess_temporal_consistency(detections)
       }
   ```

#### Performance Metrics for Unlabeled Data
- **Stability Metrics**:
  - Detection consistency across frames
  - Confidence score stability
  - Temporal coherence of detections

- **Implementation**:
  ```python
  class UnlabeledPerformanceTracker:
      def __init__(self):
          self.detection_history = []
          self.confidence_threshold = 0.5
          
      def update_metrics(self, new_detections):
          # Track detection stability
          stability_score = self.compute_stability(new_detections)
          
          # Update threshold if needed
          self.adaptive_threshold_update(stability_score)
          
          # Log performance metrics
          self.log_metrics({
              'stability_score': stability_score,
              'detection_count': len(new_detections),
              'threshold': self.confidence_threshold
          })
  ```

### Automated Quality Control

#### Real-Time Validation Checks
1. **Physical Constraints**:
   - Size consistency verification
   - Location plausibility checks
   - Temporal consistency validation

2. **Implementation Example**:
   ```python
   def validate_detection(detection, physical_constraints):
       # Check if detection meets physical constraints
       size_valid = validate_size(detection, physical_constraints['size_range'])
       location_valid = validate_location(detection, physical_constraints['valid_zones'])
       
       return {
           'is_valid': size_valid and location_valid,
           'confidence': detection.conf,
           'validation_details': {
               'size_check': size_valid,
               'location_check': location_valid
           }
       }
   ```

#### Automated Calibration System
- **Dynamic Threshold Adjustment**:
  ```python
  class AdaptiveThresholdSystem:
      def __init__(self, initial_threshold=0.5):
          self.threshold = initial_threshold
          self.performance_history = []
          
      def update_threshold(self, recent_performance):
          # Adjust threshold based on performance metrics
          if self.needs_adjustment(recent_performance):
              new_threshold = self.calculate_optimal_threshold(
                  self.performance_history[-100:])
              self.threshold = new_threshold
              
      def needs_adjustment(self, performance):
          # Decision logic for threshold adjustment
          return (performance['false_positive_rate'] > 0.1 or 
                 performance['detection_stability'] < 0.8)
  ```

### Integration with Existing Systems

#### Data Pipeline Integration
```python
class DefectDetectionPipeline:
    def __init__(self, model, performance_tracker, validator):
        self.model = model
        self.performance_tracker = performance_tracker
        self.validator = validator
        
    def process_frame(self, frame):
        # Run detection
        detections = self.model(frame)
        
        # Validate detections
        validated_detections = [
            det for det in detections 
            if self.validator.validate_detection(det)['is_valid']
        ]
        
        # Update performance metrics
        self.performance_tracker.update_metrics(validated_detections)
        
        return validated_detections
```

#### Reporting and Monitoring
- **Real-Time Dashboard**:
  - Detection confidence trends
  - False positive estimation
  - System health metrics
  - Performance degradation alerts

- **Implementation**:
  ```python
  class PerformanceMonitor:
      def __init__(self):
          self.metrics_history = defaultdict(list)
          
      def log_metrics(self, metrics):
          for key, value in metrics.items():
              self.metrics_history[key].append(value)
              
          if self.detect_anomaly(metrics):
              self.trigger_alert()
              
      def generate_report(self):
          return {
              'detection_rate': np.mean(self.metrics_history['detection_count']),
              'confidence_trend': self.analyze_trend('confidence'),
              'stability_metrics': self.calculate_stability_metrics(),
              'system_health': self.assess_system_health()
          }
  ```

## Contributing

1. Fork the Repository
2. Create your Feature Branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your Changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the Branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## Documentation and Resources

### References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

### Papers
- [YOLOv8 Paper](https://arxiv.org/abs/2304.00501)
- [YOLOv9 Paper](https://arxiv.org/abs/2402.13616)
[Additional papers...]

## Project Information

### License
MIT License - see the [LICENSE](LICENSE) file for details.

### Contact
- **Author**: Dr. Majid Memari
- **Email**: [mmemari@uvu.edu](mailto:mmemari@uvu.edu)
- **Institution**: Utah Valley University
- **Project Link**: [GitHub Repository](https://github.com/memari-majid/Multi-GPU-Object-Detection)

### Citation
```bibtex
@software{memari2024multigpu,
  author = {Memari, Majid},
  title = {Multi-GPU Object Detection with YOLO Family Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/memari-majid/Multi-GPU-Object-Detection}
}
```
