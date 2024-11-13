"""
Wind Turbine Blade Defect Analysis Script
=======================================

Enhanced version with comprehensive defect analysis including:
- Detailed size categorization
- Advanced noise analysis
- Texture and pattern recognition
- Environmental factor analysis
- Defect clustering analysis
- Comprehensive visualization
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import yaml
import logging
from rich.console import Console
from rich.progress import track

console = Console()

class DefectAnalyzer:
    def __init__(self, data_yaml_path, output_dir="analysis_results"):
        """Initialize with extended analysis capabilities."""
        self.data_yaml_path = data_yaml_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename=self.output_dir / 'analysis.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Load dataset configuration
        self.load_dataset_config()
        
        # Extended statistics tracking
        self.defect_stats = {
            'sizes': [],
            'ratios': [],
            'noise_levels': [],
            'difficulty_scores': [],
            'contrast_ratios': [],
            'texture_complexity': [],
            'edge_strength': [],
            'background_uniformity': [],
            'spatial_distribution': [],
            'defect_categories': [],
            'environmental_factors': []
        }
        
        # Define size categories
        self.size_categories = {
            'tiny': (0, 0.0001),      # <0.01% of image area
            'very_small': (0.0001, 0.001),  # 0.01-0.1% of image area
            'small': (0.001, 0.01),    # 0.1-1% of image area
            'medium': (0.01, 0.05),    # 1-5% of image area
            'large': (0.05, 1.0)       # >5% of image area
        }

    def load_dataset_config(self):
        """Load dataset configuration from YAML file."""
        try:
            with open(self.data_yaml_path, 'r') as f:
                self.config = yaml.safe_load(f)
            console.print(f"[green]Loaded dataset config from {self.data_yaml_path}")
        except Exception as e:
            console.print(f"[red]Error loading dataset config: {str(e)}")
            raise

    def calculate_defect_size(self, width, height, img_width, img_height):
        """Calculate defect size and relative size ratio."""
        defect_area = width * height
        image_area = img_width * img_height
        size_ratio = defect_area / image_area
        return defect_area, size_ratio

    def calculate_noise_level(self, img, bbox):
        """Calculate local noise level around defect."""
        try:
            x, y, w, h = [int(v) for v in bbox]  # Ensure integer coordinates
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return 0
            
            # Extract region with bounds checking
            region = img[y:y+h, x:x+w]
            if region.size == 0:
                return 0
            
            # Calculate noise metrics
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            noise_level = np.std(gray_region)
            return float(noise_level)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Error calculating noise level: {str(e)}")
            return 0

    def calculate_contrast_ratio(self, img, bbox):
        """Calculate contrast ratio between defect and background."""
        try:
            x, y, w, h = [int(v) for v in bbox]  # Ensure integer coordinates
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return 0
            
            # Get defect region
            defect_region = img[y:y+h, x:x+w]
            
            # Get surrounding background (with padding)
            pad = 10
            bg_x1 = max(0, x - pad)
            bg_y1 = max(0, y - pad)
            bg_x2 = min(img.shape[1], x + w + pad)
            bg_y2 = min(img.shape[0], y + h + pad)
            background = img[bg_y1:bg_y2, bg_x1:bg_x2]
            
            if defect_region.size == 0 or background.size == 0:
                return 0
            
            # Convert to grayscale
            defect_gray = cv2.cvtColor(defect_region, cv2.COLOR_BGR2GRAY)
            background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            
            # Calculate intensities
            defect_intensity = np.mean(defect_gray)
            bg_intensity = np.mean(background_gray)
            
            # Calculate contrast ratio
            contrast_ratio = abs(defect_intensity - bg_intensity) / 255.0
            return float(contrast_ratio)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Error calculating contrast ratio: {str(e)}")
            return 0

    def calculate_difficulty_score(self, size_ratio, noise_level, contrast_ratio):
        """Calculate overall detection difficulty score."""
        # Normalize components
        size_factor = 1 - size_ratio  # Smaller objects are harder
        noise_factor = min(noise_level / 100, 1)  # Higher noise is harder
        contrast_factor = 1 - contrast_ratio  # Lower contrast is harder
        
        # Weighted combination
        difficulty_score = (0.4 * size_factor + 
                          0.3 * noise_factor + 
                          0.3 * contrast_factor)
        return difficulty_score

    def analyze_image(self, img_path, label_path):
        """Analyze a single image and its defects."""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            
            img_height, img_width = img.shape[:2]
            console.print(f"[green]Processing image: {img_path} ({img_width}x{img_height})")
            
            # Read annotations
            if not Path(label_path).exists():
                console.print(f"[yellow]Warning: No label file found for {img_path}")
                return
            
            with open(label_path, 'r') as f:
                annotations = f.readlines()
            
            if not annotations:
                console.print(f"[yellow]Warning: No annotations found in {label_path}")
                return
            
            console.print(f"[green]Found {len(annotations)} defects in image")
            
            for idx, ann in enumerate(annotations):
                try:
                    # Parse annotation and clean it
                    parts = ann.strip().split()
                    if len(parts) != 5:
                        console.print(f"[yellow]Warning: Invalid annotation format in {label_path}, line {idx+1}")
                        continue
                    
                    # Convert all parts to float, handling potential errors
                    try:
                        cls = int(float(parts[0]))  # Class should be an integer
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                    except ValueError as ve:
                        console.print(f"[yellow]Warning: Invalid number format in annotation: {ann.strip()}")
                        continue

                    # Validate coordinates are in normalized format (0-1)
                    if not all(0 <= x <= 1 for x in [x_center, y_center, w, h]):
                        console.print(f"[yellow]Warning: Coordinates out of range [0,1] in {label_path}, line {idx+1}")
                        continue
                    
                    # Convert normalized coordinates to pixels
                    width_px = int(w * img_width)
                    height_px = int(h * img_height)
                    x_px = int((x_center * img_width) - (width_px/2))
                    y_px = int((y_center * img_height) - (height_px/2))
                    
                    # Ensure coordinates are within image bounds
                    x_px = max(0, min(x_px, img_width-1))
                    y_px = max(0, min(y_px, img_height-1))
                    width_px = min(width_px, img_width - x_px)
                    height_px = min(height_px, img_height - y_px)
                    
                    # Calculate metrics
                    area, ratio = self.calculate_defect_size(width_px, height_px, img_width, img_height)
                    noise = self.calculate_noise_level(img, (x_px, y_px, width_px, height_px))
                    contrast = self.calculate_contrast_ratio(img, (x_px, y_px, width_px, height_px))
                    difficulty = self.calculate_difficulty_score(ratio, noise, contrast)
                    
                    # Store results
                    self.defect_stats['sizes'].append(area)
                    self.defect_stats['ratios'].append(ratio)
                    self.defect_stats['noise_levels'].append(noise)
                    self.defect_stats['contrast_ratios'].append(contrast)
                    self.defect_stats['difficulty_scores'].append(difficulty)
                    
                    # Debug output
                    console.print(f"[blue]Defect {idx+1}:")
                    console.print(f"  Class: {cls}")
                    console.print(f"  Size: {area:.0f}px² ({ratio*100:.2f}% of image)")
                    console.print(f"  Position: ({x_px}, {y_px}), Size: {width_px}x{height_px}")
                    console.print(f"  Difficulty: {difficulty:.2f}")
                    
                    # Save visualization of the detection
                    debug_img = img.copy()
                    cv2.rectangle(debug_img, 
                                (x_px, y_px), 
                                (x_px + width_px, y_px + height_px), 
                                (0, 255, 0), 2)
                    
                    debug_dir = Path(self.output_dir) / 'debug_visualizations'
                    debug_dir.mkdir(exist_ok=True)
                    cv2.imwrite(
                        str(debug_dir / f"{Path(img_path).stem}_defect_{idx+1}.jpg"),
                        debug_img
                    )
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Error processing annotation {idx+1} in {label_path}: {str(e)}")
                    continue
                
        except Exception as e:
            console.print(f"[red]Error analyzing {img_path}: {str(e)}")
            logging.error(f"Error analyzing {img_path}: {str(e)}")
            raise

    def analyze_dataset(self):
        """Analyze all images in the dataset."""
        console.print("[bold blue]Starting dataset analysis...")
        
        try:
            # Initialize lists to store all defect information
            defect_info = {
                'image_path': [],
                'defect_class': [],
                'sizes': [],
                'ratios': [],
                'noise_levels': [],
                'difficulty_scores': [],
                'contrast_ratios': [],
                'width_px': [],
                'height_px': [],
                'area_px': [],
                'position_x': [],
                'position_y': []
            }
            
            # Check if train.txt exists
            train_path = Path(self.config['train'])
            if not train_path.exists():
                console.print(f"[red]Error: Training file not found at {train_path}")
                return
            
            # Get image paths from train.txt
            with open(train_path, 'r') as f:
                image_paths = f.readlines()
            
            if not image_paths:
                console.print("[red]Error: No images found in training file")
                return
            
            console.print(f"[green]Found {len(image_paths)} images in dataset")
            
            total_defects = 0
            processed_images = 0
            failed_images = []
            
            # Create debug directory
            debug_dir = Path(self.output_dir) / 'debug_visualizations'
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in track(image_paths, description="Analyzing images"):
                img_path = img_path.strip()
                img_file = Path(img_path)
                
                if not img_file.exists():
                    console.print(f"[yellow]Warning: Image not found: {img_path}")
                    failed_images.append(img_path)
                    continue
                
                label_path = img_file.with_suffix('.txt')
                if not label_path.exists():
                    console.print(f"[yellow]Warning: No label file found for {img_path}")
                    failed_images.append(img_path)
                    continue
                
                try:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise ValueError(f"Could not read image: {img_path}")
                    
                    img_height, img_width = img.shape[:2]
                    console.print(f"\n[green]Processing: {img_path} ({img_width}x{img_height})")
                    
                    # Read annotations
                    with open(label_path, 'r') as f:
                        annotations = f.readlines()
                    
                    if not annotations:
                        console.print(f"[yellow]Warning: No annotations found in {label_path}")
                        continue
                    
                    console.print(f"[blue]Found {len(annotations)} defects")
                    
                    # Process each defect
                    for idx, ann in enumerate(annotations):
                        try:
                            parts = ann.strip().split()
                            if len(parts) != 5:
                                console.print(f"[yellow]Warning: Invalid annotation format: {ann}")
                                continue
                            
                            # Parse annotation
                            cls = int(float(parts[0]))
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            # Convert to pixels
                            width_px = int(w * img_width)
                            height_px = int(h * img_height)
                            x_px = int((x_center * img_width) - (width_px/2))
                            y_px = int((y_center * img_height) - (height_px/2))
                            
                            # Calculate metrics
                            area_px = width_px * height_px
                            ratio = area_px / (img_width * img_height)
                            noise = self.calculate_noise_level(img, (x_px, y_px, width_px, height_px))
                            contrast = self.calculate_contrast_ratio(img, (x_px, y_px, width_px, height_px))
                            difficulty = self.calculate_difficulty_score(ratio, noise, contrast)
                            
                            # Store all information
                            defect_info['image_path'].append(img_path)
                            defect_info['defect_class'].append(cls)
                            defect_info['sizes'].append(area_px)
                            defect_info['ratios'].append(ratio)
                            defect_info['noise_levels'].append(noise)
                            defect_info['difficulty_scores'].append(difficulty)
                            defect_info['contrast_ratios'].append(contrast)
                            defect_info['width_px'].append(width_px)
                            defect_info['height_px'].append(height_px)
                            defect_info['area_px'].append(area_px)
                            defect_info['position_x'].append(x_px)
                            defect_info['position_y'].append(y_px)
                            
                            # Debug output
                            console.print(f"[blue]Defect {idx+1}:")
                            console.print(f"  Class: {cls}")
                            console.print(f"  Size: {area_px}px² ({ratio*100:.2f}% of image)")
                            console.print(f"  Position: ({x_px}, {y_px}), Size: {width_px}x{height_px}")
                            console.print(f"  Difficulty: {difficulty:.2f}")
                            
                            # Save visualization
                            debug_img = img.copy()
                            cv2.rectangle(debug_img, 
                                        (x_px, y_px), 
                                        (x_px + width_px, y_px + height_px), 
                                        (0, 255, 0), 2)
                            
                            cv2.imwrite(
                                str(debug_dir / f"{img_file.stem}_defect_{idx+1}.jpg"),
                                debug_img
                            )
                            
                            total_defects += 1
                            
                        except Exception as e:
                            console.print(f"[yellow]Warning: Error processing defect {idx+1}: {str(e)}")
                            continue
                    
                    processed_images += 1
                    
                except Exception as e:
                    console.print(f"[red]Error processing {img_path}: {str(e)}")
                    failed_images.append(img_path)
                    continue
            
            # Create DataFrame with all defect information
            df = pd.DataFrame(defect_info)
            
            # Save detailed CSV
            df.to_csv(self.output_dir / 'detailed_defect_analysis.csv', index=False)
            
            # Print analysis summary
            console.print("\n[bold blue]Dataset Analysis Summary:")
            console.print(f"Total images found: {len(image_paths)}")
            console.print(f"Successfully processed images: {processed_images}")
            console.print(f"Total defects found: {total_defects}")
            console.print(f"Failed images: {len(failed_images)}")
            
            if failed_images:
                with open(self.output_dir / 'failed_images.txt', 'w') as f:
                    for img in failed_images:
                        f.write(f"{img}\n")
                console.print(f"[yellow]List of failed images saved to {self.output_dir}/failed_images.txt")
            
            if total_defects > 0:
                # Generate analysis reports
                self.generate_defect_analysis_report(df)
                self.generate_visualizations(df)
                console.print("[green]Analysis reports and visualizations generated successfully")
            else:
                console.print("[red]No defects found in dataset. Please check your data and annotations.")
                
        except Exception as e:
            console.print(f"[red]Error analyzing dataset: {str(e)}")
            logging.error(f"Error analyzing dataset: {str(e)}")
            raise

    def generate_defect_analysis_report(self, df):
        """Generate comprehensive analysis report from DataFrame."""
        try:
            report = {
                'dataset_summary': {
                    'total_images': len(df['image_path'].unique()),
                    'total_defects': len(df),
                    'defects_per_image': len(df) / len(df['image_path'].unique()),
                    'unique_classes': df['defect_class'].unique().tolist()
                },
                'size_statistics': {
                    'mean_area_px': float(df['area_px'].mean()),
                    'median_area_px': float(df['area_px'].median()),
                    'std_area_px': float(df['area_px'].std()),
                    'mean_ratio': float(df['ratios'].mean()),
                    'median_ratio': float(df['ratios'].median()),
                    'std_ratio': float(df['ratios'].std())
                },
                'difficulty_assessment': {
                    'mean_difficulty': float(df['difficulty_scores'].mean()),
                    'high_difficulty_count': int(len(df[df['difficulty_scores'] > 0.7])),
                    'medium_difficulty_count': int(len(df[(df['difficulty_scores'] > 0.3) & (df['difficulty_scores'] <= 0.7)])),
                    'low_difficulty_count': int(len(df[df['difficulty_scores'] <= 0.3]))
                },
                'size_distribution': {
                    'tiny': int(len(df[df['ratios'] < 0.0001])),
                    'very_small': int(len(df[(df['ratios'] >= 0.0001) & (df['ratios'] < 0.001)])),
                    'small': int(len(df[(df['ratios'] >= 0.001) & (df['ratios'] < 0.01)])),
                    'medium': int(len(df[(df['ratios'] >= 0.01) & (df['ratios'] < 0.05)])),
                    'large': int(len(df[df['ratios'] >= 0.05]))
                }
            }
            
            # Save report
            with open(self.output_dir / 'analysis_report.yaml', 'w') as f:
                yaml.dump(report, f, default_flow_style=False)
            
            # Print summary
            console.print("\n[bold green]Analysis Report Summary:")
            console.print(f"Total Images: {report['dataset_summary']['total_images']}")
            console.print(f"Total Defects: {report['dataset_summary']['total_defects']}")
            console.print(f"Average Defects per Image: {report['dataset_summary']['defects_per_image']:.2f}")
            console.print("\nSize Distribution:")
            for category, count in report['size_distribution'].items():
                console.print(f"  {category}: {count}")
            console.print("\nDifficulty Distribution:")
            console.print(f"  High: {report['difficulty_assessment']['high_difficulty_count']}")
            console.print(f"  Medium: {report['difficulty_assessment']['medium_difficulty_count']}")
            console.print(f"  Low: {report['difficulty_assessment']['low_difficulty_count']}")
            
            return report
            
        except Exception as e:
            console.print(f"[red]Error generating analysis report: {str(e)}")
            logging.error(f"Error generating analysis report: {str(e)}")
            return None

    def analyze_texture_complexity(self, img_region):
        """Analyze texture complexity of defect region."""
        if img_region.size == 0:
            return 0
            
        # Convert to grayscale
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features
        glcm = self.calculate_glcm(gray)
        contrast = self.calculate_glcm_contrast(glcm)
        homogeneity = self.calculate_glcm_homogeneity(glcm)
        energy = self.calculate_glcm_energy(glcm)
        
        # Combine into complexity score
        complexity = (contrast * 0.4 + (1 - homogeneity) * 0.3 + (1 - energy) * 0.3)
        return complexity

    def calculate_glcm(self, gray_img):
        """Calculate Gray-Level Co-occurrence Matrix."""
        glcm = np.zeros((256, 256))
        rows, cols = gray_img.shape
        
        for i in range(rows-1):
            for j in range(cols-1):
                i_val = gray_img[i, j]
                j_val = gray_img[i+1, j+1]
                glcm[i_val, j_val] += 1
                
        # Normalize
        glcm = glcm / glcm.sum()
        return glcm

    def analyze_edge_strength(self, img_region):
        """Analyze edge strength in defect region."""
        if img_region.size == 0:
            return 0
            
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate edge magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return np.mean(magnitude)

    def analyze_environmental_factors(self, img, bbox):
        """Analyze environmental factors affecting defect visibility."""
        x, y, w, h = bbox
        
        # Expand region for context
        context_region = self.get_context_region(img, bbox, expansion_factor=2.0)
        
        # Analyze lighting conditions
        lighting_score = self.analyze_lighting(context_region)
        
        # Analyze surface conditions
        surface_score = self.analyze_surface_conditions(context_region)
        
        # Analyze background complexity
        background_score = self.analyze_background_complexity(context_region)
        
        return {
            'lighting': lighting_score,
            'surface': surface_score,
            'background': background_score
        }

    def analyze_spatial_distribution(self, img_width, img_height, x_center, y_center):
        """Analyze spatial distribution of defects."""
        # Normalize coordinates
        x_norm = x_center / img_width
        y_norm = y_center / img_height
        
        # Calculate position-based metrics
        edge_proximity = min(x_norm, 1-x_norm, y_norm, 1-y_norm)
        center_distance = np.sqrt((x_norm - 0.5)**2 + (y_norm - 0.5)**2)
        
        return {
            'edge_proximity': edge_proximity,
            'center_distance': center_distance
        }

    def generate_comprehensive_report(self, df):
        """Generate comprehensive analysis report."""
        # Size analysis
        size_distribution = self.analyze_size_distribution(df)
        
        # Difficulty analysis
        difficulty_analysis = self.analyze_difficulty_factors(df)
        
        # Environmental impact
        environmental_impact = self.analyze_environmental_impact(df)
        
        # Spatial analysis
        spatial_analysis = self.analyze_spatial_patterns(df)
        
        # Generate detailed report
        report = {
            'dataset_overview': {
                'total_images': len(df['image_path'].unique()),
                'total_defects': len(df),
                'defects_per_image': len(df) / len(df['image_path'].unique())
            },
            'size_analysis': size_distribution,
            'difficulty_analysis': difficulty_analysis,
            'environmental_analysis': environmental_impact,
            'spatial_analysis': spatial_analysis,
            'recommendations': self.generate_recommendations(df)
        }
        
        return report

    def generate_visualizations(self, df):
        """Generate comprehensive visualizations."""
        # Create visualization directory
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Size distribution plots
        self.plot_size_distributions(df, vis_dir)
        
        # Difficulty analysis plots
        self.plot_difficulty_analysis(df, vis_dir)
        
        # Spatial distribution plots
        self.plot_spatial_distribution(df, vis_dir)
        
        # Correlation analysis
        self.plot_correlation_analysis(df, vis_dir)
        
        # Environmental factor analysis
        self.plot_environmental_analysis(df, vis_dir)

    def generate_recommendations(self, df):
        """Generate training recommendations based on analysis."""
        recommendations = {
            'model_architecture': self.recommend_architecture(df),
            'training_strategy': self.recommend_training_strategy(df),
            'data_augmentation': self.recommend_augmentation(df),
            'detection_thresholds': self.recommend_thresholds(df)
        }
        return recommendations

    def save_analysis_results(self, df, report):
        """Save comprehensive analysis results."""
        # Save detailed CSV
        df.to_csv(self.output_dir / 'detailed_analysis.csv', index=False)
        
        # Save summary report
        with open(self.output_dir / 'analysis_report.yaml', 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        # Save visualizations
        self.generate_visualizations(df)
        
        # Generate HTML report
        self.generate_html_report(report, self.output_dir / 'report.html')

    def plot_correlation_analysis(self, df, vis_dir):
        """Plot correlation matrix of metrics."""
        try:
            if df.empty:
                console.print("[yellow]Warning: No data available for correlation analysis")
                return

            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                console.print("[yellow]Warning: Not enough numeric columns for correlation analysis")
                return

            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            
            # Replace NaN values with 0 for visualization
            correlation_matrix = correlation_matrix.fillna(0)
            
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       vmin=-1, 
                       vmax=1, 
                       center=0)
            plt.title('Correlation Matrix of Defect Metrics')
            plt.tight_layout()
            plt.savefig(vis_dir / 'correlation_matrix.png')
            plt.close()

        except Exception as e:
            console.print(f"[red]Error plotting correlation matrix: {str(e)}")
            logging.error(f"Error plotting correlation matrix: {str(e)}")

    def plot_size_distributions(self, df, vis_dir):
        """Plot size distribution analysis."""
        try:
            # Size ratio distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='ratios', bins=50)
            plt.title('Distribution of Defect Size Ratios')
            plt.xlabel('Defect-to-Image Ratio')
            plt.ylabel('Count')
            plt.savefig(vis_dir / 'size_ratio_distribution.png')
            plt.close()

            # Absolute size distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='area_px', bins=50)
            plt.title('Distribution of Defect Sizes (pixels²)')
            plt.xlabel('Area (pixels²)')
            plt.ylabel('Count')
            plt.savefig(vis_dir / 'size_distribution.png')
            plt.close()

            # Size categories
            size_categories = pd.cut(df['ratios'], 
                                   bins=[0, 0.0001, 0.001, 0.01, 0.05, 1.0],
                                   labels=['Tiny', 'Very Small', 'Small', 'Medium', 'Large'])
            
            plt.figure(figsize=(10, 6))
            size_categories.value_counts().plot(kind='bar')
            plt.title('Distribution of Defect Size Categories')
            plt.xlabel('Size Category')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(vis_dir / 'size_categories.png')
            plt.close()

        except Exception as e:
            console.print(f"[red]Error plotting size distributions: {str(e)}")
            logging.error(f"Error plotting size distributions: {str(e)}")

    def plot_difficulty_analysis(self, df, vis_dir):
        """Plot difficulty analysis visualizations."""
        try:
            # Difficulty score distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='difficulty_scores', bins=50)
            plt.title('Distribution of Difficulty Scores')
            plt.xlabel('Difficulty Score')
            plt.ylabel('Count')
            plt.savefig(vis_dir / 'difficulty_distribution.png')
            plt.close()

            # Difficulty vs Size
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='ratios', y='difficulty_scores')
            plt.title('Difficulty Score vs Defect Size')
            plt.xlabel('Defect-to-Image Ratio')
            plt.ylabel('Difficulty Score')
            plt.savefig(vis_dir / 'difficulty_vs_size.png')
            plt.close()

            # Difficulty categories
            plt.figure(figsize=(10, 6))
            difficulty_cats = pd.cut(df['difficulty_scores'],
                                   bins=[0, 0.3, 0.7, 1.0],
                                   labels=['Low', 'Medium', 'High'])
            difficulty_cats.value_counts().plot(kind='bar')
            plt.title('Distribution of Difficulty Categories')
            plt.xlabel('Difficulty Category')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(vis_dir / 'difficulty_categories.png')
            plt.close()

        except Exception as e:
            console.print(f"[red]Error plotting difficulty analysis: {str(e)}")
            logging.error(f"Error plotting difficulty analysis: {str(e)}")

    def plot_spatial_distribution(self, df, vis_dir):
        """Plot spatial distribution of defects."""
        try:
            # Create heatmap of defect positions
            plt.figure(figsize=(12, 8))
            plt.hist2d(df['position_x'], df['position_y'], bins=50, cmap='viridis')
            plt.colorbar(label='Number of Defects')
            plt.title('Spatial Distribution of Defects')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.savefig(vis_dir / 'spatial_distribution.png')
            plt.close()

            # Aspect ratio distribution
            plt.figure(figsize=(10, 6))
            aspect_ratios = df['width_px'] / df['height_px']
            sns.histplot(aspect_ratios, bins=50)
            plt.title('Distribution of Defect Aspect Ratios')
            plt.xlabel('Aspect Ratio (width/height)')
            plt.ylabel('Count')
            plt.savefig(vis_dir / 'aspect_ratios.png')
            plt.close()

        except Exception as e:
            console.print(f"[red]Error plotting spatial distribution: {str(e)}")
            logging.error(f"Error plotting spatial distribution: {str(e)}")

    def plot_environmental_analysis(self, df, vis_dir):
        """Plot environmental factor analysis."""
        try:
            # Noise level distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='noise_levels', bins=50)
            plt.title('Distribution of Noise Levels')
            plt.xlabel('Noise Level')
            plt.ylabel('Count')
            plt.savefig(vis_dir / 'noise_distribution.png')
            plt.close()

            # Contrast ratio distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='contrast_ratios', bins=50)
            plt.title('Distribution of Contrast Ratios')
            plt.xlabel('Contrast Ratio')
            plt.ylabel('Count')
            plt.savefig(vis_dir / 'contrast_distribution.png')
            plt.close()

            # Noise vs Contrast
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='noise_levels', y='contrast_ratios')
            plt.title('Noise Level vs Contrast Ratio')
            plt.xlabel('Noise Level')
            plt.ylabel('Contrast Ratio')
            plt.savefig(vis_dir / 'noise_vs_contrast.png')
            plt.close()

        except Exception as e:
            console.print(f"[red]Error plotting environmental analysis: {str(e)}")
            logging.error(f"Error plotting environmental analysis: {str(e)}")

if __name__ == "__main__":
    analyzer = DefectAnalyzer("data_clean/data_clean.yml")
    analyzer.analyze_dataset() 