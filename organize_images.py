"""
Wind Turbine Blade Image Organization Tool

This module organizes and annotates wind turbine blade images by categorizing them into
faulty and healthy groups. It processes images from multiple dataset splits and creates
annotated versions with visual indicators of defects and metadata.

Key Features:
- Filters and organizes grass-type blade images
- Separates images into faulty and healthy categories
- Creates annotated versions with visual fault indicators
- Generates detailed organization summary
- Supports multiple dataset splits (train/val/test)

Dependencies:
    - opencv-python (cv2)
    - rich
    - pathlib
    - logging
"""

import cv2
import logging
from pathlib import Path
from rich.console import Console
from rich.progress import track

# Initialize console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('image_organization.log')
    ]
)

def filter_grass_images(file_path):
    """
    Filter and extract only Grass-type images from a dataset file.
    
    Args:
        file_path (str): Path to the original dataset file containing image paths.
        
    Returns:
        str: Path to the new file containing only Grass image paths, or None if error occurs.
        
    Note:
        Creates a new file with '_grass.txt' suffix containing filtered paths.
    """
    try:
        with open(file_path, 'r') as f:
            image_paths = f.readlines()
        
        grass_images = [path.strip() for path in image_paths if 'Grass' in path]
        
        grass_file = str(Path(file_path)).replace('.txt', '_grass.txt')
        with open(grass_file, 'w') as f:
            f.write('\n'.join(grass_images))
        
        return grass_file
        
    except Exception as e:
        console.print(f"[bold red]Error filtering Grass images: {str(e)}")
        return None

def prepare_dataset_files():
    """
    Prepare dataset files by filtering for Grass-type images.
    
    Returns:
        dict: Dictionary containing paths to filtered dataset files for each split
              (train, val, test), or None if preparation fails.
    
    Note:
        Expects original dataset files in 'data/' directory:
        - data/train.txt
        - data/val.txt
        - data/test.txt
    """
    try:
        dataset_files = {
            'train': Path('data/train.txt'),
            'val': Path('data/val.txt'),
            'test': Path('data/test.txt')
        }
        
        grass_files = {}
        
        for dataset_type, file_path in dataset_files.items():
            if not file_path.exists():
                console.print(f"[bold red]Error: {file_path} not found")
                return None
                
            grass_file = filter_grass_images(file_path)
            if grass_file:
                grass_files[dataset_type] = grass_file
            
        return grass_files
        
    except Exception as e:
        console.print(f"[bold red]Error preparing dataset files: {str(e)}")
        return None

def organize_and_annotate_images(dataset_files, output_base_dir):
    """
    Organize and annotate images into faulty and healthy categories.
    
    This function:
    1. Processes images from all dataset splits
    2. Categorizes images as faulty or healthy based on label files
    3. Creates annotated versions with visual indicators
    4. Generates organization summary
    
    Args:
        dataset_files (dict): Dictionary of paths to dataset files for each split
        output_base_dir (str): Base directory for organized output
        
    Returns:
        bool: True if organization successful, False otherwise
        
    Directory Structure Created:
    output_base_dir/
    ├── faulty/
    │   ├── train_image1.jpg
    │   ├── val_image2.jpg
    │   └── ...
    ├── healthy/
    │   ├── train_image3.jpg
    │   ├── test_image4.jpg
    │   └── ...
    └── organization_summary.txt
    """
    try:
        # Create output directories
        output_base = Path(output_base_dir)
        faulty_dir = output_base / "faulty"
        healthy_dir = output_base / "healthy"
        
        for dir_path in [faulty_dir, healthy_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        total_processed = 0
        faulty_count = 0
        healthy_count = 0
        
        # Process all dataset splits
        for split_name, file_path in dataset_files.items():
            console.print(f"\n[blue]Processing {split_name} split...")
            
            with open(file_path, 'r') as f:
                image_paths = f.readlines()
            
            for img_path in track(image_paths, description=f"Processing {split_name} images"):
                img_path = img_path.strip()
                img = cv2.imread(img_path)
                
                if img is None:
                    logging.warning(f"Could not read image: {img_path}")
                    continue
                
                # Get corresponding label path
                label_path = img_path.replace('.JPG', '.txt').replace('.jpg', '.txt')
                is_faulty = False
                
                # Create annotated image
                img_annotated = img.copy()
                
                if Path(label_path).exists():
                    with open(label_path, 'r') as f:
                        annotations = f.readlines()
                        if annotations:
                            is_faulty = True
                            # Draw boxes for each annotation
                            for ann in annotations:
                                cls, x, y, w, h = map(float, ann.strip().split())
                                img_h, img_w = img.shape[:2]
                                x1 = int((x - w/2) * img_w)
                                y1 = int((y - h/2) * img_h)
                                x2 = int((x + w/2) * img_w)
                                y2 = int((y + h/2) * img_h)
                                
                                # Draw rectangle and label
                                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for faults
                                cv2.putText(img_annotated, 'Fault', (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Add image info overlay
                img_name = Path(img_path).name
                status_text = f"FAULTY - {split_name}" if is_faulty else f"HEALTHY - {split_name}"
                cv2.putText(img_annotated, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 0, 255) if is_faulty else (0, 255, 0), 2)  # Red for faulty, Green for healthy
                cv2.putText(img_annotated, img_name, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Save to appropriate directory
                save_dir = faulty_dir if is_faulty else healthy_dir
                save_path = save_dir / f"{split_name}_{img_name}"
                
                cv2.imwrite(str(save_path), img_annotated)
                
                # Update counters
                total_processed += 1
                if is_faulty:
                    faulty_count += 1
                else:
                    healthy_count += 1
        
        # Generate summary
        console.print("\n[bold green]Image Organization Complete!")
        console.print(f"Total images processed: {total_processed}")
        console.print(f"Faulty blades: {faulty_count}")
        console.print(f"Healthy blades: {healthy_count}")
        
        # Save summary to file
        summary_path = output_base / "organization_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Wind Turbine Blade Analysis Summary\n")
            f.write(f"================================\n")
            f.write(f"Total images processed: {total_processed}\n")
            f.write(f"Faulty blades: {faulty_count}\n")
            f.write(f"Healthy blades: {healthy_count}\n")
            f.write(f"Fault ratio: {faulty_count/total_processed:.2%}\n")
            f.write(f"\nNote: Images are organized by condition (faulty/healthy)\n")
            f.write(f"and include visual annotations showing detected faults.\n")
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]Error organizing images: {str(e)}")
        logging.error(f"Error organizing images: {str(e)}")
        return False

def main():
    """
    Main function to execute the image organization workflow.
    
    Workflow:
    1. Prepare dataset files by filtering for Grass images
    2. Create output directory structure
    3. Organize and annotate images
    4. Generate summary report
    """
    console.print("[bold blue]Starting Wind Turbine Blade Image Organization...")
    
    # Prepare dataset files
    dataset_files = prepare_dataset_files()
    if not dataset_files:
        console.print("[bold red]Failed to prepare dataset files. Exiting...")
        return
    
    # Create output directory
    output_dir = "annotated_blade_images"
    
    # Organize and annotate images
    success = organize_and_annotate_images(dataset_files, output_dir)
    
    if success:
        console.print(f"\n[bold green]Successfully organized images in {output_dir}")
        console.print("[blue]Check the organization_summary.txt file for details")
    else:
        console.print("[bold red]Failed to organize images")

if __name__ == "__main__":
    main() 