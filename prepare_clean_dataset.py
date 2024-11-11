import os
from pathlib import Path
import shutil
from rich.console import Console
import yaml
import pandas as pd

console = Console()

def prepare_clean_dataset():
    """Prepare clean dataset structure and copy files."""
    try:
        # Create clean dataset directory structure
        clean_dir = Path('clean_dataset')
        clean_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        images_dir = clean_dir / 'images'
        labels_dir = clean_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # Read the difficult images list
        difficult_df = pd.read_csv('difficult_images_analysis/difficult_images_analysis.csv')
        difficult_images = set(difficult_df['image_path'].tolist())
        
        # Process each split
        splits = ['train', 'val', 'test']
        clean_paths = {}
        
        for split in splits:
            # Create split directories
            (images_dir / split).mkdir(exist_ok=True)
            (labels_dir / split).mkdir(exist_ok=True)
            
            # Read original split file
            with open(f'data/{split}_grass.txt', 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
            
            # Filter out difficult images
            clean_images = [path for path in image_paths if path not in difficult_images]
            clean_paths[split] = []
            
            # Copy clean images and their labels
            for img_path in clean_images:
                # Copy image
                img_name = Path(img_path).name
                new_img_path = images_dir / split / img_name
                shutil.copy2(img_path, new_img_path)
                
                # Copy label if exists
                label_path = img_path.replace('.JPG', '.txt').replace('.jpg', '.txt')
                if os.path.exists(label_path):
                    label_name = Path(label_path).name
                    new_label_path = labels_dir / split / label_name
                    shutil.copy2(label_path, new_label_path)
                
                clean_paths[split].append(str(new_img_path))
            
            # Create split file
            with open(clean_dir / f'{split}.txt', 'w') as f:
                f.write('\n'.join(clean_paths[split]))
            
            console.print(f"[green]Processed {split} split: {len(clean_paths[split])} clean images")
        
        # Create data.yaml for clean dataset
        clean_yaml = {
            'path': str(clean_dir.absolute()),
            'train': str((clean_dir / 'train.txt').absolute()),
            'val': str((clean_dir / 'val.txt').absolute()),
            'test': str((clean_dir / 'test.txt').absolute()),
            'nc': 2,  # Number of classes
            'names': ['crack', 'erosion']  # Class names
        }
        
        with open(clean_dir / 'data.yaml', 'w') as f:
            yaml.dump(clean_yaml, f, default_flow_style=False)
        
        console.print(f"[bold green]Clean dataset prepared successfully in {clean_dir}")
        return str(clean_dir / 'data.yaml')
        
    except Exception as e:
        console.print(f"[bold red]Error preparing clean dataset: {str(e)}")
        return None

if __name__ == "__main__":
    clean_yaml_path = prepare_clean_dataset()
    if clean_yaml_path:
        console.print(f"\n[bold green]Clean dataset ready for training!")
        console.print(f"Use this path in training: {clean_yaml_path}") 