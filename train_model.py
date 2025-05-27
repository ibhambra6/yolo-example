#!/usr/bin/env python3
"""
Streamlined YOLO Model Training Script
=====================================

This script provides a complete end-to-end workflow for training custom YOLO models
and automatically generating configuration files for the generic classifier.

Features:
    - Interactive training wizard for beginners
    - Command-line interface for advanced users
    - Automatic dataset preparation and validation
    - Real-time training progress monitoring
    - Automatic configuration file generation
    - Integration with the generic classifier GUI

Usage:
    Interactive Mode (Recommended for beginners):
        python train_model.py --interactive
    
    Command Line Mode:
        python train_model.py --dataset_path /path/to/dataset --project_name my_classifier --classes "class1,class2,class3"
    
    Quick Training:
        python train_model.py --dataset_path data --project_name animals --classes "cat,dog,bird" --epochs 100

Author: YOLO Classifier Project
License: Proprietary - Oceaneering International Inc. Use Only
"""

# Standard library imports
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Third-party imports
import yaml

# Configuration constants
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 640
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45

def create_config_file(project_name: str, classes: list, model_path: str, description: str = "Custom YOLO Classifier") -> str:
    """
    Create a YAML configuration file for the trained model.
    
    This function generates a complete configuration file that can be used with the
    generic_yolo_classifier.py application. The config includes model parameters,
    class definitions, GUI settings, and metadata.
    
    Args:
        project_name (str): Name of the project (used for file naming)
        classes (list): List of class names for classification
        model_path (str): Path to the trained model weights file
        description (str, optional): Description of the classifier. Defaults to "Custom YOLO Classifier".
    
    Returns:
        str: Path to the created configuration file
    
    Example:
        >>> create_config_file("animals", ["cat", "dog"], "models/animals.pt", "Pet classifier")
        "configs/animals_classifier.yaml"
    """
    # Build the configuration dictionary with all necessary parameters
    config = {
        'model': {
            'weights': model_path,
            'confidence_threshold': DEFAULT_CONFIDENCE_THRESHOLD,
            'iou_threshold': DEFAULT_IOU_THRESHOLD
        },
        'classes': classes,
        'gui': {
            'title': f"{project_name} Classifier",
            'window_size': [800, 600],
            'theme': 'modern'
        },
        'training': {
            'epochs': DEFAULT_EPOCHS,
            'batch_size': DEFAULT_BATCH_SIZE,
            'image_size': DEFAULT_IMAGE_SIZE,
            'learning_rate': DEFAULT_LEARNING_RATE
        },
        'metadata': {
            'description': description,
            'created_date': datetime.now().isoformat(),
            'version': '1.0'
        }
    }
    
    # Ensure the configs directory exists
    config_path = Path("configs") / f"{project_name}_classifier.yaml"
    config_path.parent.mkdir(exist_ok=True)
    
    # Write the configuration to a YAML file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Configuration file created: {config_path}")
    return str(config_path)

def prepare_yolo_dataset(dataset_path, output_path):
    """Prepare dataset in YOLO format."""
    print(f"📁 Preparing dataset from {dataset_path}")
    
    # Create YOLO dataset structure
    yolo_path = Path(output_path)
    yolo_path.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (yolo_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Import dataset utilities
    sys.path.append('utils')
    from dataset_utils import convert_to_yolo_format, split_dataset
    
    # Convert and split dataset
    convert_to_yolo_format(dataset_path, str(yolo_path))
    split_dataset(str(yolo_path))
    
    print(f"✅ Dataset prepared at {output_path}")
    return str(yolo_path)

def create_dataset_yaml(dataset_path: str, classes: list, project_name: str) -> str:
    """
    Create a dataset.yaml file required for YOLO training.
    
    This function generates the dataset configuration file that YOLOv5 uses to
    understand the dataset structure, class names, and file locations.
    
    Args:
        dataset_path (str): Path to the dataset directory
        classes (list): List of class names in the dataset
        project_name (str): Name of the project (for reference)
    
    Returns:
        str: Path to the created dataset.yaml file
    
    Note:
        The dataset directory should have the following structure:
        dataset_path/
        ├── train/images/
        ├── train/labels/
        ├── val/images/
        ├── val/labels/
        ├── test/images/
        └── test/labels/
    """
    # Create the dataset configuration dictionary
    dataset_config = {
        'path': os.path.abspath(dataset_path),  # Absolute path to dataset
        'train': 'train/images',                # Relative path to training images
        'val': 'val/images',                   # Relative path to validation images
        'test': 'test/images',                 # Relative path to test images
        'nc': len(classes),                    # Number of classes
        'names': classes                       # Class names list
    }
    
    # Write the dataset configuration to YAML file
    yaml_path = Path(dataset_path) / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Dataset YAML created: {yaml_path}")
    return str(yaml_path)

def train_yolo_model(dataset_yaml: str, project_name: str, epochs: int = DEFAULT_EPOCHS, 
                     batch_size: int = DEFAULT_BATCH_SIZE, img_size: int = DEFAULT_IMAGE_SIZE) -> str:
    """
    Train a YOLO model using the YOLOv5 framework.
    
    This function executes the YOLOv5 training process with the specified parameters.
    It handles the training subprocess, monitors progress, and returns the path to
    the best trained model weights.
    
    Args:
        dataset_yaml (str): Path to the dataset configuration YAML file
        project_name (str): Name of the training project (used for output directory)
        epochs (int, optional): Number of training epochs. Defaults to DEFAULT_EPOCHS.
        batch_size (int, optional): Training batch size. Defaults to DEFAULT_BATCH_SIZE.
        img_size (int, optional): Input image size for training. Defaults to DEFAULT_IMAGE_SIZE.
    
    Returns:
        str: Path to the best model weights file, or None if training failed
    
    Raises:
        subprocess.CalledProcessError: If the training process fails
    
    Note:
        - Requires YOLOv5 to be installed (pip install yolov5)
        - Uses pre-trained YOLOv5s weights as starting point
        - Saves model checkpoints every 10 epochs
        - Creates training outputs in runs/train/{project_name}/
    """
    print(f"🚀 Starting training for {project_name}")
    print(f"📊 Training parameters: {epochs} epochs, batch size {batch_size}, image size {img_size}")
    
    # Ensure the runs directory exists for training outputs
    runs_dir = Path("runs/train")
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if pre-trained weights exist, use from models directory if available
    models_weights = Path("models") / "yolov5s.pt"
    weights_path = str(models_weights) if models_weights.exists() else "yolov5s.pt"
    
    # Construct the training command with all parameters
    cmd = [
        sys.executable, "-m", "yolov5.train",  # Use YOLOv5 training module
        "--data", dataset_yaml,                # Dataset configuration file
        "--weights", weights_path,             # Pre-trained weights to start from
        "--epochs", str(epochs),               # Number of training epochs
        "--batch-size", str(batch_size),       # Training batch size
        "--img", str(img_size),               # Input image size
        "--project", "runs/train",            # Output project directory
        "--name", project_name,               # Experiment name
        "--save-period", "10"                 # Save checkpoint every 10 epochs
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    
    try:
        # Execute the training process
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Training completed successfully!")
        
        # Locate the best trained model weights
        weights_path = Path("runs") / "train" / project_name / "weights" / "best.pt"
        if weights_path.exists():
            print(f"✅ Best weights saved at: {weights_path}")
            return str(weights_path)
        else:
            # Fallback to last weights if best weights not found
            last_weights_path = Path("runs") / "train" / project_name / "weights" / "last.pt"
            if last_weights_path.exists():
                print("⚠️ Best weights not found, using last weights")
                return str(last_weights_path)
            else:
                print("❌ No weights file found after training")
                return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error during training: {e}")
        return None

def interactive_training():
    """Interactive training mode."""
    print("🎯 Interactive YOLO Training Setup")
    print("=" * 40)
    
    # Get project details
    project_name = input("Enter project name (e.g., 'my_objects'): ").strip()
    if not project_name:
        project_name = "custom_classifier"
    
    description = input("Enter project description: ").strip()
    if not description:
        description = f"{project_name} YOLO Classifier"
    
    # Get dataset path
    print("\nDataset Setup:")
    print("1. I have a dataset ready in YOLO format")
    print("2. I have images and need help with annotation")
    print("3. I want to use a sample dataset")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        dataset_path = input("Enter path to your YOLO dataset: ").strip()
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return
    elif choice == "2":
        print("📝 For annotation, I recommend using:")
        print("   - LabelImg: https://github.com/tzutalin/labelImg")
        print("   - Roboflow: https://roboflow.com")
        print("   - CVAT: https://cvat.org")
        print("\nAfter annotation, run this script again with option 1.")
        return
    elif choice == "3":
        print("📦 Creating sample dataset structure...")
        dataset_path = create_sample_dataset(project_name)
    else:
        print("❌ Invalid choice")
        return
    
    # Get classes
    print(f"\nEnter class names for {project_name} (comma-separated):")
    classes_input = input("Classes: ").strip()
    classes = [cls.strip() for cls in classes_input.split(',') if cls.strip()]
    
    if not classes:
        print("❌ No classes provided")
        return
    
    # Training parameters
    print(f"\nTraining Parameters:")
    epochs = input("Number of epochs [100]: ").strip()
    epochs = int(epochs) if epochs.isdigit() else 100
    
    batch_size = input("Batch size [16]: ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 16
    
    # Confirm and start training
    print(f"\n📋 Training Summary:")
    print(f"   Project: {project_name}")
    print(f"   Classes: {', '.join(classes)}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    
    confirm = input("\nStart training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Training cancelled")
        return
    
    # Start training workflow
    return train_workflow(dataset_path, classes, project_name, description, epochs, batch_size)

def create_sample_dataset(project_name):
    """Create a sample dataset structure for user to populate."""
    dataset_path = Path("datasets") / project_name
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (dataset_path / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_path / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme_content = f"""# {project_name} Dataset

## Structure
```
{dataset_path}/
├── train/
│   ├── images/     # Training images (.jpg, .png)
│   └── labels/     # Training labels (.txt)
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
└── test/
    ├── images/     # Test images
    └── labels/     # Test labels
```

## Label Format
Each .txt file should contain one line per object:
```
class_id center_x center_y width height
```

Where all coordinates are normalized (0-1).

## Next Steps
1. Add your images to the respective folders
2. Create corresponding label files
3. Run training: `python train_model.py --dataset_path {dataset_path} --project_name {project_name}`
"""
    
    with open(dataset_path / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Sample dataset structure created at: {dataset_path}")
    print(f"📖 See {dataset_path / 'README.md'} for instructions")
    
    return str(dataset_path)

def train_workflow(dataset_path, classes, project_name, description, epochs=100, batch_size=16):
    """Complete training workflow."""
    try:
        # Step 1: Create dataset YAML
        dataset_yaml = create_dataset_yaml(dataset_path, classes, project_name)
        
        # Step 2: Train model
        weights_path = train_yolo_model(dataset_yaml, project_name, epochs, batch_size)
        
        if not weights_path:
            print("❌ Training failed")
            return False
        
        # Step 3: Create configuration file
        config_path = create_config_file(project_name, classes, weights_path, description)
        
        # Step 4: Test the model
        print(f"\n🎉 Training Complete!")
        print(f"📁 Model weights: {weights_path}")
        print(f"⚙️ Configuration: {config_path}")
        print(f"\n🚀 To test your model:")
        print(f"   python generic_yolo_classifier.py --config {config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training workflow: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Streamlined YOLO Training")
    parser.add_argument("--dataset_path", help="Path to dataset")
    parser.add_argument("--project_name", help="Name for the project")
    parser.add_argument("--classes", help="Comma-separated class names")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--description", help="Project description")
    
    args = parser.parse_args()
    
    if args.interactive or not args.dataset_path:
        interactive_training()
    else:
        if not args.project_name or not args.classes:
            print("❌ --project_name and --classes are required in non-interactive mode")
            sys.exit(1)
        
        classes = [cls.strip() for cls in args.classes.split(',')]
        description = args.description or f"{args.project_name} YOLO Classifier"
        
        success = train_workflow(
            args.dataset_path, 
            classes, 
            args.project_name, 
            description,
            args.epochs, 
            args.batch_size
        )
        
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main() 