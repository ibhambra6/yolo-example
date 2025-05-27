#!/usr/bin/env python3
"""
Configuration File Generator for YOLO Classifiers
=================================================

This script provides an easy way to create YAML configuration files for the
generic YOLO classifier application. It supports both interactive and command-line
modes for maximum flexibility.

Features:
    - Interactive configuration wizard
    - Command-line interface for automation
    - Template generation for common use cases
    - Validation of configuration parameters
    - Integration with existing model weights

Usage:
    Interactive Mode:
        python create_config.py --interactive
    
    Command Line Mode:
        python create_config.py --name my_classifier --classes "cat,dog,bird" --weights model.pt
    
    Quick Config:
        python create_config.py --name animals --classes "cat,dog" --weights models/animals.pt

Author: YOLO Classifier Project
License: Proprietary - Oceaneering International Inc. Use Only
"""

# Standard library imports
import argparse
from datetime import datetime
from pathlib import Path

# Third-party imports
import yaml

# Configuration constants
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45
DEFAULT_WINDOW_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600
DEFAULT_WEIGHTS = "yolov5s.pt"

def create_config_interactive() -> str:
    """
    Interactive configuration creation wizard.
    
    This function provides a step-by-step interactive interface for creating
    YAML configuration files. It guides users through all necessary settings
    including model parameters, class definitions, and GUI customization.
    
    Returns:
        str: Path to the created configuration file
    
    Note:
        - Provides sensible defaults for all parameters
        - Validates user input where possible
        - Creates the configs directory if it doesn't exist
        - Generates a complete, ready-to-use configuration file
    """
    print("🔧 YOLO Configuration Generator")
    print("=" * 35)
    
    # Collect basic project information
    project_name = input("Project name: ").strip()
    if not project_name:
        project_name = "custom_classifier"
        print(f"Using default project name: {project_name}")
    
    title = input(f"GUI title [{project_name} Classifier]: ").strip()
    if not title:
        title = f"{project_name} Classifier"
    
    description = input("Description: ").strip()
    if not description:
        description = f"Custom {project_name} classifier"
    
    # Configure model parameters
    print("\n🤖 Model Settings:")
    weights_path = input(f"Model weights path [{DEFAULT_WEIGHTS}]: ").strip()
    if not weights_path:
        weights_path = DEFAULT_WEIGHTS
    
    # Get confidence threshold with validation
    confidence_input = input(f"Confidence threshold [{DEFAULT_CONFIDENCE}]: ").strip()
    try:
        confidence = float(confidence_input) if confidence_input else DEFAULT_CONFIDENCE
        confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
    except ValueError:
        print(f"Invalid confidence value, using default: {DEFAULT_CONFIDENCE}")
        confidence = DEFAULT_CONFIDENCE
    
    # Get IoU threshold with validation
    iou_input = input(f"IoU threshold [{DEFAULT_IOU}]: ").strip()
    try:
        iou = float(iou_input) if iou_input else DEFAULT_IOU
        iou = max(0.0, min(1.0, iou))  # Clamp between 0 and 1
    except ValueError:
        print(f"Invalid IoU value, using default: {DEFAULT_IOU}")
        iou = DEFAULT_IOU
    
    # Configure class definitions
    print("\n📋 Classes:")
    print("Enter class names (comma-separated):")
    classes_input = input("Classes: ").strip()
    classes = [cls.strip() for cls in classes_input.split(',') if cls.strip()]
    
    if not classes:
        classes = ["object"]
        print("No classes provided, using default: ['object']")
    
    # Configure GUI settings
    print("\n🎨 GUI Settings:")
    width_input = input(f"Window width [{DEFAULT_WINDOW_WIDTH}]: ").strip()
    try:
        width = int(width_input) if width_input.isdigit() else DEFAULT_WINDOW_WIDTH
        width = max(400, min(2000, width))  # Reasonable window size limits
    except ValueError:
        width = DEFAULT_WINDOW_WIDTH
    
    height_input = input(f"Window height [{DEFAULT_WINDOW_HEIGHT}]: ").strip()
    try:
        height = int(height_input) if height_input.isdigit() else DEFAULT_WINDOW_HEIGHT
        height = max(300, min(1500, height))  # Reasonable window size limits
    except ValueError:
        height = DEFAULT_WINDOW_HEIGHT
    
    # Build the complete configuration dictionary
    config = {
        'model': {
            'weights': weights_path,
            'confidence_threshold': confidence,
            'iou_threshold': iou
        },
        'classes': classes,
        'gui': {
            'title': title,
            'window_size': [width, height],
            'theme': 'modern'
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'image_size': 640,
            'learning_rate': 0.01
        },
        'metadata': {
            'description': description,
            'created_date': datetime.now().isoformat(),
            'version': '1.0'
        }
    }
    
    # Save the configuration to file
    config_path = Path("configs") / f"{project_name}_classifier.yaml"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"\n✅ Configuration saved: {config_path}")
    print(f"🚀 Test with: python generic_yolo_classifier.py --config {config_path}")
    
    return str(config_path)

def create_config_from_args(args):
    """Create config from command line arguments."""
    classes = [cls.strip() for cls in args.classes.split(',') if cls.strip()]
    
    config = {
        'model': {
            'weights': args.weights,
            'confidence_threshold': args.confidence,
            'iou_threshold': args.iou
        },
        'classes': classes,
        'gui': {
            'title': args.title or f"{args.name} Classifier",
            'window_size': [args.width, args.height],
            'theme': 'modern'
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'image_size': 640,
            'learning_rate': 0.01
        },
        'metadata': {
            'description': args.description or f"{args.name} classifier",
            'created_date': datetime.now().isoformat(),
            'version': '1.0'
        }
    }
    
    config_path = Path("configs") / f"{args.name}_classifier.yaml"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Configuration saved: {config_path}")
    return str(config_path)

def main():
    parser = argparse.ArgumentParser(description="Create YOLO configuration files")
    parser.add_argument("--name", help="Project name")
    parser.add_argument("--classes", help="Comma-separated class names")
    parser.add_argument("--weights", default="yolov5s.pt", help="Model weights path")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--title", help="GUI title")
    parser.add_argument("--description", help="Project description")
    parser.add_argument("--width", type=int, default=800, help="Window width")
    parser.add_argument("--height", type=int, default=600, help="Window height")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or not args.name:
        create_config_interactive()
    else:
        if not args.classes:
            print("❌ --classes is required in non-interactive mode")
            return
        create_config_from_args(args)

if __name__ == "__main__":
    main() 