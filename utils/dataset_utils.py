#!/usr/bin/env python3
"""
dataset_utils.py
================
Utilities for preparing and managing datasets for YOLO training.

This module provides tools to:
1. Convert datasets from various formats to YOLO format
2. Split datasets into train/validation sets
3. Validate dataset structure
4. Generate dataset statistics
"""

import os
import json
import yaml
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import cv2
import numpy as np
from PIL import Image


class DatasetConverter:
    """Convert datasets from various formats to YOLO format."""
    
    @staticmethod
    def coco_to_yolo(coco_json_path: str, images_dir: str, output_dir: str, 
                     class_names: Optional[List[str]] = None) -> str:
        """Convert COCO format dataset to YOLO format."""
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create class mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        if class_names is None:
            class_names = list(categories.values())
        
        class_to_id = {name: i for i, name in enumerate(class_names)}
        
        # Process images and annotations
        images_info = {img['id']: img for img in coco_data['images']}
        annotations_by_image = defaultdict(list)
        
        for ann in coco_data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
        
        labels_dir = output_path / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        for image_id, image_info in images_info.items():
            image_name = Path(image_info['file_name']).stem
            label_file = labels_dir / f"{image_name}.txt"
            
            with open(label_file, 'w') as f:
                for ann in annotations_by_image[image_id]:
                    category_name = categories[ann['category_id']]
                    if category_name not in class_to_id:
                        continue
                    
                    class_id = class_to_id[category_name]
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # Convert to YOLO format (normalized center coordinates)
                    img_width = image_info['width']
                    img_height = image_info['height']
                    
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        # Copy images
        images_output_dir = output_path / "images"
        images_output_dir.mkdir(exist_ok=True)
        
        for image_info in images_info.values():
            src_path = Path(images_dir) / image_info['file_name']
            dst_path = images_output_dir / image_info['file_name']
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(output_path),
            'train': 'images',
            'val': 'images',  # Will be split later
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return str(yaml_path)


class DatasetSplitter:
    """Split datasets into train/validation sets."""
    
    @staticmethod
    def split_dataset(dataset_dir: str, train_ratio: float = 0.8, 
                     val_ratio: float = 0.2, test_ratio: float = 0.0,
                     random_seed: int = 42) -> Dict[str, str]:
        """Split dataset into train/val/test sets."""
        random.seed(random_seed)
        
        dataset_path = Path(dataset_dir)
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError("Dataset must have 'images' and 'labels' directories")
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        image_files = [f.stem for f in image_files]  # Get filenames without extension
        
        # Shuffle and split
        random.shuffle(image_files)
        
        total_files = len(image_files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:] if test_ratio > 0 else []
        
        # Create split directories
        splits = {'train': train_files, 'val': val_files}
        if test_files:
            splits['test'] = test_files
        
        split_dirs = {}
        
        for split_name, file_list in splits.items():
            split_dir = dataset_path / split_name
            split_images_dir = split_dir / "images"
            split_labels_dir = split_dir / "labels"
            
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)
            
            for filename in file_list:
                # Copy image files
                for ext in ['.jpg', '.png', '.jpeg']:
                    src_img = images_dir / f"{filename}{ext}"
                    if src_img.exists():
                        dst_img = split_images_dir / f"{filename}{ext}"
                        shutil.copy2(src_img, dst_img)
                        break
                
                # Copy label files
                src_label = labels_dir / f"{filename}.txt"
                if src_label.exists():
                    dst_label = split_labels_dir / f"{filename}.txt"
                    shutil.copy2(src_label, dst_label)
            
            split_dirs[split_name] = str(split_dir)
        
        # Update dataset.yaml
        yaml_path = dataset_path / "dataset.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config['train'] = 'train/images'
            config['val'] = 'val/images'
            if 'test' in split_dirs:
                config['test'] = 'test/images'
            
            with open(yaml_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        return split_dirs


class DatasetValidator:
    """Validate dataset structure and quality."""
    
    @staticmethod
    def validate_yolo_dataset(dataset_yaml: str) -> Dict[str, any]:
        """Validate YOLO dataset and return statistics."""
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_path = Path(config['path'])
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required fields
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in config:
                results['errors'].append(f"Missing required field: {field}")
                results['valid'] = False
        
        if not results['valid']:
            return results
        
        # Check paths exist
        train_path = dataset_path / config['train']
        val_path = dataset_path / config['val']
        
        if not train_path.exists():
            results['errors'].append(f"Training path does not exist: {train_path}")
            results['valid'] = False
        
        if not val_path.exists():
            results['errors'].append(f"Validation path does not exist: {val_path}")
            results['valid'] = False
        
        if not results['valid']:
            return results
        
        # Collect statistics
        stats = DatasetValidator._collect_statistics(train_path, val_path, config['names'])
        results['statistics'] = stats
        
        # Check for issues
        if stats['train']['images'] == 0:
            results['errors'].append("No training images found")
            results['valid'] = False
        
        if stats['val']['images'] == 0:
            results['warnings'].append("No validation images found")
        
        if stats['train']['orphaned_labels'] > 0:
            results['warnings'].append(f"{stats['train']['orphaned_labels']} orphaned label files in training set")
        
        if stats['val']['orphaned_labels'] > 0:
            results['warnings'].append(f"{stats['val']['orphaned_labels']} orphaned label files in validation set")
        
        return results
    
    @staticmethod
    def _collect_statistics(train_path: Path, val_path: Path, class_names: List[str]) -> Dict:
        """Collect dataset statistics."""
        stats = {
            'train': DatasetValidator._analyze_split(train_path, class_names),
            'val': DatasetValidator._analyze_split(val_path, class_names)
        }
        
        # Overall statistics
        stats['total_images'] = stats['train']['images'] + stats['val']['images']
        stats['total_labels'] = stats['train']['labels'] + stats['val']['labels']
        stats['class_distribution'] = {}
        
        for class_name in class_names:
            train_count = stats['train']['class_counts'].get(class_name, 0)
            val_count = stats['val']['class_counts'].get(class_name, 0)
            stats['class_distribution'][class_name] = {
                'train': train_count,
                'val': val_count,
                'total': train_count + val_count
            }
        
        return stats
    
    @staticmethod
    def _analyze_split(split_path: Path, class_names: List[str]) -> Dict:
        """Analyze a single split (train/val/test)."""
        images_dir = split_path / "images" if (split_path / "images").exists() else split_path
        labels_dir = split_path / "labels" if (split_path / "labels").exists() else split_path
        
        # Count files
        image_files = set()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.update(f.stem for f in images_dir.glob(ext))
        
        label_files = set(f.stem for f in labels_dir.glob("*.txt"))
        
        # Count class instances
        class_counts = Counter()
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(class_names):
                            class_counts[class_names[class_id]] += 1
        
        return {
            'images': len(image_files),
            'labels': len(label_files),
            'matched_pairs': len(image_files & label_files),
            'orphaned_images': len(image_files - label_files),
            'orphaned_labels': len(label_files - image_files),
            'class_counts': dict(class_counts)
        }


class DatasetAugmenter:
    """Augment datasets to increase training data."""
    
    @staticmethod
    def augment_dataset(dataset_dir: str, output_dir: str, 
                       augmentations_per_image: int = 3) -> str:
        """Apply augmentations to increase dataset size."""
        import albumentations as A
        
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Define augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomRotate90(p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        
        output_images_dir = output_path / "images"
        output_labels_dir = output_path / "labels"
        output_images_dir.mkdir(exist_ok=True)
        output_labels_dir.mkdir(exist_ok=True)
        
        # Process each image
        for image_file in images_dir.glob("*.jpg"):
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            # Load image
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load bounding boxes
            bboxes = []
            class_labels = []
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
            
            # Copy original
            shutil.copy2(image_file, output_images_dir / image_file.name)
            if label_file.exists():
                shutil.copy2(label_file, output_labels_dir / label_file.name)
            
            # Generate augmentations
            for i in range(augmentations_per_image):
                try:
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    
                    # Save augmented image
                    aug_image_name = f"{image_file.stem}_aug_{i}.jpg"
                    aug_image_path = output_images_dir / aug_image_name
                    
                    aug_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_image_path), aug_image)
                    
                    # Save augmented labels
                    aug_label_name = f"{image_file.stem}_aug_{i}.txt"
                    aug_label_path = output_labels_dir / aug_label_name
                    
                    with open(aug_label_path, 'w') as f:
                        for bbox, class_label in zip(augmented['bboxes'], augmented['class_labels']):
                            f.write(f"{class_label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                
                except Exception as e:
                    print(f"Failed to augment {image_file.name}: {e}")
        
        # Copy dataset.yaml if it exists
        yaml_file = dataset_path / "dataset.yaml"
        if yaml_file.exists():
            shutil.copy2(yaml_file, output_path / "dataset.yaml")
        
        return str(output_path)


def main():
    """Command line interface for dataset utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset utilities for YOLO training")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val sets')
    split_parser.add_argument('dataset_dir', help='Path to dataset directory')
    split_parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    split_parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation set ratio')
    split_parser.add_argument('--test-ratio', type=float, default=0.0, help='Test set ratio')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('dataset_yaml', help='Path to dataset YAML file')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert COCO to YOLO format')
    convert_parser.add_argument('coco_json', help='Path to COCO JSON file')
    convert_parser.add_argument('images_dir', help='Path to images directory')
    convert_parser.add_argument('output_dir', help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        splits = DatasetSplitter.split_dataset(
            args.dataset_dir, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
        print("Dataset split completed:")
        for split_name, split_path in splits.items():
            print(f"  {split_name}: {split_path}")
    
    elif args.command == 'validate':
        results = DatasetValidator.validate_yolo_dataset(args.dataset_yaml)
        print(f"Dataset validation: {'PASSED' if results['valid'] else 'FAILED'}")
        
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        if results['statistics']:
            stats = results['statistics']
            print(f"\nStatistics:")
            print(f"  Total images: {stats['total_images']}")
            print(f"  Total labels: {stats['total_labels']}")
            print(f"  Training images: {stats['train']['images']}")
            print(f"  Validation images: {stats['val']['images']}")
    
    elif args.command == 'convert':
        yaml_path = DatasetConverter.coco_to_yolo(
            args.coco_json, args.images_dir, args.output_dir
        )
        print(f"Conversion completed. Dataset config saved to: {yaml_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 