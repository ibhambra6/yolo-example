#!/usr/bin/env python3
"""
generic_yolo_classifier.py
==========================
A flexible YOLO classifier that can be trained on any dataset and classify any objects.

Features:
1. Loads YOLOv5 model (PyTorch) with configurable weights
2. Supports training on custom datasets with any number of classes
3. Dynamic GUI that adapts to the number of classes
4. Configuration file support for easy customization
5. Batch processing capabilities

Dependencies
------------
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
$ pip install yolov5 pillow pyyaml

Usage
-----
$ python generic_yolo_classifier.py                    # run with default config
$ python generic_yolo_classifier.py --config my_config.yaml  # run with custom config

Training on custom dataset:
$ python -m yolov5 train \
        --img 640 --batch 16 --epochs 50 \
        --data your_dataset.yaml \
        --weights yolov5s.pt
"""
from __future__ import annotations

import sys
import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess

import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# ----------------------------------------------------------------------------
# Configuration Management
# ----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model": {
        "weights": "yolov5s",  # Can be "yolov5s", "yolov5m", "yolov5l", "yolov5x", or path to custom weights
        "confidence_threshold": 0.25,
        "image_size": 640
    },
    "classes": {
        "names": ["cat", "dog"],  # Default classes
        "colors": ["#FF6B6B", "#4ECDC4"]  # Colors for visualization
    },
    "gui": {
        "title": "Generic YOLO Classifier",
        "window_size": "800x900",
        "max_image_size": (600, 500)
    },
    "training": {
        "epochs": 50,
        "batch_size": 16,
        "image_size": 640
    }
}

class ConfigManager:
    """Manages configuration for the classifier."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            self._merge_config(self.config, user_config)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration.")
    
    def _merge_config(self, base: Dict, update: Dict) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

# ----------------------------------------------------------------------------
# Model utilities
# ----------------------------------------------------------------------------
MODEL: torch.nn.Module | None = None
CONFIG: ConfigManager | None = None

def load_model(config: ConfigManager) -> None:
    """Lazy-load YOLOv5 with configurable weights."""
    global MODEL
    if MODEL is not None:
        return

    weights = config.config["model"]["weights"]
    
    # Check if it's a path to custom weights
    if Path(weights).exists():
        print(f"Loading custom weights from: {weights}")
        MODEL = torch.hub.load("ultralytics/yolov5", "custom", path=weights, force_reload=True)
    else:
        print(f"Loading pre-trained weights: {weights}")
        MODEL = torch.hub.load("ultralytics/yolov5", weights, pretrained=True)
    
    MODEL.eval()
    print("Model loaded successfully!")

def train_model(data_yaml: str, config: ConfigManager, output_dir: str = "runs/train") -> None:
    """Train YOLOv5 on a custom dataset."""
    training_config = config.config["training"]
    model_config = config.config["model"]
    
    # Determine the weights to use
    weights = model_config["weights"]
    
    # If using a model name (like "yolov5s"), add .pt extension
    if weights in ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]:
        weights = f"{weights}.pt"
    
    cmd = [
        sys.executable,
        "-m",
        "yolov5.train",
        "--data", data_yaml,
        "--weights", weights,
        "--epochs", str(training_config["epochs"]),
        "--batch-size", str(training_config["batch_size"]),
        "--img", str(training_config["image_size"]),
        "--project", output_dir,
        "--name", "custom_classifier",
        "--cache",  # Cache images for faster training
        "--save-period", "10",  # Save checkpoint every 10 epochs
        "--patience", "20"  # Early stopping patience
    ]
    
    print(f"🚀 Starting YOLOv5 training...")
    print(f"📊 Parameters: {training_config['epochs']} epochs, batch size {training_config['batch_size']}")
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        # Run training with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        # Wait for completion
        process.wait()
        
        if process.returncode == 0:
            print("✅ Training completed successfully!")
            
            # Check for output weights
            weights_path = Path(output_dir) / "custom_classifier" / "weights" / "best.pt"
            if weights_path.exists():
                print(f"📁 Best weights saved: {weights_path}")
                return str(weights_path)
            else:
                print("⚠️ Training completed but best weights not found")
                return True
        else:
            print(f"❌ Training failed with exit code: {process.returncode}")
            raise RuntimeError(f"Training failed with exit code: {process.returncode}")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

def predict(image_path: str | Path, config: ConfigManager) -> Tuple[str, float, List[Dict]]:
    """Return prediction, confidence, and all detections for the supplied image."""
    load_model(config)
    
    image_size = config.config["model"]["image_size"]
    conf_threshold = config.config["model"]["confidence_threshold"]
    class_names = config.config["classes"]["names"]
    
    results = MODEL(str(image_path), size=image_size)
    detections = results.xyxy[0].tolist()  # List of [x1,y1,x2,y2,conf,cls]
    
    # Process detections
    all_detections = []
    class_confidences = {}
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        cls = int(cls)
        
        if conf < conf_threshold:
            continue
            
        # Map class index to name
        if cls < len(class_names):
            class_name = class_names[cls]
        else:
            class_name = f"Class_{cls}"
        
        all_detections.append({
            "class": class_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })
        
        # Track highest confidence for each class
        if class_name not in class_confidences or conf > class_confidences[class_name]:
            class_confidences[class_name] = conf
    
    if not class_confidences:
        return "Unknown", 0.0, all_detections
    
    # Return the class with highest confidence
    best_class = max(class_confidences.items(), key=lambda x: x[1])
    return best_class[0], best_class[1], all_detections

# ----------------------------------------------------------------------------
# Dataset Management
# ----------------------------------------------------------------------------

class DatasetManager:
    """Manages dataset creation and validation."""
    
    @staticmethod
    def create_dataset_yaml(dataset_path: str, class_names: List[str], 
                          train_path: str = "train", val_path: str = "val") -> str:
        """Create a YAML configuration file for the dataset."""
        dataset_config = {
            "path": dataset_path,
            "train": train_path,
            "val": val_path,
            "nc": len(class_names),
            "names": class_names
        }
        
        yaml_path = Path(dataset_path) / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return str(yaml_path)
    
    @staticmethod
    def fix_roboflow_dataset_yaml(yaml_path: str) -> str:
        """
        Fix Roboflow dataset YAML to be compatible with YOLOv5 training.
        
        Roboflow datasets often have relative paths and missing 'path' key.
        This function converts them to the expected format.
        """
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get the directory containing the YAML file
            yaml_dir = Path(yaml_path).parent.resolve()
            
            # Add missing 'path' key if not present
            if 'path' not in config:
                config['path'] = str(yaml_dir)
                print(f"✅ Added missing 'path' key: {yaml_dir}")
            
            # Convert relative paths for Roboflow datasets
            for key in ['train', 'val', 'test']:
                if key in config and isinstance(config[key], str):
                    original_path = config[key]
                    
                    if original_path.startswith('../'):
                        # For Roboflow datasets, the paths are relative to the parent directory
                        # but the actual data is usually in the same directory as the YAML
                        
                        # Extract the folder name (e.g., 'train/images' from '../train/images')
                        folder_name = original_path.replace('../', '')
                        
                        # Check if the folder exists in the same directory as the YAML
                        local_path = yaml_dir / folder_name
                        if local_path.exists():
                            config[key] = folder_name
                            print(f"✅ Fixed {key} path: {folder_name} (found locally)")
                        else:
                            # Try to resolve the original relative path
                            abs_path = (yaml_dir / original_path).resolve()
                            if abs_path.exists():
                                try:
                                    rel_path = abs_path.relative_to(Path(config['path']))
                                    config[key] = str(rel_path)
                                    print(f"✅ Fixed {key} path: {rel_path}")
                                except ValueError:
                                    config[key] = str(abs_path)
                                    print(f"✅ Using absolute {key} path: {abs_path}")
                            else:
                                print(f"⚠️ Warning: {key} path not found: {original_path}")
                    else:
                        # Path doesn't start with '../', check if it exists
                        check_path = yaml_dir / original_path
                        if check_path.exists():
                            print(f"✅ {key} path verified: {original_path}")
                        else:
                            print(f"⚠️ Warning: {key} path not found: {original_path}")
            
            # Create a fixed version of the YAML file
            fixed_yaml_path = yaml_dir / "dataset_fixed.yaml"
            with open(fixed_yaml_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            print(f"✅ Created fixed dataset YAML: {fixed_yaml_path}")
            return str(fixed_yaml_path)
            
        except Exception as e:
            print(f"❌ Error fixing Roboflow dataset YAML: {e}")
            raise
    
    @staticmethod
    def validate_dataset(dataset_yaml: str) -> bool:
        """Validate that the dataset structure is correct."""
        try:
            if not os.path.exists(dataset_yaml):
                print(f"Dataset YAML file does not exist: {dataset_yaml}")
                return False
            
            with open(dataset_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                print(f"Dataset YAML file is empty or invalid: {dataset_yaml}")
                return False
            
            print(f"🔍 Validating dataset config: {dataset_yaml}")
            print(f"📋 Config keys found: {list(config.keys())}")
            
            # Check if this is a Roboflow dataset that needs fixing
            needs_fixing = 'path' not in config or (
                'path' in config and any(
                    isinstance(config.get(key), str) and config[key].startswith('../')
                    for key in ['train', 'val', 'test']
                    if key in config
                )
            )
            
            if needs_fixing and not dataset_yaml.endswith('_fixed.yaml'):
                print(f"🤖 Detected Roboflow dataset format that needs fixing")
                print(f"🔧 Attempting to fix dataset configuration...")
                
                try:
                    fixed_yaml_path = DatasetManager.fix_roboflow_dataset_yaml(dataset_yaml)
                    print(f"✅ Dataset fixed! Using: {fixed_yaml_path}")
                    
                    # Recursively validate the fixed dataset
                    return DatasetManager.validate_dataset(fixed_yaml_path)
                    
                except Exception as e:
                    print(f"❌ Failed to fix Roboflow dataset: {e}")
                    return False
            
            required_keys = ["path", "train", "val", "nc", "names"]
            missing_keys = []
            
            for key in required_keys:
                if key not in config:
                    missing_keys.append(key)
            
            if missing_keys:
                print(f"❌ Missing required keys in dataset config: {missing_keys}")
                print(f"📄 Current config content:")
                for key, value in config.items():
                    print(f"   {key}: {value}")
                return False
            
            dataset_path = Path(config["path"])
            train_path = dataset_path / config["train"]
            val_path = dataset_path / config["val"]
            
            print(f"📁 Dataset path: {dataset_path}")
            print(f"🚂 Train path: {train_path}")
            print(f"✅ Val path: {val_path}")
            
            if not train_path.exists():
                print(f"❌ Training path does not exist: {train_path}")
                return False
            
            if not val_path.exists():
                print(f"❌ Validation path does not exist: {val_path}")
                return False
            
            print(f"✅ Dataset validation passed!")
            return True
            
        except Exception as e:
            print(f"❌ Error validating dataset: {e}")
            import traceback
            traceback.print_exc()
            return False

# ----------------------------------------------------------------------------
# Enhanced GUI
# ----------------------------------------------------------------------------

class GenericClassifierApp(tk.Tk):
    def __init__(self, config: ConfigManager) -> None:
        super().__init__()
        self.config = config
        self.setup_gui()
        self.current_image_path: Optional[str] = None
        self.current_detections: List[Dict] = []
        
    def setup_gui(self) -> None:
        """Set up the GUI based on configuration."""
        gui_config = self.config.config["gui"]
        
        self.title(gui_config["title"])
        self.geometry(gui_config["window_size"])
        self.resizable(True, True)
        self.configure(padx=12, pady=12)
        
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        self.create_control_panel(main_frame)
        
        # Image display area
        self.create_image_area(main_frame)
        
        # Results area
        self.create_results_area(main_frame)
        
        # Training panel
        self.create_training_panel(main_frame)
    
    def create_control_panel(self, parent: ttk.Frame) -> None:
        """Create the control panel with buttons."""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Upload button
        self.upload_btn = ttk.Button(
            control_frame, text="Upload Image", command=self.on_upload
        )
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Batch process button
        self.batch_btn = ttk.Button(
            control_frame, text="Batch Process", command=self.on_batch_process
        )
        self.batch_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Config button
        self.config_btn = ttk.Button(
            control_frame, text="Load Config", command=self.on_load_config
        )
        self.config_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Save config button
        self.save_config_btn = ttk.Button(
            control_frame, text="Save Config", command=self.on_save_config
        )
        self.save_config_btn.pack(side=tk.LEFT)
    
    def create_image_area(self, parent: ttk.Frame) -> None:
        """Create the image display area."""
        image_frame = ttk.LabelFrame(parent, text="Image", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.img_lbl = ttk.Label(image_frame, text="No image loaded")
        self.img_lbl.pack(expand=True)
    
    def create_results_area(self, parent: ttk.Frame) -> None:
        """Create the results display area."""
        results_frame = ttk.LabelFrame(parent, text="Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Main prediction
        self.pred_lbl = ttk.Label(results_frame, text="Prediction: —", font=("Arial", 14, "bold"))
        self.pred_lbl.pack(anchor=tk.W)
        
        # Confidence
        self.conf_lbl = ttk.Label(results_frame, text="Confidence: —")
        self.conf_lbl.pack(anchor=tk.W)
        
        # All detections
        self.detections_frame = ttk.Frame(results_frame)
        self.detections_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(self.detections_frame, text="All Detections:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        # Scrollable text for detections
        self.detections_text = tk.Text(self.detections_frame, height=6, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(self.detections_frame, orient=tk.VERTICAL, command=self.detections_text.yview)
        self.detections_text.configure(yscrollcommand=scrollbar.set)
        
        self.detections_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_training_panel(self, parent: ttk.Frame) -> None:
        """Create the training control panel."""
        training_frame = ttk.LabelFrame(parent, text="Training", padding=10)
        training_frame.pack(fill=tk.X)
        
        # Dataset selection
        dataset_frame = ttk.Frame(training_frame)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(dataset_frame, text="Dataset YAML:").pack(side=tk.LEFT)
        self.dataset_var = tk.StringVar()
        self.dataset_entry = ttk.Entry(dataset_frame, textvariable=self.dataset_var, width=40)
        self.dataset_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        ttk.Button(dataset_frame, text="Browse", command=self.on_browse_dataset).pack(side=tk.LEFT)
        
        # Training controls
        controls_frame = ttk.Frame(training_frame)
        controls_frame.pack(fill=tk.X)
        
        ttk.Button(controls_frame, text="Start Training", command=self.on_start_training).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(controls_frame, text="Create Dataset Config", command=self.on_create_dataset_config).pack(side=tk.LEFT)
    
    def on_upload(self) -> None:
        """Handle image upload."""
        fpath = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if not fpath:
            return
        
        self.current_image_path = fpath
        self.process_image(fpath)
    
    def on_batch_process(self) -> None:
        """Handle batch processing of images."""
        folder_path = filedialog.askdirectory(title="Select folder with images")
        if not folder_path:
            return
        
        # Create results window
        self.show_batch_results(folder_path)
    
    def show_batch_results(self, folder_path: str) -> None:
        """Show batch processing results in a new window."""
        batch_window = tk.Toplevel(self)
        batch_window.title("Batch Processing Results")
        batch_window.geometry("600x400")
        
        # Create treeview for results
        tree = ttk.Treeview(batch_window, columns=("File", "Prediction", "Confidence"), show="headings")
        tree.heading("File", text="File")
        tree.heading("Prediction", text="Prediction")
        tree.heading("Confidence", text="Confidence")
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Process images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        folder = Path(folder_path)
        
        for image_path in folder.iterdir():
            if image_path.suffix.lower() in image_extensions:
                try:
                    prediction, confidence, _ = predict(str(image_path), self.config)
                    tree.insert("", tk.END, values=(image_path.name, prediction, f"{confidence:.3f}"))
                except Exception as e:
                    tree.insert("", tk.END, values=(image_path.name, "Error", str(e)))
        
        # Add export button
        ttk.Button(batch_window, text="Export Results", 
                  command=lambda: self.export_batch_results(tree)).pack(pady=10)
    
    def export_batch_results(self, tree: ttk.Treeview) -> None:
        """Export batch results to CSV."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return
        
        import csv
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File", "Prediction", "Confidence"])
            
            for item in tree.get_children():
                values = tree.item(item)["values"]
                writer.writerow(values)
        
        messagebox.showinfo("Export Complete", f"Results exported to {file_path}")
    
    def process_image(self, image_path: str) -> None:
        """Process and display an image."""
        try:
            # Load and display image
            pil_img = Image.open(image_path).convert("RGB")
            self.display_image(pil_img)
            
            # Get prediction
            prediction, confidence, detections = predict(image_path, self.config)
            self.current_detections = detections
            
            # Update results
            self.pred_lbl.configure(text=f"Prediction: {prediction}")
            self.conf_lbl.configure(text=f"Confidence: {confidence:.3f}")
            
            # Update detections display
            self.update_detections_display(detections)
            
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to process image:\n{exc}")
    
    def display_image(self, pil_img: Image.Image) -> None:
        """Display image in the GUI."""
        max_size = self.config.config["gui"]["max_image_size"]
        
        # Resize image to fit display
        w, h = pil_img.size
        scale = min(max_size[0] / w, max_size[1] / h, 1)
        disp_img = pil_img.resize((int(w * scale), int(h * scale)))
        
        tk_img = ImageTk.PhotoImage(disp_img)
        self.img_lbl.configure(image=tk_img, text="")
        self.img_lbl.image = tk_img  # Keep reference
    
    def update_detections_display(self, detections: List[Dict]) -> None:
        """Update the detections text display."""
        self.detections_text.delete(1.0, tk.END)
        
        if not detections:
            self.detections_text.insert(tk.END, "No objects detected above confidence threshold.")
            return
        
        for i, detection in enumerate(detections, 1):
            text = f"{i}. {detection['class']} (confidence: {detection['confidence']:.3f})\n"
            self.detections_text.insert(tk.END, text)
    
    def on_load_config(self) -> None:
        """Load configuration from file."""
        config_path = filedialog.askopenfilename(
            title="Select configuration file",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if config_path:
            self.config.load_config(config_path)
            messagebox.showinfo("Config Loaded", f"Configuration loaded from {config_path}")
            # Reset model to reload with new config
            global MODEL
            MODEL = None
    
    def on_save_config(self) -> None:
        """Save current configuration to file."""
        config_path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if config_path:
            self.config.save_config(config_path)
            messagebox.showinfo("Config Saved", f"Configuration saved to {config_path}")
    
    def on_browse_dataset(self) -> None:
        """Browse for dataset YAML file."""
        dataset_path = filedialog.askopenfilename(
            title="Select dataset YAML file",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if dataset_path:
            self.dataset_var.set(dataset_path)
    
    def on_start_training(self) -> None:
        """Start model training."""
        dataset_yaml = self.dataset_var.get()
        if not dataset_yaml:
            messagebox.showerror("Error", "Please select a dataset YAML file.")
            return
        
        # Check if dataset is valid and get the correct YAML path
        try:
            with open(dataset_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            # If it's a Roboflow dataset, fix it first
            if 'roboflow' in config or 'path' not in config:
                print("🤖 Detected Roboflow dataset, fixing configuration...")
                fixed_yaml = DatasetManager.fix_roboflow_dataset_yaml(dataset_yaml)
                dataset_yaml = fixed_yaml  # Use the fixed version for training
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read dataset configuration:\n{e}")
            return
        
        if not DatasetManager.validate_dataset(dataset_yaml):
            messagebox.showerror("Error", "Invalid dataset configuration.")
            return
        
        # Confirm training
        result = messagebox.askyesno(
            "Start Training",
            f"Start training on dataset: {dataset_yaml}?\n\n"
            f"This may take a long time depending on your dataset size and hardware."
        )
        
        if result:
            try:
                train_model(dataset_yaml, self.config)
                messagebox.showinfo("Training Complete", "Model training completed successfully!")
            except Exception as e:
                messagebox.showerror("Training Error", f"Training failed:\n{e}")
    
    def on_create_dataset_config(self) -> None:
        """Create a new dataset configuration."""
        DatasetConfigDialog(self, self.config)

class DatasetConfigDialog:
    """Dialog for creating dataset configurations."""
    
    def __init__(self, parent: tk.Tk, config: ConfigManager):
        self.parent = parent
        self.config = config
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Create Dataset Configuration")
        self.dialog.geometry("500x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_dialog()
    
    def setup_dialog(self) -> None:
        """Set up the dataset configuration dialog."""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dataset path
        ttk.Label(main_frame, text="Dataset Root Path:").pack(anchor=tk.W)
        self.dataset_path_var = tk.StringVar()
        path_frame = ttk.Frame(main_frame)
        path_frame.pack(fill=tk.X, pady=(5, 15))
        
        ttk.Entry(path_frame, textvariable=self.dataset_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="Browse", command=self.browse_dataset_path).pack(side=tk.LEFT, padx=(5, 0))
        
        # Class names
        ttk.Label(main_frame, text="Class Names (one per line):").pack(anchor=tk.W)
        self.classes_text = tk.Text(main_frame, height=8, width=40)
        self.classes_text.pack(fill=tk.BOTH, expand=True, pady=(5, 15))
        
        # Pre-fill with current classes
        current_classes = self.config.config["classes"]["names"]
        self.classes_text.insert(tk.END, "\n".join(current_classes))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Create Config", command=self.create_config).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT)
    
    def browse_dataset_path(self) -> None:
        """Browse for dataset root directory."""
        path = filedialog.askdirectory(title="Select dataset root directory")
        if path:
            self.dataset_path_var.set(path)
    
    def create_config(self) -> None:
        """Create the dataset configuration file."""
        dataset_path = self.dataset_path_var.get()
        if not dataset_path:
            messagebox.showerror("Error", "Please select a dataset path.")
            return
        
        classes_text = self.classes_text.get(1.0, tk.END).strip()
        if not classes_text:
            messagebox.showerror("Error", "Please enter class names.")
            return
        
        class_names = [line.strip() for line in classes_text.split('\n') if line.strip()]
        
        try:
            yaml_path = DatasetManager.create_dataset_yaml(dataset_path, class_names)
            messagebox.showinfo("Success", f"Dataset configuration created at:\n{yaml_path}")
            self.dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dataset configuration:\n{e}")

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generic YOLO Classifier")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Create and run application
    app = GenericClassifierApp(config)
    
    # Load model in background
    try:
        load_model(config)
    except Exception as e:
        messagebox.showerror("Model Loading Error", f"Failed to load model:\n{e}")
    
    app.mainloop()

if __name__ == "__main__":
    main() 