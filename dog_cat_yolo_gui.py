#!/usr/bin/env python3
"""
Simple Dog vs Cat Classifier using YOLOv5
=========================================

A beginner-friendly GUI application for classifying images as containing dogs or cats.
This application uses YOLOv5 with pre-trained COCO weights and provides a simple
Tkinter interface for image upload and classification.

Features:
    - Simple drag-and-drop interface
    - Real-time classification
    - Support for common image formats
    - Automatic model loading
    - Custom model support (dog_cat.pt)

Usage:
    python dog_cat_yolo_gui.py

Requirements:
    - torch
    - torchvision
    - yolov5
    - tkinter
    - Pillow

Author: YOLO Classifier Project
License: Proprietary - Oceaneering International Inc. Use Only
"""

from __future__ import annotations

# Standard library imports
import sys
import subprocess
from pathlib import Path

# Third-party imports
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ============================================================================
# Configuration and Constants
# ============================================================================

# Global model instance (lazy-loaded)
MODEL: torch.nn.Module | None = None

# Classification confidence threshold
CONF_THRESHOLD = 0.25

# COCO dataset class IDs for animals
COCO_DOG_CLASS_ID = 16  # Dog class in COCO dataset
COCO_CAT_CLASS_ID = 15  # Cat class in COCO dataset

# GUI configuration - simplified and smaller
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 600
MAX_DISPLAY_WIDTH = 400
MAX_DISPLAY_HEIGHT = 300

# ============================================================================
# Model Management Functions
# ============================================================================

def load_model() -> None:
    """
    Load the YOLOv5 model for dog/cat classification.
    
    This function loads the YOLOv5 model on first use. It first checks for a custom
    fine-tuned model (dog_cat.pt) in the script directory, and falls back to the
    pre-trained COCO weights if not found.
    
    Model Loading Priority:
        1. Custom dog_cat.pt weights (if available)
        2. Models directory (models/yolov5s.pt)
        3. Pre-trained YOLOv5s COCO weights (downloaded automatically)
    
    Note:
        - Model is cached globally to avoid reloading
        - Model is set to evaluation mode for inference
        - Uses PyTorch Hub for automatic model downloading
    
    Raises:
        RuntimeError: If model loading fails
    """
    global MODEL
    
    # Return early if model is already loaded
    if MODEL is not None:
        return
    
    print("🔄 Loading YOLOv5 model...")
    
    # Check for custom fine-tuned weights first
    script_dir = Path(__file__).resolve().parent
    custom_weights = script_dir / "dog_cat.pt"
    
    # Check models directory for pre-trained weights
    models_weights = script_dir / "models" / "yolov5s.pt"
    
    try:
        # Determine which weights to use and load accordingly
        if custom_weights.exists():
            print(f"📦 Using custom weights: {custom_weights}")
            MODEL = torch.hub.load("ultralytics/yolov5", "custom", path=str(custom_weights))
        elif models_weights.exists():
            print(f"📦 Using local weights: {models_weights}")
            MODEL = torch.hub.load("ultralytics/yolov5", "custom", path=str(models_weights))
        else:
            print("📦 Using pre-trained COCO weights (will download if needed)")
            MODEL = torch.hub.load("ultralytics/yolov5", "yolov5s")
        
        MODEL.eval()  # Set to evaluation mode
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Could not load YOLOv5 model: {e}")

def predict(image_path: str | Path) -> tuple[str, float]:
    """
    Classify an image as containing a Dog, Cat, or Unknown.
    
    Args:
        image_path (str | Path): Path to the image file to classify
    
    Returns:
        tuple[str, float]: Classification result and confidence score
    """
    # Ensure model is loaded
    load_model()
    
    # Run inference on the image
    results = MODEL(str(image_path), size=640)
    
    # Extract detection results: [x1, y1, x2, y2, confidence, class_id]
    detections = results.xyxy[0]  # (N, 6) tensor format
    
    # Track highest confidence for each animal type
    dog_confidence = 0.0
    cat_confidence = 0.0
    
    # Process each detection
    for detection in detections.tolist():
        # Unpack detection: [x1, y1, x2, y2, confidence, class_id]
        *bbox, confidence, class_id = detection
        
        # Skip low-confidence detections
        if confidence < CONF_THRESHOLD:
            continue
        
        class_id = int(class_id)
        
        # Check for dog detection (COCO class 16)
        if class_id == COCO_DOG_CLASS_ID:
            dog_confidence = max(dog_confidence, confidence)
            
        # Check for cat detection (COCO class 15)
        elif class_id == COCO_CAT_CLASS_ID:
            cat_confidence = max(cat_confidence, confidence)
    
    # Determine final classification
    if dog_confidence == 0 and cat_confidence == 0:
        return "Unknown", 0.0
    
    # Return the class with higher confidence
    if dog_confidence >= cat_confidence:
        return "Dog", dog_confidence
    else:
        return "Cat", cat_confidence

# ============================================================================
# Simplified GUI Application Class
# ============================================================================

class DogCatClassifierApp(tk.Tk):
    """
    Simplified GUI application for the Dog vs Cat Classifier.
    
    Features:
        - Clean, minimal interface
        - Large upload button
        - Clear image display
        - Simple prediction results
    """
    
    def __init__(self) -> None:
        """Initialize the simplified GUI application."""
        super().__init__()
        
        # Configure main window
        self.title("🐕🐱 Dog vs Cat Classifier")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.resizable(False, False)
        self.configure(bg='white', padx=20, pady=20)
        
        # Create UI elements
        self._create_widgets()
        
        # Center the window on screen
        self._center_window()
    
    def _create_widgets(self) -> None:
        """Create simplified GUI widgets."""
        
        # Title
        title_lbl = tk.Label(
            self,
            text="🐕🐱 Dog vs Cat Classifier",
            font=("Arial", 18, "bold"),
            fg='#2c3e50',
            bg='white'
        )
        title_lbl.pack(pady=(0, 20))
        
        # Upload button - large and prominent
        self.upload_btn = tk.Button(
            self,
            text="📁 Choose Image",
            font=("Arial", 14, "bold"),
            command=self.on_upload,
            bg='#3498db',
            fg='white',
            relief='flat',
            padx=30,
            pady=15,
            cursor='hand2'
        )
        self.upload_btn.pack(pady=10)
        
        # Image display area - cleaner design
        self.img_frame = tk.Frame(self, bg='#ecf0f1', relief='solid', bd=1)
        self.img_frame.pack(pady=20, fill='both', expand=True)
        
        self.img_lbl = tk.Label(
            self.img_frame,
            text="No image selected\n\n📸 Click 'Choose Image' to start",
            font=("Arial", 11),
            fg='#7f8c8d',
            bg='#ecf0f1',
            justify='center'
        )
        self.img_lbl.pack(expand=True)
        
        # Results area - simplified
        results_frame = tk.Frame(self, bg='white')
        results_frame.pack(pady=(10, 0), fill='x')
        
        # Prediction result
        self.pred_lbl = tk.Label(
            results_frame,
            text="Prediction: —",
            font=("Arial", 16, "bold"),
            fg='#2c3e50',
            bg='white'
        )
        self.pred_lbl.pack()
        
        # Confidence score
        self.conf_lbl = tk.Label(
            results_frame,
            text="",
            font=("Arial", 12),
            fg='#7f8c8d',
            bg='white'
        )
        self.conf_lbl.pack(pady=(5, 0))
        
        # Instructions
        instructions = tk.Label(
            self,
            text="Supports: JPG, PNG, BMP, TIFF",
            font=("Arial", 9),
            fg='#95a5a6',
            bg='white'
        )
        instructions.pack(pady=(10, 0))
    
    def _center_window(self) -> None:
        """Center the window on the screen."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def on_upload(self) -> None:
        """Handle image upload and classification."""
        # File dialog for image selection
        fpath = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not fpath:
            return

        try:
            # Load and display image
            pil_img = Image.open(fpath).convert("RGB")
            self.display_image(pil_img)
            
            # Get prediction
            prediction, confidence = predict(fpath)
            
            # Update results with color coding
            if prediction == "Dog":
                color = "#e74c3c"  # Red for dog
                emoji = "🐕"
            elif prediction == "Cat":
                color = "#9b59b6"  # Purple for cat
                emoji = "🐱"
            else:
                color = "#95a5a6"  # Gray for unknown
                emoji = "❓"
            
            self.pred_lbl.configure(
                text=f"{emoji} {prediction}",
                fg=color
            )
            
            if confidence > 0:
                self.conf_lbl.configure(
                    text=f"Confidence: {confidence:.1%}",
                    fg='#27ae60' if confidence > 0.5 else '#f39c12'
                )
            else:
                self.conf_lbl.configure(text="No animals detected", fg='#95a5a6')
                
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to process image:\n{exc}")
    
    def display_image(self, pil_img: Image.Image) -> None:
        """Display image in the GUI with proper scaling."""
        # Calculate scaling to fit display area
        w, h = pil_img.size
        scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1)
        
        # Resize image
        new_w, new_h = int(w * scale), int(h * scale)
        disp_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to Tkinter format
        tk_img = ImageTk.PhotoImage(disp_img)
        
        # Update display
        self.img_lbl.configure(image=tk_img, text="")
        self.img_lbl.image = tk_img  # Keep reference to prevent garbage collection

# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    """
    Main entry point for the Dog vs Cat Classifier application.
    """
    print("🐕🐱 Starting Dog vs Cat Classifier...")
    
    # Pre-load the model for faster first classification
    try:
        load_model()
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-load model: {e}")
        print("Model will be loaded on first classification attempt.")
    
    # Create and run the GUI application
    app = DogCatClassifierApp()
    print("✅ Application started successfully!")
    app.mainloop()

if __name__ == "__main__":
    main() 