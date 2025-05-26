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
License: MIT
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

# GUI configuration
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 720
MAX_DISPLAY_WIDTH = 600
MAX_DISPLAY_HEIGHT = 500

# ============================================================================
# Model Management Functions
# ============================================================================


def load_model() -> None:
    """
    Lazy-load the YOLOv5 model for dog/cat classification.
    
    This function loads the YOLOv5 model on first use. It first checks for a custom
    fine-tuned model (dog_cat.pt) in the script directory, and falls back to the
    pre-trained COCO weights if not found.
    
    Model Loading Priority:
        1. Custom dog_cat.pt weights (if available)
        2. Pre-trained YOLOv5s COCO weights
        3. Models directory (models/yolov5s.pt)
    
    Note:
        - Model is cached globally to avoid reloading
        - Model is set to evaluation mode for inference
        - Uses PyTorch Hub for automatic model downloading
    
    Raises:
        RuntimeError: If model loading fails
        ConnectionError: If unable to download pre-trained weights
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
    
    # Determine which weights to use
    if custom_weights.exists():
        weights = str(custom_weights)
        print(f"📦 Using custom weights: {weights}")
    elif models_weights.exists():
        weights = str(models_weights)
        print(f"📦 Using local weights: {weights}")
    else:
        weights = "yolov5s"
        print("📦 Using pre-trained COCO weights (will download if needed)")
    
    try:
        # Load model from PyTorch Hub
        MODEL = torch.hub.load("ultralytics/yolov5", "custom", path=weights if custom_weights.exists() or models_weights.exists() else "yolov5s", pretrained=True)
        MODEL.eval()  # Set to evaluation mode
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Could not load YOLOv5 model: {e}")


def fine_tune(data_yaml: str, epochs: int = 50, weights: str = "yolov5s.pt") -> None:
    """Fine-tune YOLOv5 on a two-class dog/cat dataset.

    The *data_yaml* file should follow the YOLOv5 dataset schema and list the
    two class names in the order `["cat", "dog"]`.
    """
    cmd = [
        sys.executable,
        "-m",
        "yolov5.train",
        "--img",
        "640",
        "--batch",
        "16",
        "--epochs",
        str(epochs),
        "--data",
        data_yaml,
        "--weights",
        weights,
    ]
    subprocess.run(cmd, check=True)


def predict(image_path: str | Path) -> str:
    """
    Classify an image as containing a Dog, Cat, or Unknown.
    
    This function processes an image through the YOLOv5 model and determines
    whether it contains a dog or cat based on detection confidence scores.
    
    Args:
        image_path (str | Path): Path to the image file to classify
    
    Returns:
        str: Classification result - "Dog", "Cat", or "Unknown"
    
    Algorithm:
        1. Load the model if not already loaded
        2. Run inference on the image
        3. Extract detection results (bounding boxes, confidence, class)
        4. Find highest confidence dog and cat detections
        5. Return the class with higher confidence, or "Unknown" if neither
    
    Note:
        - Uses COCO class IDs: 15 for cat, 16 for dog
        - Only considers detections above CONF_THRESHOLD
        - Returns "Unknown" if no animals detected with sufficient confidence
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
        return "Unknown"
    
    # Return the class with higher confidence
    return "Dog" if dog_confidence >= cat_confidence else "Cat"


# ============================================================================
# GUI Application Class
# ============================================================================

class DogCatClassifierApp(tk.Tk):
    """
    Main GUI application for the Dog vs Cat Classifier.
    
    This class creates a simple Tkinter interface that allows users to upload
    images and get real-time classification results. The interface includes
    an upload button, image display area, and prediction label.
    
    Features:
        - File dialog for image selection
        - Automatic image resizing for display
        - Real-time classification feedback
        - Error handling for invalid images
        - Clean, user-friendly interface
    """
    
    def __init__(self) -> None:
        """
        Initialize the GUI application.
        
        Sets up the main window, creates all UI elements, and configures
        the layout. The window is fixed-size to ensure consistent appearance.
        """
        super().__init__()
        
        # Configure main window
        self.title("Dog vs Cat Classifier (YOLOv5)")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.resizable(False, False)  # Fixed size for consistent layout
        self.configure(padx=12, pady=12, bg='#f0f0f0')
        
        # Create and configure UI elements
        self._create_widgets()
        
        # Center the window on screen
        self._center_window()
    
    def _create_widgets(self) -> None:
        """Create and layout all GUI widgets."""
        
        # Upload button - main interaction element
        self.upload_btn = tk.Button(
            self,
            text="📁 Upload Image",
            font=("Arial", 12, "bold"),
            command=self.on_upload,
            bg='#4CAF50',
            fg='white',
            relief='raised',
            padx=20,
            pady=10
        )
        self.upload_btn.pack(pady=15)
        
        # Image display label - shows uploaded image
        self.img_lbl = tk.Label(
            self,
            text="No image selected\n\nClick 'Upload Image' to get started",
            font=("Arial", 10),
            fg='#666666',
            bg='#f0f0f0',
            width=50,
            height=20,
            relief='sunken',
            bd=1
        )
        self.img_lbl.pack(pady=10)
        
        # Prediction label - shows classification result
        self.pred_lbl = tk.Label(
            self,
            text="Prediction: —",
            font=("Arial", 16, "bold"),
            fg='#333333',
            bg='#f0f0f0'
        )
        self.pred_lbl.pack(pady=15)
        
        # Instructions label
        instructions = tk.Label(
            self,
            text="Supported formats: JPG, JPEG, PNG, BMP",
            font=("Arial", 9),
            fg='#888888',
            bg='#f0f0f0'
        )
        instructions.pack(pady=5)
    
    def _center_window(self) -> None:
        """Center the window on the screen."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def on_upload(self) -> None:
        fpath = filedialog.askopenfilename(
            title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not fpath:
            return

        try:
            pil_img = Image.open(fpath).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to open image:\n{exc}")
            return

        # Display (resize to fit)
        w, h = pil_img.size
        scale = min(600 / w, 500 / h, 1)
        disp_img = pil_img.resize((int(w * scale), int(h * scale)))
        tk_img = ImageTk.PhotoImage(disp_img)
        self.img_lbl.configure(image=tk_img)
        self.img_lbl.image = tk_img  # keep reference

        prediction = predict(fpath)
        self.pred_lbl.configure(text=f"Prediction: {prediction}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for the Dog vs Cat Classifier application.
    
    This function initializes the model and starts the GUI application.
    The model is loaded in advance to provide faster first-time classification.
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