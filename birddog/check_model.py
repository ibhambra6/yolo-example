#!/usr/bin/env python3
"""
check_model.py - Utility to inspect your YOLO model and show available classes
"""
import torch
import sys
import os

def check_yolov5_model(model_path):
    """Check YOLOv5 model classes and info."""
    try:
        print("Loading YOLOv5 model...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
        
        print(f"\n✅ Successfully loaded YOLOv5 model from: {model_path}")
        print(f"📊 Model info:")
        print(f"   - Classes: {len(model.names)}")
        print(f"   - Class names: {model.names}")
        
        # Test inference on a dummy image
        print(f"\n🧪 Testing inference...")
        import cv2
        import numpy as np
        
        # Create a test image (640x640 RGB)
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_img)
        print(f"   - Inference test: ✅ Success")
        print(f"   - Detection format: {type(results)}")
        
        return model.names
        
    except Exception as e:
        print(f"❌ Error loading YOLOv5 model: {e}")
        return None

def check_yolov8_model(model_path):
    """Check YOLOv8 model classes and info."""
    try:
        from ultralytics import YOLO
        print("Loading YOLOv8 model...")
        model = YOLO(model_path)
        
        print(f"\n✅ Successfully loaded YOLOv8 model from: {model_path}")
        print(f"📊 Model info:")
        print(f"   - Classes: {len(model.names)}")
        print(f"   - Class names: {model.names}")
        
        # Test inference on a dummy image
        print(f"\n🧪 Testing inference...")
        import cv2
        import numpy as np
        
        # Create a test image (640x640 RGB)
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_img, verbose=False)
        print(f"   - Inference test: ✅ Success")
        print(f"   - Detection format: {type(results[0])}")
        
        return model.names
        
    except Exception as e:
        print(f"❌ Error loading YOLOv8 model: {e}")
        return None

def main():
    model_path = "best.pt"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Usage: python check_model.py [model_path]")
        print("Default: python check_model.py best.pt")
        return
    
    print(f"🔍 Checking YOLO model: {model_path}")
    print("=" * 50)
    
    # Try YOLOv8 first, then YOLOv5
    print("🔄 Attempting YOLOv8 format...")
    class_names = check_yolov8_model(model_path)
    
    if class_names is None:
        print("\n🔄 Attempting YOLOv5 format...")
        class_names = check_yolov5_model(model_path)
    
    if class_names is not None:
        print(f"\n📝 Configuration for tracking:")
        print(f"   MODEL_PATH = \"{model_path}\"")
        print(f"   TRACK_CLASS options:")
        for i, name in class_names.items():
            print(f"      {i}: {name}")
        print(f"\n💡 Set TRACK_CLASS to the ID of the object you want to track")
        print(f"   For example: TRACK_CLASS = 0  # to track '{class_names.get(0, 'unknown')}'")
    else:
        print(f"\n❌ Failed to load model. Please check:")
        print(f"   - Model file exists and is not corrupted")
        print(f"   - Model was trained with YOLO (YOLOv5 or YOLOv8)")
        print(f"   - Required dependencies are installed")

if __name__ == "__main__":
    main() 