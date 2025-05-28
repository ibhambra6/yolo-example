#!/bin/bash
echo "Activating YOLO Classifier environment..."
source yolo_env/bin/activate
echo "Environment activated! You can now run:"
echo "  python dog_cat_yolo_gui.py"
echo "  python generic_yolo_classifier.py"
echo "  python train_model.py --interactive"
exec "$SHELL"
