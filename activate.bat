@echo off
echo Activating YOLO Classifier environment...
call yolo_env\Scripts\activate.bat
echo Environment activated! You can now run:
echo   python dog_cat_yolo_gui.py
echo   python generic_yolo_classifier.py
echo   python train_model.py --interactive
cmd /k
