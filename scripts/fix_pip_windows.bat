@echo off
echo ========================================
echo YOLO Project - Windows Pip Fix Script
echo ========================================
echo.

echo This script will help fix pip installation issues on Windows.
echo.

:: Check if virtual environment exists
if not exist "yolo_env\Scripts\activate.bat" (
    echo ERROR: Virtual environment 'yolo_env' not found!
    echo Please run 'python setup.py' first to create the virtual environment.
    pause
    exit /b 1
)

echo Activating virtual environment...
call yolo_env\Scripts\activate.bat

echo.
echo Step 1: Upgrading pip using Python module method...
python -m pip install --upgrade pip

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Pip upgrade failed. Trying alternative method...
    python -m pip install --upgrade --force-reinstall pip
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo WARNING: Pip upgrade failed. Continuing with existing pip version...
    ) else (
        echo SUCCESS: Pip upgraded successfully!
    )
) else (
    echo SUCCESS: Pip upgraded successfully!
)

echo.
echo Step 2: Installing project requirements...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to install requirements!
    echo You may need to install packages individually:
    echo   pip install torch torchvision
    echo   pip install yolov5
    echo   pip install Pillow opencv-python
    echo   pip install numpy pandas PyYAML
    pause
    exit /b 1
) else (
    echo SUCCESS: All requirements installed successfully!
)

echo.
echo Step 3: Testing installation...
python -c "import torch; import yolov5; import PIL; print('SUCCESS: All packages imported correctly!')"

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some packages may not have installed correctly.
) else (
    echo SUCCESS: Installation validated!
)

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo You can now run the YOLO applications:
echo   python dog_cat_yolo_gui.py
echo   python generic_yolo_classifier.py
echo   python train_model.py --interactive
echo.
echo To activate this environment later, run:
echo   yolo_env\Scripts\activate.bat
echo.
pause 