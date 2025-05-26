@echo off
REM ============================================================================
REM YOLO Classifier Environment Activation Script (Windows)
REM ============================================================================
REM This script activates the Python virtual environment and provides
REM helpful commands for using the YOLO classifier applications.
REM
REM Usage: activate.bat
REM ============================================================================

echo.
echo ========================================
echo   YOLO Classifier Environment
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "yolo_env\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo.
    echo Please run setup first:
    echo   python setup.py
    echo.
    pause
    exit /b 1
)

REM Activate the virtual environment
echo 🔄 Activating YOLO Classifier environment...
call yolo_env\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment!
    echo.
    echo Try running setup again:
    echo   python setup.py
    echo.
    pause
    exit /b 1
)

echo ✅ Environment activated successfully!
echo.

REM Display available commands
echo 🎯 Available Applications:
echo.
echo   🐕🐱 Simple Classifier:
echo     python run.py --app dog_cat
echo     python dog_cat_yolo_gui.py
echo.
echo   🎯 Advanced Classifier:
echo     python run.py --app generic
echo     python generic_yolo_classifier.py
echo.
echo   🚀 Training Wizard:
echo     python run.py --app train
echo     python train_model.py --interactive
echo.
echo   ⚙️  Configuration Creator:
echo     python run.py --app config
echo     python create_config.py --interactive
echo.
echo   📋 Interactive Menu:
echo     python run.py
echo.

REM Display helpful information
echo 💡 Helpful Commands:
echo.
echo   Check installation:     python run.py --check
echo   List applications:      python run.py --list
echo   Setup help:            python setup.py --help
echo.

REM Display current environment info
echo 📊 Environment Info:
echo   Virtual Environment: %VIRTUAL_ENV%
echo   Python Version: 
python --version 2>nul || echo   Python not found in PATH
echo   Current Directory: %CD%
echo.

echo 🎉 Ready to use YOLO Classifier!
echo Type 'deactivate' to exit the environment.
echo.

REM Keep the command prompt open with the activated environment
cmd /k 