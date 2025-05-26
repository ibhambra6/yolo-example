#!/usr/bin/env python3
"""
Cross-Platform Setup Script for YOLO Classifier Project
=======================================================

This script provides automated setup for the YOLO classifier project on
Windows, Linux, and macOS. It handles virtual environment creation,
dependency installation, and initial configuration.

Features:
    - Cross-platform compatibility (Windows, Linux, macOS)
    - Virtual environment management
    - Dependency installation with error handling
    - Model weight downloading
    - Configuration validation
    - Interactive setup wizard

Usage:
    python setup.py                    # Interactive setup
    python setup.py --quick           # Quick setup with defaults
    python setup.py --dev             # Development setup with extra tools
    python setup.py --check           # Check existing installation

Author: YOLO Classifier Project
License: MIT
"""

import sys
import subprocess
import platform
import argparse
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

# Configuration constants
PYTHON_MIN_VERSION = (3, 8)
RECOMMENDED_PYTHON_VERSION = (3, 9)
VENV_NAME = "yolo_env"
REQUIREMENTS_FILE = "requirements.txt"

# Model weights URLs
MODEL_WEIGHTS = {
    "yolov5s.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    "yolov5m.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
    "yolov5l.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
}

class Colors:
    """ANSI color codes for cross-platform colored output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @classmethod
    def disable_on_windows(cls):
        """Disable colors on Windows if not supported."""
        if platform.system() == "Windows":
            try:
                # Try to enable ANSI colors on Windows 10+
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                # Fallback: disable colors
                for attr in dir(cls):
                    if not attr.startswith('_') and attr != 'disable_on_windows':
                        setattr(cls, attr, '')

# Initialize colors
Colors.disable_on_windows()

def print_colored(message: str, color: str = Colors.WHITE) -> None:
    """Print colored message with cross-platform support."""
    print(f"{color}{message}{Colors.END}")

def print_header(title: str) -> None:
    """Print a formatted header."""
    print_colored("\n" + "=" * 60, Colors.CYAN)
    print_colored(f"  {title}", Colors.BOLD + Colors.CYAN)
    print_colored("=" * 60, Colors.CYAN)

def print_step(step: str) -> None:
    """Print a setup step."""
    print_colored(f"\n🔧 {step}", Colors.BLUE)

def print_success(message: str) -> None:
    """Print success message."""
    print_colored(f"✅ {message}", Colors.GREEN)

def print_warning(message: str) -> None:
    """Print warning message."""
    print_colored(f"⚠️  {message}", Colors.YELLOW)

def print_error(message: str) -> None:
    """Print error message."""
    print_colored(f"❌ {message}", Colors.RED)

def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    current_version = sys.version_info[:2]
    
    if current_version < PYTHON_MIN_VERSION:
        print_error(f"Python {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}+ required. "
                   f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    if current_version < RECOMMENDED_PYTHON_VERSION:
        print_warning(f"Python {RECOMMENDED_PYTHON_VERSION[0]}.{RECOMMENDED_PYTHON_VERSION[1]}+ recommended. "
                     f"Current version: {current_version[0]}.{current_version[1]}")
    else:
        print_success(f"Python version {current_version[0]}.{current_version[1]} is compatible")
    
    return True

def get_platform_info() -> Tuple[str, str]:
    """Get platform information."""
    system = platform.system()
    architecture = platform.machine()
    
    print_success(f"Detected platform: {system} {architecture}")
    
    return system, architecture

def create_virtual_environment() -> bool:
    """Create virtual environment with cross-platform support."""
    print_step("Creating virtual environment")
    
    venv_path = Path(VENV_NAME)
    
    if venv_path.exists():
        print_warning(f"Virtual environment '{VENV_NAME}' already exists")
        return True
    
    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", VENV_NAME], check=True)
        print_success(f"Virtual environment created: {VENV_NAME}")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error creating virtual environment: {e}")
        return False

def get_venv_python() -> str:
    """Get path to Python executable in virtual environment."""
    system = platform.system()
    venv_path = Path(VENV_NAME)
    
    if system == "Windows":
        return str(venv_path / "Scripts" / "python.exe")
    else:
        return str(venv_path / "bin" / "python")

def get_venv_pip() -> str:
    """Get path to pip executable in virtual environment."""
    system = platform.system()
    venv_path = Path(VENV_NAME)
    
    if system == "Windows":
        return str(venv_path / "Scripts" / "pip.exe")
    else:
        return str(venv_path / "bin" / "pip")

def install_dependencies() -> bool:
    """Install Python dependencies."""
    print_step("Installing dependencies")
    
    if not Path(REQUIREMENTS_FILE).exists():
        print_error(f"Requirements file not found: {REQUIREMENTS_FILE}")
        return False
    
    pip_path = get_venv_pip()
    
    try:
        # Upgrade pip first
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        print_success("pip upgraded successfully")
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", REQUIREMENTS_FILE], check=True)
        print_success("Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error installing dependencies: {e}")
        return False

def download_model_weights(models: List[str] = None) -> bool:
    """Download pre-trained model weights."""
    print_step("Downloading model weights")
    
    if models is None:
        models = ["yolov5s.pt"]  # Default to smallest model
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    success = True
    
    for model_name in models:
        if model_name not in MODEL_WEIGHTS:
            print_warning(f"Unknown model: {model_name}")
            continue
        
        model_path = models_dir / model_name
        
        if model_path.exists():
            print_success(f"Model already exists: {model_name}")
            continue
        
        try:
            print(f"📥 Downloading {model_name}...")
            urllib.request.urlretrieve(MODEL_WEIGHTS[model_name], model_path)
            print_success(f"Downloaded: {model_name}")
            
        except Exception as e:
            print_error(f"Failed to download {model_name}: {e}")
            success = False
    
    return success

def create_directories() -> bool:
    """Create necessary project directories."""
    print_step("Creating project directories")
    
    directories = [
        "configs",
        "models", 
        "datasets",
        "runs",
        "test_images",
        "docs",
        "utils"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print_success("Project directories created")
    return True

def validate_installation() -> bool:
    """Validate the installation."""
    print_step("Validating installation")
    
    python_path = get_venv_python()
    
    # Test imports
    test_imports = [
        "torch",
        "torchvision", 
        "yolov5",
        "PIL",
        "yaml",
        "tkinter"
    ]
    
    failed_imports = []
    
    for module in test_imports:
        try:
            result = subprocess.run(
                [python_path, "-c", f"import {module}"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                failed_imports.append(module)
        except Exception:
            failed_imports.append(module)
    
    if failed_imports:
        print_error(f"Failed to import: {', '.join(failed_imports)}")
        return False
    
    print_success("All dependencies validated successfully")
    return True

def create_activation_scripts() -> bool:
    """Create platform-specific activation scripts."""
    print_step("Creating activation scripts")
    
    system = platform.system()
    
    # Create activation script for current platform
    if system == "Windows":
        script_content = f"""@echo off
echo Activating YOLO Classifier environment...
call {VENV_NAME}\\Scripts\\activate.bat
echo Environment activated! You can now run:
echo   python dog_cat_yolo_gui.py
echo   python generic_yolo_classifier.py
echo   python train_model.py --interactive
cmd /k
"""
        script_path = "activate.bat"
    else:
        script_content = f"""#!/bin/bash
echo "Activating YOLO Classifier environment..."
source {VENV_NAME}/bin/activate
echo "Environment activated! You can now run:"
echo "  python dog_cat_yolo_gui.py"
echo "  python generic_yolo_classifier.py"
echo "  python train_model.py --interactive"
exec "$SHELL"
"""
        script_path = "activate.sh"
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if system != "Windows":
        Path(script_path).chmod(0o755)
    
    print_success(f"Activation script created: {script_path}")
    return True

def interactive_setup() -> bool:
    """Interactive setup wizard."""
    print_header("YOLO Classifier Interactive Setup")
    
    print_colored("\nWelcome to the YOLO Classifier setup wizard!", Colors.CYAN)
    print_colored("This will guide you through the installation process.\n", Colors.WHITE)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Get platform info
    system, arch = get_platform_info()
    
    # Ask about model weights
    print_colored("\nWhich model weights would you like to download?", Colors.CYAN)
    print("1. YOLOv5s (smallest, fastest) - Recommended")
    print("2. YOLOv5m (medium)")
    print("3. YOLOv5l (large)")
    print("4. All models")
    print("5. Skip download (download later)")
    
    choice = input("\nEnter choice (1-5) [1]: ").strip()
    
    model_choices = {
        "1": ["yolov5s.pt"],
        "2": ["yolov5m.pt"],
        "3": ["yolov5l.pt"],
        "4": list(MODEL_WEIGHTS.keys()),
        "5": []
    }
    
    models_to_download = model_choices.get(choice, ["yolov5s.pt"])
    
    # Confirm setup
    print_colored(f"\nSetup Summary:", Colors.CYAN)
    print(f"  Platform: {system} {arch}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Virtual Environment: {VENV_NAME}")
    print(f"  Models: {', '.join(models_to_download) if models_to_download else 'None'}")
    
    confirm = input("\nProceed with setup? (y/N): ").strip().lower()
    if confirm != 'y':
        print_colored("Setup cancelled.", Colors.YELLOW)
        return False
    
    # Run setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
    ]
    
    if models_to_download:
        steps.append(("Downloading model weights", lambda: download_model_weights(models_to_download)))
    
    steps.extend([
        ("Validating installation", validate_installation),
        ("Creating activation scripts", create_activation_scripts)
    ])
    
    for step_name, step_func in steps:
        if not step_func():
            print_error(f"Setup failed at step: {step_name}")
            return False
    
    return True

def quick_setup() -> bool:
    """Quick setup with defaults."""
    print_header("YOLO Classifier Quick Setup")
    
    if not check_python_version():
        return False
    
    get_platform_info()
    
    steps = [
        create_directories,
        create_virtual_environment,
        install_dependencies,
        lambda: download_model_weights(["yolov5s.pt"]),
        validate_installation,
        create_activation_scripts
    ]
    
    for step_func in steps:
        if not step_func():
            return False
    
    return True

def check_installation() -> bool:
    """Check existing installation."""
    print_header("Installation Check")
    
    issues = []
    
    # Check Python version
    if not check_python_version():
        issues.append("Python version")
    
    # Check virtual environment
    venv_path = Path(VENV_NAME)
    if not venv_path.exists():
        issues.append("Virtual environment missing")
        print_error(f"Virtual environment not found: {VENV_NAME}")
    else:
        print_success(f"Virtual environment found: {VENV_NAME}")
    
    # Check dependencies
    if venv_path.exists():
        if not validate_installation():
            issues.append("Dependencies")
    
    # Check directories
    required_dirs = ["configs", "models", "utils"]
    for directory in required_dirs:
        if not Path(directory).exists():
            issues.append(f"Missing directory: {directory}")
            print_error(f"Directory missing: {directory}")
        else:
            print_success(f"Directory found: {directory}")
    
    # Check model weights
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pt"))
        if model_files:
            print_success(f"Model weights found: {[f.name for f in model_files]}")
        else:
            print_warning("No model weights found in models/ directory")
    
    if issues:
        print_error(f"Issues found: {', '.join(issues)}")
        return False
    else:
        print_success("Installation looks good!")
        return True

def print_next_steps() -> None:
    """Print next steps after successful setup."""
    system = platform.system()
    
    print_header("Setup Complete!")
    
    print_colored("🎉 Installation completed successfully!", Colors.GREEN)
    print_colored("\nNext steps:", Colors.CYAN)
    
    if system == "Windows":
        print("1. Run: activate.bat")
    else:
        print("1. Run: ./activate.sh")
    
    print("2. Try the simple classifier: python dog_cat_yolo_gui.py")
    print("3. Create custom classifier: python train_model.py --interactive")
    print("4. Use generic classifier: python generic_yolo_classifier.py")
    
    print_colored("\nDocumentation:", Colors.CYAN)
    print("• README.md - Project overview")
    print("• docs/ - Detailed guides")
    
    print_colored("\nNeed help?", Colors.CYAN)
    print("• Check the troubleshooting section in README.md")
    print("• Review the documentation in docs/")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="YOLO Classifier Setup")
    parser.add_argument("--quick", action="store_true", help="Quick setup with defaults")
    parser.add_argument("--dev", action="store_true", help="Development setup")
    parser.add_argument("--check", action="store_true", help="Check installation")
    
    args = parser.parse_args()
    
    try:
        if args.check:
            success = check_installation()
        elif args.quick:
            success = quick_setup()
        else:
            success = interactive_setup()
        
        if success:
            if not args.check:
                print_next_steps()
            sys.exit(0)
        else:
            print_error("Setup failed. Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print_colored("\n\nSetup interrupted by user.", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 