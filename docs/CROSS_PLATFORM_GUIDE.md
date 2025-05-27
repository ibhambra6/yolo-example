# Cross-Platform Usage Guide 🌍

This guide provides platform-specific instructions and troubleshooting for the YOLO Classifier Project on Windows, Linux, and macOS.

## 🎯 Quick Start by Platform

### Windows 🪟

```batch
# Check Python installation
python --version

# Clone and setup
git clone <repository-url>
cd yolo-example
python setup.py

# Activate environment (when needed)
yolo_env\Scripts\activate

# Launch applications
python run.py
```

### Linux 🐧

```bash
# Check Python installation
python3 --version

# Install dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install python3-venv python3-pip git

# Clone and setup
git clone <repository-url>
cd yolo-example
python3 setup.py

# Activate environment (when needed)
source yolo_env/bin/activate

# Launch applications
python3 run.py
```

### macOS 🍎

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python via Homebrew (recommended)
brew install python

# Clone and setup
git clone <repository-url>
cd yolo-example
python3 setup.py

# Activate environment (when needed)
source yolo_env/bin/activate

# Launch applications
python3 run.py
```

## 🔧 Platform-Specific Features

### Windows Features

- **Colored Output**: Automatic ANSI color support detection
- **Path Handling**: Proper Windows path separators
- **Executable Extensions**: `.exe` and `.bat` file handling
- **Virtual Environment**: Windows-style Scripts directory

### Linux Features

- **Package Manager Integration**: APT/YUM package suggestions
- **Shell Integration**: Bash-compatible scripts
- **Permissions**: Automatic executable permissions
- **System Python**: Support for system Python installations

### macOS Features

- **Homebrew Integration**: Recommended Python installation method
- **Xcode Tools**: Automatic detection and installation prompts
- **Metal GPU**: Optimized for Apple Silicon when available
- **App Bundle**: Future support for .app packaging

## 🚨 Common Issues and Solutions

### Python Version Issues

**Problem**: "Python not found" or version too old

**Windows Solution**:

```batch
# Download Python from python.org
# Ensure "Add to PATH" is checked during installation
# Restart command prompt
python --version
```

**Linux Solution**:

```bash
# Ubuntu/Debian
sudo apt install python3 python3-pip python3-venv

# CentOS/RHEL
sudo yum install python3 python3-pip

# Arch Linux
sudo pacman -S python python-pip
```

**macOS Solution**:

```bash
# Using Homebrew (recommended)
brew install python

# Or download from python.org
# Ensure /usr/local/bin is in PATH
```

### Virtual Environment Issues

**Problem**: Virtual environment creation fails

**Solution**:

```bash
# Ensure venv module is available
python -m venv --help

# If not available on Linux:
sudo apt install python3-venv

# Alternative: use virtualenv
pip install virtualenv
virtualenv yolo_env
```

### Permission Issues (Linux/macOS)

**Problem**: Permission denied when running scripts

**Solution**:

```bash
# Make scripts executable
chmod +x *.sh
chmod +x run.py

# Or run with python explicitly
python3 run.py
```

### GUI Issues

**Problem**: Tkinter not available

**Windows**: Usually included with Python
**Linux**:

```bash
# Ubuntu/Debian
sudo apt install python3-tk

# CentOS/RHEL
sudo yum install tkinter
```

**macOS**: Usually included with Python

### GPU/CUDA Issues

**Problem**: CUDA not detected or version mismatch

**Solution**:

```bash
# Check CUDA installation
nvidia-smi

# Install CUDA-specific PyTorch (if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

**Problem**: Out of memory during training

**Solution**:

- Reduce batch size: `--batch_size 8`
- Use smaller model: `yolov5s` instead of `yolov5l`
- Reduce image size: `--img_size 416`
- Close other applications

## 🔍 Debugging and Diagnostics

### Environment Check

```bash
# Check installation
python run.py --check

# Detailed setup check
python setup.py --check

# List available applications
python run.py --list
```

### Dependency Verification

```python
# Test imports manually
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import yolov5; print('YOLOv5: OK')"
python -c "import tkinter; print('Tkinter: OK')"
```

### System Information

```python
# Get system info
python -c "
import platform
import sys
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'Architecture: {platform.machine()}')
"
```

## 📁 File System Considerations

### Path Separators

The project uses `pathlib.Path` for cross-platform path handling:

```python
# Good (cross-platform)
from pathlib import Path
config_path = Path("configs") / "my_config.yaml"

# Avoid (platform-specific)
config_path = "configs/my_config.yaml"  # Unix-style
config_path = "configs\\my_config.yaml"  # Windows-style
```

### Line Endings

- **Windows**: CRLF (`\r\n`)
- **Linux/macOS**: LF (`\n`)

Git handles this automatically with `core.autocrlf` settings.

### Case Sensitivity

- **Windows**: Case-insensitive file system
- **Linux**: Case-sensitive file system
- **macOS**: Case-insensitive by default (can be case-sensitive)

Always use consistent casing for file names.

## 🚀 Performance Optimization

### Windows Optimization

```batch
# Use Windows Terminal for better performance
# Enable hardware acceleration in terminal settings
# Consider WSL2 for Linux-like performance
```

### Linux Optimization

```bash
# Use GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Optimize CPU usage
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### macOS Optimization

```bash
# Use Metal Performance Shaders (if available)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Optimize for Apple Silicon
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## 🔄 Migration Between Platforms

### Moving Projects

1. **Export configuration**: Use YAML files for portability
2. **Model weights**: Copy `.pt` files (platform-independent)
3. **Datasets**: Ensure consistent path structure
4. **Dependencies**: Run `pip freeze > requirements.txt`

### Path Conversion

```python
# Convert paths when moving between platforms
import os
from pathlib import Path

# Convert Windows path to Unix
windows_path = "C:\\Users\\Name\\project\\data"
unix_path = str(Path(windows_path).as_posix())

# Convert Unix path to Windows
unix_path = "/home/user/project/data"
windows_path = str(Path(unix_path).resolve())
```

## 📞 Getting Help

### Platform-Specific Support

- **Windows**: Check Windows Defender/antivirus settings
- **Linux**: Check distribution-specific package managers
- **macOS**: Verify Xcode Command Line Tools installation

### Community Resources

- **GitHub Issues**: Platform-specific bug reports
- **Stack Overflow**: General Python/PyTorch questions
- **PyTorch Forums**: Deep learning specific issues

### Diagnostic Information

When reporting issues, include:

```bash
# System information
python -c "import platform; print(platform.platform())"

# Python version
python --version

# Package versions
pip list | grep -E "(torch|yolov5|PIL)"

# Error logs
python run.py --app train 2>&1 | tee error.log
```

## 🎯 Best Practices

### Development

1. **Use virtual environments** on all platforms
2. **Test on multiple platforms** before release
3. **Use pathlib** for file operations
4. **Handle platform differences** gracefully
5. **Document platform-specific requirements**

### Deployment

1. **Create platform-specific installers** when needed
2. **Test installation scripts** on clean systems
3. **Provide fallback options** for dependencies
4. **Include troubleshooting guides** for each platform

### Maintenance

1. **Keep dependencies updated** across platforms
2. **Monitor platform-specific issues** in bug reports
3. **Test new Python versions** on all platforms
4. **Maintain compatibility matrices** for supported versions

---

**Need more help?** Check the main [README.md](README.md) or open an issue with your platform details!
