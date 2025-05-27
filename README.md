# YOLO Classifier Project 🎯

A comprehensive, production-ready computer vision suite featuring both simple and advanced YOLO-based image classifiers. Designed for beginners learning computer vision and professionals building custom classification systems.

## 🌟 Key Features

### 🌍 Cross-Platform Compatibility
- **Windows, Linux, macOS** support with unified codebase
- **Automated setup** with platform detection and virtual environment management
- **Interactive launcher** for all applications with unified entry point
- **Activation scripts** for easy environment management

### 🐕🐱 Simple Dog/Cat Classifier (`dog_cat_yolo_gui.py`)
- **Beginner-friendly** simplified GUI with modern styling
- **Pre-trained** YOLOv5 model with COCO weights
- **One-click** image classification with confidence scoring
- **Perfect for learning** computer vision fundamentals

### 🎯 Generic YOLO Classifier (`generic_yolo_classifier.py`)
- **Fully configurable** for any objects/classes via YAML configuration
- **Custom model** support with automatic loading
- **Batch processing** with progress tracking and CSV export
- **Professional GUI** with advanced features and real-time monitoring

### 🚀 Training Pipeline (`train_model.py`)
- **Interactive training** wizard for beginners
- **Command-line interface** for automation
- **Automatic dataset** preparation and configuration generation
- **Real-time progress** monitoring with error handling

### ⚙️ Configuration Generator (`create_config.py`)
- **Interactive wizard** for easy setup
- **Template generation** for common use cases
- **Parameter validation** and optimization

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Interactive setup with guided wizard
python setup.py

# Quick setup with defaults
python setup.py --quick

# Check existing installation
python setup.py --check
```

### Option 2: Interactive Launcher
```bash
# Launch any application with guided menu
python run.py

# Or launch specific apps directly
python run.py --app dog_cat     # Simple dog/cat classifier
python run.py --app generic     # Advanced classifier
python run.py --app train       # Training wizard
python run.py --app config      # Configuration creator
```

### Option 3: Environment Activation
After running setup, you can use the generated activation script:

**Windows:**
```cmd
activate.bat
```

**Linux/macOS:**
```bash
./activate.sh
```

This activates the virtual environment and provides helpful commands.

## 📁 Project Structure

```
yolo-example/
├── 📄 Core Applications
│   ├── dog_cat_yolo_gui.py          # Simple dog/cat classifier
│   ├── generic_yolo_classifier.py   # Advanced configurable classifier
│   ├── train_model.py               # Complete training pipeline
│   └── create_config.py             # Configuration file generator
│
├── 📄 Setup & Launcher
│   ├── setup.py                     # Cross-platform setup script
│   ├── run.py                       # Unified application launcher
│   ├── activate.sh / activate.bat   # Environment activation (auto-generated)
│   └── requirements.txt             # Dependencies
│
├── 📁 Configuration & Data
│   ├── configs/                     # YAML configuration files
│   ├── models/                      # Pre-trained model weights
│   ├── test_images/                 # Sample images for testing
│   └── utils/                       # Utility modules
│
└── 📁 Auto-Generated
    ├── datasets/                    # Training datasets
    ├── runs/                        # Training outputs
    ├── yolo_env/                    # Virtual environment
    └── docs/                        # Documentation
```

## 🎯 Use Cases & Examples

### Quick Classification
```bash
# Simple dog/cat classification
python run.py --app dog_cat

# Multi-class object detection
python run.py --app generic
```

### Custom Training
```bash
# Interactive training wizard
python run.py --app train

# Command line training
python train_model.py --dataset_path data --project_name my_classifier --classes "cat,dog,bird"
```

### Specialized Applications
```bash
# Vehicle detection
python generic_yolo_classifier.py --config configs/vehicle_classifier.yaml

# Food classification
python generic_yolo_classifier.py --config configs/food_classifier.yaml
```

## 📋 Installation

### Prerequisites
- Python 3.8+ (3.9+ recommended)
- pip package manager
- 4GB+ RAM (8GB+ recommended for training)

### Automated Installation
```bash
# Clone or download the project
cd yolo-example

# Run setup (handles everything automatically)
python setup.py
```

The setup script will:
- ✅ Check Python version compatibility
- ✅ Create virtual environment (`yolo_env/`)
- ✅ Install all dependencies
- ✅ Download model weights
- ✅ Create activation scripts
- ✅ Validate installation

### Manual Installation (Advanced Users)
```bash
# Create virtual environment
python -m venv yolo_env

# Activate environment
# Windows:
yolo_env\Scripts\activate
# Linux/macOS:
source yolo_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python run.py --check
```

### Platform-Specific Notes

**Windows:**
- Use `python` command (not `python3`)
- Virtual environment: `yolo_env\Scripts\activate`
- Activation script: `activate.bat`

**Linux/macOS:**
- May need `python3` instead of `python`
- Virtual environment: `source yolo_env/bin/activate`
- Activation script: `./activate.sh`
- On some systems: `sudo apt install python3-venv`

## 🔧 Configuration

### Creating Custom Configurations
```bash
# Interactive configuration wizard
python create_config.py --interactive

# Command line configuration
python create_config.py --name my_classifier --classes "cat,dog,bird" --weights models/my_model.pt
```

### Example Configuration (YAML)
```yaml
model:
  weights: runs/train/my_classifier/weights/best.pt
  confidence_threshold: 0.5

classes:
  - cat
  - dog
  - bird

gui:
  title: My Custom Classifier
  window_size: [800, 600]

metadata:
  description: Custom animal classifier
  version: 1.0
```

## 🎓 Training Your Own Models

### Interactive Training (Recommended)
```bash
python train_model.py --interactive
```

Follow the step-by-step wizard:
1. Enter project name and description
2. Choose dataset option
3. Define class names
4. Set training parameters
5. Start training and monitor progress

### Command Line Training
```bash
python train_model.py \
    --dataset_path /path/to/dataset \
    --project_name my_classifier \
    --classes "class1,class2,class3" \
    --epochs 200 \
    --batch_size 16
```

### Dataset Format
Your dataset should follow YOLO format:
```
dataset/
├── train/
│   ├── images/     # Training images (.jpg, .png)
│   └── labels/     # Training labels (.txt)
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
└── test/
    ├── images/     # Test images
    └── labels/     # Test labels
```

## 📊 Results & Monitoring

### Training Outputs
After training, you'll find:
- **Model weights**: `runs/train/{project_name}/weights/best.pt`
- **Training metrics**: `runs/train/{project_name}/results.png`
- **Configuration file**: `configs/{project_name}_classifier.yaml`

### Testing Your Model
```bash
# Use the unified launcher
python run.py --app generic

# Or run directly
python generic_yolo_classifier.py --config configs/my_classifier_classifier.yaml
```

## 🔍 Troubleshooting

### Common Issues

**Environment Issues**
```bash
# Check installation
python setup.py --check

# Recreate environment
python setup.py --quick
```

**Model Loading Errors**
- Ensure virtual environment is activated
- Check that model weights exist
- Verify YAML configuration syntax
- If missing cv2: `pip install opencv-python`

**Training Issues**
- Check dataset format and paths
- Ensure sufficient disk space
- Reduce batch size if out of memory

### Getting Help
1. Check environment with `python run.py --check`
2. Review training logs in `runs/train/`
3. Use interactive mode for guidance
4. Test with sample datasets first

## 📚 Documentation

- **[Training Guide](docs/TRAINING_GUIDE.md)** - Comprehensive training documentation
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Detailed project organization
- **[Cross-Platform Guide](CROSS_PLATFORM_GUIDE.md)** - Platform-specific usage

## 📄 License

This project is proprietary software licensed exclusively for Oceaneering International Inc. use only. All other use, modification, or distribution is prohibited. See the [LICENSE](LICENSE) file for complete terms and restrictions.

## 🙏 Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [PyTorch](https://pytorch.org/) for deep learning framework
- [COCO Dataset](https://cocodataset.org/) for pre-trained models

## 🎯 What's Next?

- **Web Interface**: Deploy as a web application
- **API Integration**: RESTful API for programmatic access
- **Mobile App**: React Native or Flutter mobile version
- **Cloud Training**: Integration with cloud training services

---

**Ready to get started?** Run `python setup.py` for automated setup or `python run.py` to launch the application!