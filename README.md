# YOLO Classifier Project 🎯

A comprehensive, production-ready computer vision suite featuring both simple and advanced YOLO-based image classifiers. Designed for beginners learning computer vision and professionals building custom classification systems.

## 🌟 Key Features

### 🌍 Cross-Platform Compatibility
- **Windows, Linux, macOS** support with unified codebase
- **Automated setup** with platform detection
- **Interactive launcher** for all applications
- **Virtual environment** management
- **Path handling** with cross-platform file operations
- **Colored terminal** output with fallback support

### 🐕🐱 Simple Dog/Cat Classifier (`dog_cat_yolo_gui.py`)
- **Beginner-friendly** Tkinter GUI with modern styling
- **Pre-trained** YOLOv5 model with COCO weights
- **One-click** image classification
- **Real-time** predictions with confidence scoring
- **Perfect for learning** computer vision fundamentals
- **Automatic model management** and caching

### 🎯 Generic YOLO Classifier (`generic_yolo_classifier.py`)
- **Fully configurable** for any objects/classes
- **YAML-based** configuration system
- **Custom model** support with automatic loading
- **Batch processing** with progress tracking
- **Integrated training** capabilities
- **Professional GUI** with advanced features
- **Export functionality** (CSV, JSON, reports)
- **Real-time monitoring** and statistics

### 🚀 Streamlined Training Pipeline (`train_model.py`)
- **Interactive training** wizard for beginners
- **Command-line interface** for automation
- **Automatic dataset** preparation and validation
- **Configuration generation** with best practices
- **Real-time progress** monitoring
- **Multiple model sizes** support (YOLOv5s/m/l/x)
- **Error handling** and recovery

### ⚙️ Configuration Generator (`create_config.py`)
- **Interactive wizard** for easy setup
- **Template generation** for common use cases
- **Parameter validation** and optimization
- **Integration** with existing models

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Cross-platform setup (Windows, Linux, macOS)
python setup.py

# Or quick setup with defaults
python setup.py --quick
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

### Option 3: Manual Setup
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

# Run applications
python dog_cat_yolo_gui.py
python train_model.py --interactive
```

## 📁 Project Structure

```
yolo-example/
├── 📄 Core Applications
│   ├── dog_cat_yolo_gui.py      # Simple dog/cat classifier (beginner-friendly)
│   ├── generic_yolo_classifier.py  # Advanced configurable classifier
│   ├── train_model.py           # Complete training pipeline
│   └── create_config.py         # Configuration file generator
│
├── 📁 configs/                  # Configuration files for different classifiers
│   ├── vehicle_classifier.yaml # Pre-configured vehicle detection
│   ├── food_classifier.yaml    # Pre-configured food classification
│   └── custom_objects.yaml     # Template for custom objects
│
├── 📄 setup.py                  # Cross-platform setup script
├── 📄 run.py                    # Unified application launcher
│
├── 📁 utils/                    # Utility modules and helper functions
├── 📁 models/                   # Pre-trained model weights
├── 📁 test_images/              # Sample images for testing
├── 📁 docs/                     # Comprehensive documentation
└── 📁 datasets/, runs/, tests/  # Auto-created during use
```

## 🎯 Use Cases

### Medical Imaging
```bash
python train_model.py --dataset_path medical_scans --project_name medical_ai --classes "normal,abnormal,critical"
```

### Quality Control
```bash
python train_model.py --dataset_path factory_parts --project_name qc_inspector --classes "pass,fail,review"
```

### Security Monitoring
```bash
python train_model.py --dataset_path security_footage --project_name security_ai --classes "person,vehicle,animal,clear"
```

### Wildlife Monitoring
```bash
python train_model.py --dataset_path wildlife_cameras --project_name wildlife_detector --classes "deer,bear,wolf,empty"
```

## 📋 Installation

### Prerequisites
- Python 3.8+ (3.9+ recommended)
- pip package manager
- 4GB+ RAM (8GB+ recommended for training)
- GPU support optional but recommended for training

### Automated Setup (All Platforms)
```bash
# Interactive setup wizard
python setup.py

# Quick setup with defaults
python setup.py --quick

# Check existing installation
python setup.py --check
```

### Manual Installation
```bash
# Clone the repository
git clone <repository-url>
cd yolo-example

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
- Ensure Python is added to PATH
- Use `python` command (not `python3`)
- Virtual environment activation: `yolo_env\Scripts\activate`

**Linux/macOS:**
- May need `python3` instead of `python`
- Virtual environment activation: `source yolo_env/bin/activate`
- On some systems, install `python3-venv`: `sudo apt install python3-venv`

**macOS:**
- Install Xcode Command Line Tools: `xcode-select --install`
- Consider using Homebrew for Python: `brew install python`

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
  iou_threshold: 0.45

classes:
  - cat
  - dog
  - bird

gui:
  title: My Custom Classifier
  window_size: [800, 600]
  theme: modern

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
2. Choose dataset option (existing, annotation needed, or sample)
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

## 📊 Monitoring and Results

### Training Outputs
After training, you'll find:
- **Model weights**: `runs/train/{project_name}/weights/best.pt`
- **Training metrics**: `runs/train/{project_name}/results.png`
- **Confusion matrix**: `runs/train/{project_name}/confusion_matrix.png`
- **Configuration file**: `configs/{project_name}_classifier.yaml`

### Testing Your Model
```bash
# Use the quick launcher
./scripts/run_generic.sh

# Or run directly
python generic_yolo_classifier.py --config configs/my_classifier_classifier.yaml
```

## 🔍 Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch size
python train_model.py --batch_size 8
```

**Poor Accuracy**
- Check label quality and consistency
- Increase training data (1000+ images per class recommended)
- Use more epochs (200-500)
- Balance your dataset

**Model Not Loading**
- Check file paths in configuration
- Ensure model weights exist
- Verify YAML syntax

**Training Too Slow**
- Use GPU if available
- Reduce image size (416 instead of 640)
- Use smaller model (YOLOv5s)

### Getting Help
1. Check the comprehensive guides in `docs/`
2. Review training logs in `runs/train/`
3. Start with interactive mode for guidance
4. Test with sample datasets first

## 📚 Documentation

- **[Cross-Platform Guide](CROSS_PLATFORM_GUIDE.md)** - Windows, Linux, macOS support
- **[Training Guide](docs/TRAINING_GUIDE.md)** - Comprehensive training documentation
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Detailed project organization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper comments
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [PyTorch](https://pytorch.org/) for deep learning framework
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for GUI development

## 🎯 What's Next?

- **Web Interface**: Deploy as a web application
- **API Integration**: RESTful API for programmatic access
- **Mobile App**: React Native or Flutter mobile version
- **Cloud Training**: Integration with cloud training services
- **Model Zoo**: Pre-trained models for common use cases

---

**Ready to get started?** Run `./scripts/quick_train.sh` for the complete workflow or `python dog_cat_yolo_gui.py` for a quick demo!