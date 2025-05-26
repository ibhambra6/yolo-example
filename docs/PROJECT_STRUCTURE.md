# YOLO Classifier Project Structure 📁

This document explains the organization and purpose of every file and directory in the project.

## 📂 Root Directory Structure

```
yolo-example/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT license
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Cross-platform setup script
├── 📄 run.py                       # Unified application launcher
├── 📄 .gitignore                   # Git ignore rules
├── 📄 CROSS_PLATFORM_GUIDE.md     # Platform-specific usage guide
│
├── 🎯 Core Applications/
│   ├── 📄 dog_cat_yolo_gui.py      # Simple dog/cat classifier (beginner-friendly)
│   ├── 📄 generic_yolo_classifier.py  # Advanced configurable classifier
│   ├── 📄 train_model.py           # Complete training pipeline
│   └── 📄 create_config.py         # Configuration file generator
│
├── 📁 configs/                     # Configuration files for different classifiers
│   ├── 📄 vehicle_classifier.yaml # Pre-configured vehicle detection
│   ├── 📄 food_classifier.yaml    # Pre-configured food classification
│   └── 📄 custom_objects.yaml     # Template for custom objects
│
├── 📁 utils/                       # Utility modules and helper functions
│   └── 📄 dataset_utils.py         # Dataset preparation and conversion tools
│
├── 📁 docs/                        # Documentation
│   ├── 📄 PROJECT_STRUCTURE.md     # This file - project organization guide
│   └── 📄 TRAINING_GUIDE.md        # Comprehensive training documentation
│
├── 📁 test_images/                 # Sample images for testing classifiers
│   ├── 🖼️ dog1.jpg, dog2.jpg      # Sample dog images
│   ├── 🖼️ cat1.jpg, cat2.jpg      # Sample cat images
│   └── 🖼️ ...                     # Additional test images
│
├── 📁 tests/                       # Unit tests and test utilities
│   ├── 📄 test_dog_cat_classifier.py  # Tests for classifier functionality
│   └── 📄 __init__.py              # Test package initialization
│
├── 📁 datasets/                    # Training datasets (created during use)
│   └── 📁 [project_name]/          # Individual project datasets
│       ├── 📁 train/               # Training data
│       ├── 📁 val/                 # Validation data
│       └── 📁 test/                # Test data
│
├── 📁 runs/                        # Training outputs and model weights
│   └── 📁 train/                   # Training run results
│       └── 📁 [project_name]/      # Individual training runs
│           ├── 📁 weights/         # Model weights (best.pt, last.pt)
│           ├── 📄 results.png      # Training metrics visualization
│           └── 📄 confusion_matrix.png  # Model performance analysis
│
├── 📁 models/                      # Pre-trained model weights
│   └── 📄 yolov5s.pt              # YOLOv5 small model weights
│
└── 📁 yolo_env/                    # Python virtual environment (auto-created)
    └── ...                         # Virtual environment files
```

## 🎯 Core Applications

### 1. `dog_cat_yolo_gui.py` - Simple Classifier
**Purpose**: Beginner-friendly dog/cat classifier with basic GUI
**Features**:
- Simple Tkinter interface
- Pre-trained YOLOv5 model
- Drag-and-drop image loading
- Real-time classification
- Perfect for learning and demos

**Usage**:
```bash
python run.py --app dog_cat
# or directly:
python dog_cat_yolo_gui.py
```

### 2. `generic_yolo_classifier.py` - Advanced Classifier
**Purpose**: Configurable classifier for any objects/classes
**Features**:
- YAML-based configuration
- Custom model support
- Batch processing
- Training integration
- Professional GUI
- Export capabilities

**Usage**:
```bash
python run.py --app generic
# or directly:
python generic_yolo_classifier.py --config configs/my_classifier.yaml
```

### 3. `train_model.py` - Training Pipeline
**Purpose**: Complete end-to-end model training system
**Features**:
- Interactive training wizard
- Dataset preparation
- Automatic configuration generation
- Progress monitoring
- Error handling and validation

**Usage**:
```bash
python run.py --app train
# or directly:
python train_model.py --interactive
```

### 4. `create_config.py` - Configuration Generator
**Purpose**: Create configuration files for existing models
**Features**:
- Interactive configuration wizard
- Command-line interface
- Template generation
- Validation and testing

**Usage**:
```bash
python run.py --app config
# or directly:
python create_config.py --interactive
```

## 🚀 Setup and Launcher Scripts

### `setup.py` - Cross-Platform Setup
**Purpose**: Automated environment setup for all platforms
**Features**:
- Platform detection (Windows, Linux, macOS)
- Virtual environment creation
- Dependency installation
- Model weight downloading
- Environment validation

**Usage**:
```bash
python setup.py              # Interactive setup
python setup.py --quick      # Quick setup with defaults
python setup.py --check      # Check existing installation
```

### `run.py` - Unified Launcher
**Purpose**: Single entry point for all applications
**Features**:
- Interactive application menu
- Direct application launching
- Environment validation
- Cross-platform compatibility

**Usage**:
```bash
python run.py                # Interactive menu
python run.py --app dog_cat   # Launch specific app
python run.py --list         # List available apps
python run.py --check        # Check environment
```

## 📁 Directory Purposes

### `/configs/` - Configuration Files
Contains YAML configuration files that define:
- Model weights and parameters
- Class definitions
- GUI settings
- Training parameters
- Metadata

### `/utils/` - Utility Modules
Contains helper functions and utilities:
- Dataset format conversion
- Data preprocessing
- Validation tools
- Common operations

### `/docs/` - Documentation
Project documentation including:
- Project structure (this file)
- Training guides
- Usage instructions

### `/test_images/` - Sample Images
Pre-included test images for:
- Quick testing of classifiers
- Demonstration purposes
- Validation of setup

### `/tests/` - Unit Tests
Automated tests for:
- Classifier functionality
- Training pipeline
- Configuration handling
- Error conditions

### `/datasets/` - Training Data
Auto-created during training:
- Organized by project name
- YOLO format structure
- Train/validation/test splits

### `/runs/` - Training Outputs
Contains training results:
- Model weights (best.pt, last.pt)
- Training metrics and visualizations
- Performance analysis
- Logs and checkpoints

### `/models/` - Pre-trained Weights
Contains downloaded model weights:
- YOLOv5 variants (s, m, l, x)
- Custom trained models
- Backup weights

## 🚀 Quick Start Workflows

### For Beginners
1. **Setup Environment**:
   ```bash
   python setup.py
   ```

2. **Simple Classification**:
   ```bash
   python run.py --app dog_cat
   ```

3. **Custom Training**:
   ```bash
   python run.py --app train
   ```

### For Advanced Users
1. **Custom Classifier**:
   ```bash
   python run.py --app generic
   ```

2. **Command Line Training**:
   ```bash
   python train_model.py --dataset_path data --project_name my_model --classes "a,b,c"
   ```

3. **Configuration Management**:
   ```bash
   python create_config.py --name my_model --classes "a,b,c" --weights model.pt
   ```

## 🔧 Development Workflow

### Adding New Features
1. **Core Applications**: Add to root directory
2. **Utilities**: Add to `/utils/`
3. **Tests**: Add to `/tests/`
4. **Documentation**: Update relevant docs

### Configuration Management
1. **Create Config**: Use `create_config.py`
2. **Test Config**: Use `generic_yolo_classifier.py`
3. **Store Config**: Save in `/configs/`

### Training Workflow
1. **Prepare Dataset**: Use `dataset_utils.py`
2. **Train Model**: Use `train_model.py`
3. **Validate Results**: Check `/runs/train/`
4. **Create Config**: Use `create_config.py`
5. **Test Model**: Use `generic_yolo_classifier.py`

## 📊 File Dependencies

```
setup.py → requirements.txt
run.py → [all core applications]
train_model.py → utils/dataset_utils.py
generic_yolo_classifier.py → configs/*.yaml
create_config.py → configs/*.yaml
```

## 🎯 Best Practices

### File Organization
- Keep core applications in root directory
- Use descriptive file names
- Group related files in directories
- Maintain consistent naming conventions

### Configuration Management
- Use YAML for all configurations
- Include metadata in config files
- Validate configurations before use
- Provide example configurations

### Documentation
- Update this file when adding new components
- Include usage examples in docstrings
- Maintain cross-references between docs
- Keep documentation current with code changes

---

**Need to understand a specific component?** Check the relevant source file or refer to the main [README.md](../README.md) for usage examples. 