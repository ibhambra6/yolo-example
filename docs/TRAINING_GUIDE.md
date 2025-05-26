# YOLO Training Guide 🚀

Complete guide for training custom YOLO models and creating configuration files.

## Quick Start (One Command)

```bash
# Make script executable and run
chmod +x quick_train.sh
./quick_train.sh
```

This launches an interactive menu with all training options.

## Training Workflows

### 1. Interactive Training (Recommended for Beginners)

```bash
python train_model.py --interactive
```

This will guide you through:
- Project setup
- Dataset preparation
- Class definition
- Training parameters
- Automatic configuration generation

### 2. Command Line Training

```bash
python train_model.py \
    --dataset_path /path/to/dataset \
    --project_name my_classifier \
    --classes "cat,dog,bird" \
    --epochs 100 \
    --batch_size 16
```

### 3. Configuration File Only

```bash
python create_config.py --interactive
```

Or with parameters:
```bash
python create_config.py \
    --name my_classifier \
    --classes "cat,dog,bird" \
    --weights path/to/model.pt
```

## Dataset Preparation

### Option 1: YOLO Format Dataset

If you already have a YOLO format dataset:
```
dataset/
├── train/
│   ├── images/     # .jpg, .png files
│   └── labels/     # .txt files
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Option 2: Create Sample Dataset Structure

```bash
python train_model.py --interactive
# Choose option 3: "I want to use a sample dataset"
```

This creates a template structure you can populate with your images and labels.

### Option 3: Convert from Other Formats

The training script can convert from:
- COCO format
- Pascal VOC format
- Custom annotation formats

## Label Format

Each `.txt` file should contain one line per object:
```
class_id center_x center_y width height
```

Where:
- `class_id`: Integer starting from 0
- `center_x, center_y`: Object center (normalized 0-1)
- `width, height`: Object dimensions (normalized 0-1)

Example for a cat at image center:
```
0 0.5 0.5 0.3 0.4
```

## Annotation Tools

### Recommended Tools:
1. **LabelImg** (Free, Desktop)
   ```bash
   pip install labelImg
   labelImg
   ```

2. **Roboflow** (Web-based, Free tier)
   - Visit: https://roboflow.com
   - Upload images, annotate online
   - Export in YOLO format

3. **CVAT** (Free, Self-hosted)
   - Visit: https://cvat.org
   - Professional annotation platform

4. **Labelbox** (Commercial)
   - Enterprise-grade annotation

## Training Parameters

### Basic Parameters:
- **Epochs**: Number of training cycles (100-300 recommended)
- **Batch Size**: Images per batch (16 for most GPUs)
- **Image Size**: Input resolution (640 recommended)
- **Learning Rate**: Training speed (0.01 default)

### Model Sizes:
- **YOLOv5s**: Fastest, smallest (14MB)
- **YOLOv5m**: Balanced (42MB)
- **YOLOv5l**: More accurate (94MB)
- **YOLOv5x**: Most accurate (167MB)

## Complete Training Example

### Step 1: Prepare Dataset
```bash
# Create dataset structure
mkdir -p my_dataset/{train,val,test}/{images,labels}

# Add your images and labels to respective folders
```

### Step 2: Train Model
```bash
python train_model.py \
    --dataset_path my_dataset \
    --project_name animal_classifier \
    --classes "cat,dog,bird,fish" \
    --epochs 200 \
    --batch_size 16 \
    --description "Animal classification model"
```

### Step 3: Test Model
```bash
# Configuration file is automatically created
python generic_yolo_classifier.py --config configs/animal_classifier_classifier.yaml
```

## File Structure After Training

```
yolo-example/
├── configs/
│   └── animal_classifier_classifier.yaml    # Auto-generated config
├── runs/
│   └── train/
│       └── animal_classifier/
│           ├── weights/
│           │   ├── best.pt                  # Best model weights
│           │   └── last.pt                  # Latest weights
│           ├── results.png                  # Training metrics
│           └── confusion_matrix.png         # Model performance
├── datasets/
│   └── my_dataset/                          # Your training data
└── train_model.py                           # Training script
```

## Configuration File Structure

```yaml
model:
  weights: runs/train/animal_classifier/weights/best.pt
  confidence_threshold: 0.5
  iou_threshold: 0.45

classes:
  - cat
  - dog
  - bird
  - fish

gui:
  title: Animal Classifier
  window_size: [800, 600]
  theme: modern

training:
  epochs: 200
  batch_size: 16
  image_size: 640
  learning_rate: 0.01

metadata:
  description: Animal classification model
  created_date: 2024-01-15T10:30:00
  version: 1.0
```

## Common Use Cases

### 1. Medical Image Classification
```bash
python train_model.py \
    --dataset_path medical_scans \
    --project_name medical_classifier \
    --classes "normal,abnormal,suspicious" \
    --epochs 300
```

### 2. Industrial Quality Control
```bash
python train_model.py \
    --dataset_path factory_parts \
    --project_name quality_control \
    --classes "good,defective,damaged" \
    --epochs 250
```

### 3. Wildlife Monitoring
```bash
python train_model.py \
    --dataset_path wildlife_cameras \
    --project_name wildlife_detector \
    --classes "deer,bear,wolf,empty" \
    --epochs 200
```

### 4. Document Classification
```bash
python train_model.py \
    --dataset_path documents \
    --project_name doc_classifier \
    --classes "invoice,receipt,contract,other" \
    --epochs 150
```

## Performance Optimization

### For Better Accuracy:
- Use more training data (1000+ images per class)
- Increase epochs (200-500)
- Use larger model (YOLOv5l or YOLOv5x)
- Apply data augmentation
- Balance your dataset

### For Faster Training:
- Use smaller model (YOLOv5s)
- Reduce image size (416 instead of 640)
- Increase batch size (if GPU allows)
- Use fewer epochs for testing

### For Faster Inference:
- Use YOLOv5s model
- Lower confidence threshold
- Reduce image size
- Use GPU acceleration

## Troubleshooting

### Common Issues:

1. **Out of Memory Error**
   ```bash
   # Reduce batch size
   python train_model.py --batch_size 8
   ```

2. **Poor Accuracy**
   - Check label quality
   - Increase training data
   - Use more epochs
   - Balance dataset

3. **Training Too Slow**
   - Use GPU if available
   - Reduce image size
   - Use smaller model

4. **Model Not Loading**
   - Check file paths in config
   - Ensure weights file exists
   - Verify YAML syntax

### Getting Help:

1. Check training logs in `runs/train/[project_name]/`
2. View training metrics in `results.png`
3. Examine confusion matrix for class performance
4. Test with different confidence thresholds

## Advanced Features

### Custom Data Augmentation
```python
# In train_model.py, modify augmentation settings
augmentation_config = {
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0
}
```

### Transfer Learning
```bash
# Start from a pre-trained model
python train_model.py \
    --weights yolov5m.pt \
    --dataset_path my_dataset \
    --project_name my_classifier
```

### Multi-GPU Training
```bash
# Use multiple GPUs
python -m torch.distributed.launch --nproc_per_node 2 train_model.py \
    --dataset_path my_dataset \
    --project_name my_classifier \
    --device 0,1
```

## Integration Examples

### Web API Integration
```python
from generic_yolo_classifier import YOLOClassifier

# Load your trained model
classifier = YOLOClassifier('configs/my_classifier.yaml')

# Use in web app
@app.route('/classify', methods=['POST'])
def classify_image():
    image = request.files['image']
    results = classifier.predict(image)
    return jsonify(results)
```

### Batch Processing
```python
import os
from generic_yolo_classifier import YOLOClassifier

classifier = YOLOClassifier('configs/my_classifier.yaml')

# Process all images in a folder
for filename in os.listdir('input_images/'):
    if filename.endswith(('.jpg', '.png')):
        results = classifier.predict(f'input_images/{filename}')
        print(f"{filename}: {results}")
```

## Best Practices

1. **Data Quality**
   - Use high-quality, diverse images
   - Ensure accurate annotations
   - Balance classes (similar number of samples)

2. **Training Strategy**
   - Start with pre-trained weights
   - Use appropriate learning rate
   - Monitor validation loss

3. **Model Evaluation**
   - Test on unseen data
   - Check confusion matrix
   - Validate with domain experts

4. **Deployment**
   - Test thoroughly before production
   - Monitor model performance
   - Plan for model updates

## Resources

- [YOLOv5 Documentation](https://docs.ultralytics.com/)
- [Computer Vision Datasets](https://public.roboflow.com/)
- [Annotation Guidelines](https://blog.roboflow.com/how-to-annotate/)
- [Model Optimization](https://docs.ultralytics.com/tutorials/model-optimization/)

---

**Need Help?** 
- Check the troubleshooting section above
- Review training logs in `runs/train/`
- Test with sample datasets first
- Start with interactive mode for guidance 