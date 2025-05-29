# BirdDog X1 YOLO Object Tracking System

This system automatically controls a BirdDog X1 camera to keep a YOLO-detected object centered in the frame using pan/tilt movements.

## Features

- Real-time object detection using YOLO
- Automatic camera pan/tilt control via VISCA-IP
- NDI video stream reception
- Visual feedback with detection overlay
- Configurable tracking parameters
- **GUI application for manual camera control**

## Prerequisites

- Python ≥ 3.9
- BirdDog X1 camera with NDI streaming enabled
- Network connectivity to the camera
- Trained YOLO model file (`best.pt`)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the NewTek NDI runtime installed on your system

## Configuration

Edit the configuration variables in `birddog.py`:

```python
CAMERA_IP   = "192.168.100.150"    # Your X1 camera IP address
NDI_NAME    = "X1-Studio-Cam"      # NDI stream name from your camera
MODEL_PATH  = "best.pt"            # Path to your trained YOLO model
TRACK_CLASS = 0                    # Class ID to track (0 = first class in your model)
```

### Important Configuration Parameters:

- **CAMERA_IP**: Set this to your BirdDog X1's actual IP address
- **NDI_NAME**: Check your camera's NDI stream name (may vary)
- **MODEL_PATH**: Place your trained YOLO model file in the same directory or update the path
- **TRACK_CLASS**: Set to the class ID of the object you want to track
- **DEADBAND**: Adjustment threshold (0.05 = 5% of frame center)
- **FOV_H_DEG**: Horizontal field of view in degrees (55.8° for X1 at widest)

## Applications

This package includes two main applications:

### 1. Automated Object Tracking (`birddog.py`)
Automatically tracks objects detected by your trained YOLO model.

### 2. Manual Camera Control GUI (`birddog_gui.py`)
A graphical interface for manual camera control with live video feed.

## Usage

### Automated Tracking

1. Ensure your BirdDog X1 is powered on and connected to the network
2. Make sure NDI streaming is enabled on the camera
3. Place your trained YOLO model file (`best.pt`) in the project directory
4. Run the tracking system:

```bash
python birddog.py
```

### Manual Camera Control

To use the GUI application for manual camera control:

```bash
python birddog_gui.py
```

The GUI provides:
- **Live NDI video feed display**
- **Pan/Tilt controls** with adjustable speed (1-24)
- **Zoom controls** with adjustable speed (1-7)
- **Home position button** to reset camera
- **Real-time status display**
- **Connection status monitoring**

#### GUI Controls:
- **Arrow buttons**: Pan and tilt the camera
- **Red STOP button**: Immediately stop all movement
- **Zoom In/Out**: Control camera zoom
- **Speed sliders**: Adjust pan/tilt and zoom speeds
- **Home Position**: Return camera to default position

5. The system will:
   - Connect to the NDI stream
   - Start object detection
   - Display the video feed with detection overlays
   - Automatically pan/tilt the camera to keep detected objects centered

6. Press `q` or `ESC` to quit

## How It Works

1. **Video Reception**: Receives video frames from the BirdDog X1 via NDI
2. **Object Detection**: Runs YOLO inference on each frame
3. **Target Selection**: Selects the highest-confidence detection of the target class
4. **Position Calculation**: Calculates the offset from frame center
5. **Camera Control**: Sends VISCA-IP commands to pan/tilt the camera
6. **Visual Feedback**: Displays detection boxes, center lines, and offset values

## Troubleshooting

- **No NDI stream found**: Check camera IP, NDI name, and network connectivity
- **No detections**: Verify your YOLO model and class ID configuration
- **Camera not responding**: Check VISCA-IP port (52381) and camera IP
- **Poor tracking**: Adjust DEADBAND, speed settings, or retrain your YOLO model

## Model Training

To train a custom YOLO model for your specific tracking target:

1. Collect and annotate training data
2. Train using YOLOv8/Ultralytics:
```bash
yolo train data=your_dataset.yaml model=yolov8n.pt epochs=100
```
3. Replace `best.pt` with your trained model
4. Update `TRACK_CLASS` to match your model's class structure 