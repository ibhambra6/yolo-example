# BirdDog X1 YOLO Object Tracking & Camera Control System

This comprehensive system provides both automated object tracking and manual camera control for BirdDog X1 cameras. It uses YOLO for real-time object detection and VISCA-over-IP for precise camera control.

## 🎯 Features

### Automated Object Tracking
- Real-time object detection using YOLOv8/Ultralytics
- Automatic camera pan/tilt control to keep objects centered
- Configurable tracking parameters and deadband zones
- Visual feedback with detection overlays and center lines
- Support for custom trained YOLO models

### Manual Camera Control GUI
- **Live NDI video feed** in separate OpenCV window
- **Pan/Tilt controls** with adjustable speed (1-24)
- **Zoom controls** with adjustable speed (1-7) 
- **Camera settings** (focus, exposure, white balance, backlight)
- **Picture flip controls** (horizontal and vertical)
- **Preset positions** (save and recall up to 10 presets)
- **Real-time status** and connection monitoring
- **Graceful shutdown** handling for both windows

### Video Processing
- **NDI stream reception** with multiple format support (BGRA, RGB, UYVY)
- **Automatic color conversion** for proper display
- **Format detection** and fallback handling
- **High-quality video display** in separate window

## 📋 Prerequisites

### Hardware Requirements
- **BirdDog X1 camera** with NDI streaming enabled
- **Network connectivity** between camera and computer
- **Windows/macOS/Linux** computer with Python support

### Software Requirements
- **Python 3.10.11** (recommended version for compatibility)
- **NewTek NDI Runtime** installed on your system
- **Trained YOLO model** file (for object tracking mode)

## 🚀 Installation

### 1. Install Python 3.10.11

**Windows:**
```powershell
# Option A: Download from python.org (Recommended)
# 1. Visit https://www.python.org/downloads/release/python-31011/
# 2. Download "Windows installer (64-bit)" 
# 3. Run installer and check "Add Python to PATH"
# 4. Verify installation:
python --version
# Should show: Python 3.10.11

# Option B: Using winget (Windows Package Manager)
winget install Python.Python.3.10

# Option C: Using chocolatey
choco install python --version=3.10.11

# Verify installation
python --version
py -3.10 --version
```

**macOS:**
```bash
# Option A: Download from python.org (Recommended)
# 1. Visit https://www.python.org/downloads/release/python-31011/
# 2. Download "macOS 64-bit universal2 installer"
# 3. Run the installer package
# 4. Verify installation:
python3.10 --version
# Should show: Python 3.10.11

# Option B: Using Homebrew
brew install python@3.10

# Option C: Using pyenv (for multiple Python versions)
brew install pyenv
pyenv install 3.10.11
pyenv global 3.10.11

# Verify installation
python3.10 --version
```

**Linux (Ubuntu/Debian):**
```bash
# Option A: Using apt (Ubuntu 22.04+)
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-pip

# Option B: Using deadsnakes PPA (for older Ubuntu versions)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-pip python3.10-dev

# Option C: Compile from source
cd /tmp
wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz
tar -xzf Python-3.10.11.tgz
cd Python-3.10.11
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall

# Verify installation
python3.10 --version
# Should show: Python 3.10.11
```

**Linux (CentOS/RHEL/Fedora):**
```bash
# CentOS/RHEL 8+
sudo dnf install python3.10 python3.10-pip python3.10-venv

# Fedora
sudo dnf install python3.10 python3-pip

# CentOS/RHEL 7 (compile from source)
sudo yum groupinstall "Development Tools"
sudo yum install openssl-devel bzip2-devel libffi-devel
cd /tmp
wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz
tar -xzf Python-3.10.11.tgz
cd Python-3.10.11
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall

# Verify installation
python3.10 --version
```

### 2. Clone or Download
```bash
# If using git
git clone <repository-url>
cd birddog

# Or download and extract the files
```

### 3. Create Virtual Environment with Python 3.10

**Windows:**
```powershell
# Verify Python 3.10 is available
python --version
# If multiple Python versions, use specific launcher:
py -3.10 --version

# Create virtual environment with Python 3.10
python -m venv venv
# Or if you have multiple Python versions:
py -3.10 -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify the virtual environment is using Python 3.10
python --version
# Should show: Python 3.10.11

# Verify activation (should show (venv) in prompt)
# Prompt should look like: (venv) PS C:\path\to\birddog>
```

**macOS/Linux:**
```bash
# Verify Python 3.10 is available
python3.10 --version
# Should show: Python 3.10.11

# Create virtual environment with Python 3.10
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify the virtual environment is using Python 3.10
python --version
# Should show: Python 3.10.11

# Verify activation (should show (venv) in prompt)
# Prompt should look like: (venv) username@hostname:~/birddog$
```

**Troubleshooting Virtual Environment:**
```bash
# If venv creation fails, ensure venv module is available:
# Windows:
py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install virtualenv

# macOS/Linux:
python3.10 -m pip install --upgrade pip
python3.10 -m pip install virtualenv

# Alternative venv creation using virtualenv:
# Windows:
virtualenv -p python3.10 venv
# macOS/Linux:
virtualenv -p python3.10 venv
```

**To deactivate virtual environment later:**
```bash
deactivate
```

### 4. Install Python Dependencies
```bash
# Make sure virtual environment is activated first
# You should see (venv) in your prompt

# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Or install manually with specific versions:
pip install ultralytics>=8.0.0 opencv-python>=4.5.0 numpy>=1.21.0 visca-over-ip>=1.0.0 ndi-python>=5.1.0

# Verify installations
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import cv2, numpy, ultralytics, NDIlib; print('All packages imported successfully')"
```

### 5. Install NDI Runtime & Tools

**Option A: Manual Download (Recommended)**
1. **Visit NDI Downloads**: [https://ndi.tv/sdk/](https://ndi.tv/sdk/)
2. **Register/Login** (free account required)
3. **Download NDI Runtime** for your platform:
   - Windows: `NDI_x_x_x_Runtime_Windows.exe`
   - macOS: `NDI_x_x_x_Runtime_macOS.pkg`
   - Linux: `NDI_x_x_x_Runtime_Linux.tar.gz`
4. **Install the downloaded file**

**Option B: Command Line Download (Advanced)**

**Windows (PowerShell):**
```powershell
# Note: Direct download requires NDI account authentication
# Manual download is recommended, but here's the general approach:

# Create downloads directory
New-Item -ItemType Directory -Force -Path "C:\temp\ndi"
cd C:\temp\ndi

# You'll need to get the actual download URL from NDI website after login
# Example structure (URL will be different):
# Invoke-WebRequest -Uri "https://downloads.ndi.tv/SDK/NDI_x_x_x_Runtime_Windows.exe" -OutFile "NDI_Runtime.exe"

# Install silently (once downloaded)
# .\NDI_Runtime.exe /S
```

**macOS:**
```bash
# Create downloads directory
mkdir -p ~/Downloads/ndi
cd ~/Downloads/ndi

# Download NDI Runtime (you'll need to get actual URL from NDI website)
# curl -L "https://downloads.ndi.tv/SDK/NDI_x_x_x_Runtime_macOS.pkg" -o NDI_Runtime.pkg

# Install the package
# sudo installer -pkg NDI_Runtime.pkg -target /
```

**Linux (Ubuntu/Debian):**
```bash
# Create downloads directory
mkdir -p ~/Downloads/ndi
cd ~/Downloads/ndi

# Download NDI Runtime (get actual URL from NDI website)
# wget "https://downloads.ndi.tv/SDK/NDI_x_x_x_Runtime_Linux.tar.gz"

# Extract and install
# tar -xzf NDI_x_x_x_Runtime_Linux.tar.gz
# cd NDI_x_x_x_Runtime_Linux
# sudo ./install.sh
```

**Option C: Package Managers (Limited Availability)**

**Windows (Chocolatey - if available):**
```powershell
# Check if NDI is available in chocolatey
choco search ndi

# Install if available
# choco install ndi-runtime
```

**macOS (Homebrew - if available):**
```bash
# Check if NDI is available in homebrew
brew search ndi

# Install if available  
# brew install --cask ndi-runtime
```

**Verify NDI Installation:**
```bash
# Windows - Check if NDI DLLs are in system PATH
where /R C:\ Processing.NDI.Lib.x64.dll

# macOS/Linux - Check for NDI libraries
find /usr -name "*ndi*" 2>/dev/null
```

### 6. Download NDI Tools (Optional but Useful)

**NDI Tools include useful utilities like NDI Studio Monitor for testing:**

1. **Visit NDI Tools page**: [https://ndi.tv/tools/](https://ndi.tv/tools/)
2. **Download NDI Tools** for your platform
3. **Install NDI Tools**

**Useful NDI Tools for testing:**
- **NDI Studio Monitor**: View NDI streams (test your camera)
- **NDI Access Manager**: Manage NDI permissions
- **NDI Test Patterns**: Generate test NDI streams

### 7. Prepare YOLO Model (for tracking mode)
```bash
# Place your trained YOLO model in the project directory
# Example:
cp /path/to/your/best.pt ./best.pt

# Or download a pre-trained model for testing
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"
```

### 8. Verify Installation
```bash
# Test Python imports
python -c "import cv2, numpy, ultralytics, NDIlib; print('All packages imported successfully')"

# Test NDI discovery (should show available NDI sources)
python ndi_debug.py

# Test camera connection (update IP address first)
python -c "from visca_over_ip import Camera; c = Camera('192.168.0.13'); print('Camera connection test passed')"
```

## ⚙️ Configuration

### Camera Network Setup
1. **Connect your BirdDog X1** to the same network as your computer
2. **Note the camera's IP address** (check camera menu or network settings)
3. **Enable NDI streaming** on the camera (usually enabled by default)
4. **Verify NDI stream name** (typically "CAM", "CAM2", or similar)

### Software Configuration
Edit the configuration variables in both `birddog.py` and `birddog_gui.py`:

```python
# Network Configuration
CAMERA_IP   = "192.168.0.13"       # Your X1 camera IP address
NDI_NAME    = "CAM2"               # NDI stream name from your camera
VISCA_PORT  = 52381                # VISCA-IP port (usually 52381)

# Tracking Configuration (birddog.py only)
MODEL_PATH  = "best.pt"            # Path to your trained YOLO model
TRACK_CLASS = 0                    # Class ID to track (0 = first class)
FOV_H_DEG   = 55.8                 # Horizontal field of view in degrees
DEADBAND    = 0.05                 # Center dead-zone (±5% of frame)
PT_SPEED_MIN, PT_SPEED_MAX = 0x01, 0x18  # VISCA velocity range
```

### Important Configuration Notes:
- **CAMERA_IP**: Must match your camera's actual IP address
- **NDI_NAME**: Check your camera's NDI output name (may be "CAM", "CAM2", etc.)
- **MODEL_PATH**: Only needed for automated tracking mode
- **TRACK_CLASS**: Set to the class ID of the object you want to track
- **DEADBAND**: Smaller values = more sensitive tracking

## 🎮 Usage

### Manual Camera Control GUI

**Start the GUI application:**
```bash
python birddog_gui.py
```

**GUI Controls:**
- **Pan/Tilt**: Use arrow buttons for directional movement
- **Speed Control**: Adjust pan/tilt speed with slider (1-24)
- **Zoom**: Use Zoom In/Out buttons with speed control (1-7)
- **Stop**: Red STOP button immediately halts all movement
- **Home**: Return camera to home position
- **Reset**: Reset camera settings

**Camera Settings:**
- **Backlight**: Toggle backlight compensation
- **Auto Focus/Iris**: Enable automatic focus and iris control
- **White Balance**: Select auto, indoor, outdoor, or manual
- **Auto Exposure**: Enable automatic exposure control
- **Picture Flip**: Separate horizontal and vertical flip controls
- **Gain**: Manual gain adjustment controls

**Preset Management:**
- **Set Preset**: Save current position (1-10)
- **Call Preset**: Return to saved position
- **Preset Selection**: Use spinner to select preset number

**Window Management:**
- **Main GUI**: Control interface with all camera settings
- **Video Feed**: Separate OpenCV window with live NDI stream
- **Graceful Close**: All windows close properly when app is closed

### Automated Object Tracking

**Start object tracking:**
```bash
python birddog.py
```

**Tracking Behavior:**
1. **Video Reception**: Connects to NDI stream from camera
2. **Object Detection**: Runs YOLO inference on each frame
3. **Target Selection**: Chooses highest-confidence detection
4. **Position Calculation**: Calculates offset from frame center
5. **Camera Control**: Sends pan/tilt commands to center the object
6. **Visual Feedback**: Shows detection boxes and tracking info

**Keyboard Controls:**
- **ESC** or **Q**: Quit the application
- **Window close**: Click X to exit

## 🔧 Troubleshooting

### Connection Issues

**"Camera: Unknown and NDI: disconnected"**
- ✅ Check camera IP address in configuration
- ✅ Verify camera is powered on and connected to network
- ✅ Ensure NDI streaming is enabled on camera
- ✅ Check NDI stream name (try "CAM", "CAM2", "X1-Studio-Cam")
- ✅ Verify NewTek NDI Runtime is installed

**VISCA commands not working**
- ✅ Check CAMERA_IP matches your camera's IP
- ✅ Verify VISCA port 52381 is accessible
- ✅ Ensure camera supports VISCA-over-IP (BirdDog X1 does)
- ✅ Check firewall settings

### Video Issues

**Black and white video instead of color**
- ✅ Video format should auto-detect and convert properly
- ✅ Check NDI stream settings on camera
- ✅ Verify NDI Runtime installation

**No video window appears**
- ✅ Check that OpenCV is properly installed
- ✅ Verify NDI stream is available
- ✅ Check console output for error messages

**Video window freezes or crashes**
- ✅ Close application gracefully using GUI close button
- ✅ Check network stability
- ✅ Restart both camera and application

### Camera Control Issues

**Pan/tilt not responding**
- ✅ Check VISCA connection (IP and port)
- ✅ Verify camera is not in local control mode
- ✅ Check console output for VISCA error messages
- ✅ Try camera reset from GUI

**Picture flip not working**
- ✅ Use separate Flip H and Flip V checkboxes
- ✅ Check console output for flip command status
- ✅ Some camera settings may override flip commands

**Zoom control issues**
- ✅ Verify zoom speed settings (1-7)
- ✅ Check if camera has reached zoom limits
- ✅ Try zoom stop before changing direction

### Object Tracking Issues

**No object detections**
- ✅ Verify YOLO model file (`best.pt`) exists
- ✅ Check `TRACK_CLASS` matches your model's classes
- ✅ Ensure adequate lighting and object visibility
- ✅ Test model with ultralytics: `yolo predict model=best.pt source=0`

**Poor tracking performance**
- ✅ Adjust `DEADBAND` value (try 0.02-0.1)
- ✅ Modify speed settings (`PT_SPEED_MIN/MAX`)
- ✅ Retrain YOLO model with more diverse data
- ✅ Check camera field of view setting (`FOV_H_DEG`)

## 🎓 Advanced Usage

### Custom YOLO Model Training

**Prepare training data:**
1. Collect images of your target object
2. Annotate using tools like LabelImg or Roboflow
3. Create dataset in YOLO format

**Train the model:**
```bash
# Install ultralytics if not already installed
pip install ultralytics

# Train your model
yolo train data=your_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640

# Test the trained model
yolo predict model=runs/detect/train/weights/best.pt source=0
```

**Use your trained model:**
1. Copy `runs/detect/train/weights/best.pt` to your project directory
2. Update `MODEL_PATH = "best.pt"` in configuration
3. Set `TRACK_CLASS` to match your model's target class

### Network Configuration

**Static IP Setup (Recommended):**
1. Set camera to static IP (e.g., 192.168.0.13)
2. Configure computer network adapter to same subnet
3. Update `CAMERA_IP` in configuration files

**DHCP Setup:**
1. Note camera's assigned IP from router/DHCP server
2. Update `CAMERA_IP` when IP changes
3. Consider MAC address reservation for consistent IP

## 📁 File Structure

```
birddog/
├── birddog.py              # Main object tracking application
├── birddog_gui.py           # Manual camera control GUI
├── requirements.txt         # Python dependencies
├── README.md               # This documentation
├── VISCA_COMMANDS.md       # VISCA command reference
├── ndi_debug.py            # NDI troubleshooting tool
├── best.pt                 # Your trained YOLO model (add this)
└── venv/                   # Virtual environment (if used)
```

## 🤝 Support

**For issues related to:**
- **BirdDog X1**: Check BirdDog support documentation
- **NDI**: Verify NewTek NDI Runtime installation
- **YOLO/Ultralytics**: See [Ultralytics documentation](https://docs.ultralytics.com/)
- **Python dependencies**: Check pip installation and virtual environments

**Debug tools:**
- `ndi_debug.py`: Test NDI stream connectivity
- Console output: Monitor for error messages and status updates
- GUI status panel: Real-time connection and operation status

## 📄 License

This project uses the following open-source libraries:
- **Ultralytics**: AGPL-3.0 License
- **OpenCV**: Apache 2.0 License  
- **visca-over-ip**: Check library documentation
- **ndi-python**: Check library documentation

---

**Created for BirdDog X1 camera control and automated object tracking** 