#!/usr/bin/env python3
"""
BirdDog X1 Enhanced Camera Tracking System - Setup Verification Script

This script verifies that all dependencies and system requirements are properly 
installed and configured for the BirdDog X1 Enhanced Camera Tracking System.

Run this script after installation to ensure everything is working correctly.
"""

import sys
import platform
import subprocess
import importlib
import socket
import os
from pathlib import Path

class SetupVerifier:
    def __init__(self):
        self.results = {
            'python': False,
            'dependencies': {},
            'system': {},
            'network': {},
            'models': {},
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
    def print_header(self, text):
        """Print formatted header"""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}")
    
    def print_result(self, test_name, passed, details=""):
        """Print test result with formatting"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        if not passed:
            self.results['errors'].append(f"{test_name}: {details}")
    
    def print_warning(self, test_name, message):
        """Print warning message"""
        print(f"⚠️  WARN {test_name}")
        print(f"    {message}")
        self.results['warnings'].append(f"{test_name}: {message}")
    
    def check_python_version(self):
        """Verify Python version requirements"""
        self.print_header("Python Version Check")
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        # Check for Python 3.10.11 (recommended) or compatible versions
        if version.major == 3 and version.minor == 10:
            if version.micro == 11:
                self.print_result("Python Version", True, f"Perfect! {version_str} (recommended)")
                self.results['python'] = True
            else:
                self.print_result("Python Version", True, f"Good! {version_str} (Python 3.10.x compatible)")
                self.results['python'] = True
        elif version.major == 3 and version.minor >= 8:
            self.print_warning("Python Version", f"Using {version_str}. Python 3.10.11 is recommended for best compatibility")
            self.results['python'] = True
        else:
            self.print_result("Python Version", False, f"Unsupported version {version_str}. Need Python 3.8+ (3.10.11 recommended)")
        
        print(f"    Platform: {platform.platform()}")
        print(f"    Architecture: {platform.architecture()[0]}")
    
    def check_dependencies(self):
        """Check all required Python packages"""
        self.print_header("Python Dependencies Check")
        
        # Core dependencies with minimum versions
        dependencies = {
            'cv2': ('opencv-python', '4.8.0'),
            'numpy': ('numpy', '1.21.0'),
            'ultralytics': ('ultralytics', '8.0.0'),
            'torch': ('torch', '1.13.0'),
            'torchvision': ('torchvision', '0.14.0'),
            'scipy': ('scipy', '1.9.0'),
            'numba': ('numba', '0.56.0'),
            'NDIlib': ('ndi-python', '5.1.0'),
            'PIL': ('Pillow', '9.0.0'),
            'tkinter': ('tkinter', 'built-in')
        }
        
        # Optional dependencies
        optional_deps = {
            'cyndilib': ('cyndilib', '0.0.5'),
        }
        
        for module_name, (package_name, min_version) in dependencies.items():
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'unknown')
                
                self.print_result(f"Import {package_name}", True, f"Version: {version}")
                self.results['dependencies'][package_name] = True
                
                # Special checks for key modules
                if module_name == 'torch':
                    cuda_available = module.cuda.is_available()
                    if cuda_available:
                        print(f"    🚀 CUDA GPU acceleration available")
                        self.results['system']['gpu'] = True
                    else:
                        print(f"    💻 CPU-only mode (GPU acceleration not available)")
                        self.results['system']['gpu'] = False
                
                elif module_name == 'cv2':
                    print(f"    OpenCV build info: {module.getBuildInformation().split('\\n')[0] if hasattr(module, 'getBuildInformation') else 'N/A'}")
                
            except ImportError:
                self.print_result(f"Import {package_name}", False, f"Module not found - install with: pip install {package_name}")
                self.results['dependencies'][package_name] = False
        
        # Check optional dependencies
        print(f"\n  Optional Dependencies:")
        for module_name, (package_name, min_version) in optional_deps.items():
            try:
                importlib.import_module(module_name)
                self.print_result(f"Import {package_name} (optional)", True, "Available")
            except ImportError:
                self.print_warning(f"Import {package_name} (optional)", f"Not installed - may improve NDI performance")
    
    def check_visca_capability(self):
        """Check VISCA-over-IP camera control capability"""
        self.print_header("VISCA Camera Control Check")
        
        try:
            # Try to import visca_over_ip
            from visca_over_ip import Camera
            self.print_result("VISCA Library", True, "visca_over_ip imported successfully")
            self.results['dependencies']['visca_over_ip'] = True
            
            # Test basic camera object creation (without connection)
            try:
                test_camera = Camera("192.168.0.1")  # Dummy IP for testing
                self.print_result("VISCA Object Creation", True, "Camera object created successfully")
            except Exception as e:
                self.print_warning("VISCA Object Creation", f"Issue creating camera object: {e}")
            
        except ImportError:
            self.print_result("VISCA Library", False, "visca_over_ip not found - install with: pip install visca_over_ip")
            self.results['dependencies']['visca_over_ip'] = False
    
    def check_models(self):
        """Check for YOLO model files"""
        self.print_header("YOLO Model Check")
        
        # Check for model files
        model_files = ['best.pt', 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        found_models = []
        
        for model_file in model_files:
            if Path(model_file).exists():
                size = Path(model_file).stat().st_size / (1024*1024)  # MB
                found_models.append(f"{model_file} ({size:.1f}MB)")
                self.results['models'][model_file] = True
        
        if found_models:
            self.print_result("YOLO Models", True, f"Found: {', '.join(found_models)}")
        else:
            self.print_warning("YOLO Models", "No YOLO model files found. For tracking mode, you need a trained model (best.pt)")
            self.results['recommendations'].append("Download a YOLO model or train your own for object tracking")
        
        # Test YOLO model loading if ultralytics is available
        if self.results['dependencies'].get('ultralytics', False):
            try:
                from ultralytics import YOLO
                
                # Try to load a basic model
                if Path('best.pt').exists():
                    model = YOLO('best.pt')
                    self.print_result("Custom Model Loading", True, "best.pt loaded successfully")
                else:
                    # Try to download and load a standard model
                    model = YOLO('yolov8n.pt')  # This will download if not present
                    self.print_result("Standard Model Loading", True, "yolov8n.pt downloaded and loaded")
                    
            except Exception as e:
                self.print_result("Model Loading Test", False, f"Error loading YOLO model: {e}")
    
    def check_network_capability(self):
        """Check network configuration for camera connectivity"""
        self.print_header("Network Configuration Check")
        
        # Get local IP addresses
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            self.print_result("Network Interface", True, f"Local IP: {local_ip}")
            self.results['network']['local_ip'] = local_ip
        except Exception as e:
            self.print_result("Network Interface", False, f"Cannot determine local IP: {e}")
        
        # Check if we can bind to VISCA port (52381)
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.bind(('', 52381))
            test_socket.close()
            self.print_result("VISCA Port (52381)", True, "Port available for camera control")
        except OSError as e:
            if e.errno == 98:  # Address already in use
                self.print_warning("VISCA Port (52381)", "Port in use - camera control may conflict")
            else:
                self.print_result("VISCA Port (52381)", False, f"Port test failed: {e}")
        
        # Check NDI port range (5353 for discovery)
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            test_socket.bind(('', 5353))
            test_socket.close()
            self.print_result("NDI Discovery Port (5353)", True, "Port available for NDI discovery")
        except OSError as e:
            if e.errno == 98:  # Address already in use
                self.print_warning("NDI Discovery Port (5353)", "Port in use - NDI discovery may be active")
            else:
                self.print_result("NDI Discovery Port (5353)", False, f"Port test failed: {e}")
    
    def check_system_resources(self):
        """Check system resources and performance"""
        self.print_header("System Resources Check")
        
        # Memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb >= 16:
                self.print_result("System Memory", True, f"{memory_gb:.1f}GB (Excellent for 4K)")
            elif memory_gb >= 8:
                self.print_result("System Memory", True, f"{memory_gb:.1f}GB (Good for 1080p)")
            else:
                self.print_warning("System Memory", f"{memory_gb:.1f}GB (May struggle with high resolution)")
            
            # CPU check
            cpu_count = psutil.cpu_count()
            self.print_result("CPU Cores", True, f"{cpu_count} cores")
            
        except ImportError:
            self.print_warning("System Resources", "psutil not available - install with 'pip install psutil' for detailed system info")
        
        # Disk space check
        try:
            disk_usage = os.statvfs('.')
            free_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
            
            if free_gb >= 10:
                self.print_result("Disk Space", True, f"{free_gb:.1f}GB free")
            else:
                self.print_warning("Disk Space", f"{free_gb:.1f}GB free (May need more space for models/recordings)")
        except:
            print("    Cannot check disk space on this platform")
    
    def check_ndi_installation(self):
        """Check NDI Runtime installation"""
        self.print_header("NDI Runtime Check")
        
        # Check if NDI Python module works
        ndi_available = self.results['dependencies'].get('ndi-python', False)
        
        if ndi_available:
            try:
                import NDIlib as ndi
                
                # Try to initialize NDI
                if not ndi.initialize():
                    self.print_result("NDI Runtime", False, "NDI failed to initialize - check NDI Runtime installation")
                else:
                    self.print_result("NDI Runtime", True, "NDI initialized successfully")
                    
                    # Try NDI source discovery
                    find = ndi.find_create_v2()
                    if find is None:
                        self.print_warning("NDI Discovery", "Cannot create NDI finder - check network configuration")
                    else:
                        ndi.find_wait_for_sources(find, 1000)  # Wait 1 second
                        sources = ndi.find_get_current_sources(find)
                        
                        if sources:
                            source_names = [ndi.source_get_name(s) for s in sources]
                            self.print_result("NDI Source Discovery", True, f"Found {len(sources)} sources: {', '.join(source_names)}")
                            self.results['network']['ndi_sources'] = source_names
                        else:
                            self.print_warning("NDI Source Discovery", "No NDI sources found - ensure camera is on network and streaming")
                        
                        ndi.find_destroy(find)
                    ndi.destroy()
                    
            except Exception as e:
                self.print_result("NDI Runtime Test", False, f"NDI runtime error: {e}")
        else:
            self.print_result("NDI Python Module", False, "NDI module not available")
    
    def generate_recommendations(self):
        """Generate setup recommendations based on results"""
        self.print_header("Setup Recommendations")
        
        if not self.results['python']:
            print("🔴 CRITICAL: Install Python 3.10.11 from https://python.org")
        
        failed_deps = [name for name, status in self.results['dependencies'].items() if not status]
        if failed_deps:
            print(f"🔴 CRITICAL: Install missing dependencies:")
            print(f"    pip install {' '.join(failed_deps)}")
        
        if not self.results['models']:
            print("🟡 RECOMMENDED: Download or train a YOLO model for object tracking")
            print("    Quick start: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
        
        if not self.results['system'].get('gpu', False):
            print("🟡 OPTIONAL: Consider GPU acceleration for better performance")
            print("    Install CUDA and run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        if not self.results['network'].get('ndi_sources'):
            print("🟡 SETUP: Configure your BirdDog X1 camera:")
            print("    1. Connect camera to same network as computer")
            print("    2. Enable NDI streaming on camera")
            print("    3. Note camera's IP address for configuration")
        
        # Custom recommendations
        for rec in self.results['recommendations']:
            print(f"💡 TIP: {rec}")
    
    def test_basic_functionality(self):
        """Test basic system functionality"""
        self.print_header("Basic Functionality Test")
        
        # Test GUI capability
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Hide the window
            root.destroy()
            self.print_result("GUI Framework", True, "Tkinter available for configuration dialogs")
        except Exception as e:
            self.print_result("GUI Framework", False, f"Tkinter error: {e}")
        
        # Test image processing
        if self.results['dependencies'].get('opencv-python', False):
            try:
                import cv2
                import numpy as np
                
                # Create a test image
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                success = cv2.imwrite('test_image.jpg', test_img)
                if success and Path('test_image.jpg').exists():
                    Path('test_image.jpg').unlink()  # Clean up
                    self.print_result("Image Processing", True, "OpenCV can create and save images")
                else:
                    self.print_result("Image Processing", False, "OpenCV cannot save images")
                    
            except Exception as e:
                self.print_result("Image Processing", False, f"OpenCV test failed: {e}")
        
        # Test mathematical operations
        if self.results['dependencies'].get('numba', False):
            try:
                from numba import jit
                import numpy as np
                
                @jit
                def test_function(x):
                    return x * 2
                
                result = test_function(np.array([1, 2, 3]))
                self.print_result("JIT Compilation", True, "Numba JIT compilation working")
            except Exception as e:
                self.print_result("JIT Compilation", False, f"Numba JIT test failed: {e}")
    
    def run_full_verification(self):
        """Run complete system verification"""
        print("🎥 BirdDog X1 Enhanced Camera Tracking System")
        print("📋 Setup Verification Script")
        print("=" * 60)
        
        self.check_python_version()
        self.check_dependencies()
        self.check_visca_capability()
        self.check_ndi_installation()
        self.check_models()
        self.check_network_capability()
        self.check_system_resources()
        self.test_basic_functionality()
        self.generate_recommendations()
        
        # Final summary
        self.print_header("Verification Summary")
        
        total_checks = sum([
            1,  # Python
            len(self.results['dependencies']),
            len(self.results['system']),
            len(self.results['network']),
        ])
        
        passed_checks = sum([
            1 if self.results['python'] else 0,
            sum(self.results['dependencies'].values()),
            sum(self.results['system'].values()),
            1 if self.results['network'] else 0,
        ])
        
        if len(self.results['errors']) == 0:
            print("✅ ALL CRITICAL CHECKS PASSED")
            print("🚀 System is ready for BirdDog X1 Enhanced Camera Tracking!")
        else:
            print(f"❌ {len(self.results['errors'])} CRITICAL ISSUES FOUND")
            print("🔧 Please resolve the issues above before proceeding")
        
        if self.results['warnings']:
            print(f"⚠️  {len(self.results['warnings'])} warnings (optional improvements)")
        
        print(f"\n📊 Overall Score: {passed_checks}/{total_checks} checks passed")
        
        return len(self.results['errors']) == 0

def main():
    """Main verification function"""
    verifier = SetupVerifier()
    success = verifier.run_full_verification()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 