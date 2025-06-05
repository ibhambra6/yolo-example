"""
BirdDog X1 Camera Control GUI

A graphical interface for controlling the BirdDog X1 camera with live NDI feed display
and pan/tilt/zoom controls.

Requirements:
- All dependencies from birddog.py
- tkinter (usually included with Python)
- visca_over_ip library for camera control
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
from visca_over_ip import Camera

# Import NDI and camera control from the main module
import NDIlib as ndi
from birddog import X1Visca, ndi_frames

### ------------------- GUI CONFIG -------------------
CAMERA_IP   = "192.168.0.7"       # X1 address
NDI_NAME    = "CAM1"         # NDI stream name
VISCA_PORT  = 52381                   # VISCA-IP port
WINDOW_WIDTH = 800                    # GUI window width (reduced since no video)
WINDOW_HEIGHT = 600                   # GUI window height (reduced since no video)
### -------------------------------------------------

class BirdDogGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BirdDog X1 Camera Control")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        
        # Initialize camera control using visca_over_ip library
        self.camera = X1Visca(CAMERA_IP)
        
        # Video streaming variables
        self.video_running = False
        self.video_thread = None
        
        # Control state
        self.is_moving = False
        
        self.setup_gui()
        self.start_video_stream()
    
    def setup_gui(self):
        """Create the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Video status area (instead of video display)
        self.video_frame = ttk.LabelFrame(main_frame, text="Live Feed Status", padding="5")
        self.video_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.video_status_label = tk.Label(self.video_frame, text="Live video will open in separate window", 
                                          font=("Arial", 11), fg="blue")
        self.video_status_label.pack(pady=10)
        
        # Control panels
        self.setup_pan_tilt_controls(main_frame)
        self.setup_zoom_controls(main_frame)
        self.setup_camera_settings(main_frame)
        self.setup_status_panel(main_frame)
    
    def setup_pan_tilt_controls(self, parent):
        """Create pan/tilt control buttons"""
        pt_frame = ttk.LabelFrame(parent, text="Pan/Tilt Control", padding="10")
        pt_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Speed control
        speed_frame = ttk.Frame(pt_frame)
        speed_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        ttk.Label(speed_frame, text="Speed:").grid(row=0, column=0, padx=(0, 5))
        self.speed_var = tk.IntVar(value=10)
        self.speed_scale = ttk.Scale(speed_frame, from_=1, to=24, variable=self.speed_var, orient=tk.HORIZONTAL)
        self.speed_scale.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.speed_label = ttk.Label(speed_frame, text="10")
        self.speed_label.grid(row=0, column=2, padx=(5, 0))
        
        speed_frame.columnconfigure(1, weight=1)
        self.speed_var.trace_add("write", self.update_speed_label)
        
        # Pan/Tilt buttons in cross pattern
        # Tilt Up
        self.tilt_up_btn = tk.Button(pt_frame, text="▲", font=("Arial", 16), width=4, height=2)
        self.tilt_up_btn.grid(row=1, column=1, padx=2, pady=2)
        self.tilt_up_btn.bind("<ButtonPress-1>", lambda e: self.start_move("tilt_up"))
        self.tilt_up_btn.bind("<ButtonRelease-1>", lambda e: self.stop_move())
        
        # Pan Left, Stop, Pan Right
        self.pan_left_btn = tk.Button(pt_frame, text="◄", font=("Arial", 16), width=4, height=2)
        self.pan_left_btn.grid(row=2, column=0, padx=2, pady=2)
        self.pan_left_btn.bind("<ButtonPress-1>", lambda e: self.start_move("pan_left"))
        self.pan_left_btn.bind("<ButtonRelease-1>", lambda e: self.stop_move())
        
        self.stop_btn = tk.Button(pt_frame, text="STOP", font=("Arial", 12, "bold"), 
                                 width=4, height=2, bg="red", fg="white")
        self.stop_btn.grid(row=2, column=1, padx=2, pady=2)
        self.stop_btn.bind("<Button-1>", lambda e: self.stop_move())
        
        self.pan_right_btn = tk.Button(pt_frame, text="►", font=("Arial", 16), width=4, height=2)
        self.pan_right_btn.grid(row=2, column=2, padx=2, pady=2)
        self.pan_right_btn.bind("<ButtonPress-1>", lambda e: self.start_move("pan_right"))
        self.pan_right_btn.bind("<ButtonRelease-1>", lambda e: self.stop_move())
        
        # Tilt Down
        self.tilt_down_btn = tk.Button(pt_frame, text="▼", font=("Arial", 16), width=4, height=2)
        self.tilt_down_btn.grid(row=3, column=1, padx=2, pady=2)
        self.tilt_down_btn.bind("<ButtonPress-1>", lambda e: self.start_move("tilt_down"))
        self.tilt_down_btn.bind("<ButtonRelease-1>", lambda e: self.stop_move())
    
    def setup_zoom_controls(self, parent):
        """Create zoom control buttons"""
        zoom_frame = ttk.LabelFrame(parent, text="Zoom Control", padding="10")
        zoom_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Zoom speed
        zoom_speed_frame = ttk.Frame(zoom_frame)
        zoom_speed_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(zoom_speed_frame, text="Zoom Speed:").grid(row=0, column=0, padx=(0, 5))
        self.zoom_speed_var = tk.IntVar(value=5)
        self.zoom_speed_scale = ttk.Scale(zoom_speed_frame, from_=1, to=7, variable=self.zoom_speed_var, orient=tk.HORIZONTAL)
        self.zoom_speed_scale.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.zoom_speed_label = ttk.Label(zoom_speed_frame, text="5")
        self.zoom_speed_label.grid(row=0, column=2, padx=(5, 0))
        
        zoom_speed_frame.columnconfigure(1, weight=1)
        self.zoom_speed_var.trace_add("write", self.update_zoom_speed_label)
        
        # Zoom buttons
        self.zoom_in_btn = tk.Button(zoom_frame, text="Zoom In\n(+)", font=("Arial", 12), 
                                    width=10, height=3, bg="lightblue")
        self.zoom_in_btn.grid(row=1, column=0, padx=5, pady=5)
        self.zoom_in_btn.bind("<ButtonPress-1>", lambda e: self.start_zoom("in"))
        self.zoom_in_btn.bind("<ButtonRelease-1>", lambda e: self.stop_zoom())
        
        self.zoom_out_btn = tk.Button(zoom_frame, text="Zoom Out\n(-)", font=("Arial", 12), 
                                     width=10, height=3, bg="lightcoral")
        self.zoom_out_btn.grid(row=1, column=1, padx=5, pady=5)
        self.zoom_out_btn.bind("<ButtonPress-1>", lambda e: self.start_zoom("out"))
        self.zoom_out_btn.bind("<ButtonRelease-1>", lambda e: self.stop_zoom())
        
        # Preset buttons
        ttk.Separator(zoom_frame, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=2, 
                                                            sticky=(tk.W, tk.E), pady=10)
        
        preset_frame = ttk.Frame(zoom_frame)
        preset_frame.grid(row=3, column=0, columnspan=2)
        
        self.home_btn = tk.Button(preset_frame, text="Home Position", font=("Arial", 10), 
                                 width=12, height=2, bg="lightgreen")
        self.home_btn.grid(row=0, column=0, padx=2)
        self.home_btn.bind("<Button-1>", lambda e: self.go_home())
        
        # Reset button
        self.reset_btn_zoom = tk.Button(preset_frame, text="Reset Camera", font=("Arial", 10),
                                       width=12, height=2, bg="lightcoral")
        self.reset_btn_zoom.grid(row=0, column=1, padx=2)
        self.reset_btn_zoom.bind("<Button-1>", lambda e: self.reset_camera())
    
    def setup_camera_settings(self, parent):
        """Create camera settings control buttons"""
        settings_frame = ttk.LabelFrame(parent, text="Camera Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Row 1: Basic controls
        basic_frame = ttk.Frame(settings_frame)
        basic_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Backlight toggle
        self.backlight_var = tk.BooleanVar(value=False)
        self.backlight_check = ttk.Checkbutton(basic_frame, text="Backlight", 
                                              variable=self.backlight_var,
                                              command=lambda: self.set_backlight(self.backlight_var.get()))
        self.backlight_check.grid(row=0, column=0, padx=5)
        
        # Auto Focus toggle
        self.auto_focus_var = tk.BooleanVar(value=True)
        self.auto_focus_check = ttk.Checkbutton(basic_frame, text="Auto Focus",
                                               variable=self.auto_focus_var,
                                               command=lambda: self.set_focus_mode(self.auto_focus_var.get()))
        self.auto_focus_check.grid(row=0, column=1, padx=5)
        
        # Auto Iris toggle  
        self.auto_iris_var = tk.BooleanVar(value=True)
        self.auto_iris_check = ttk.Checkbutton(basic_frame, text="Auto Iris",
                                              variable=self.auto_iris_var,
                                              command=lambda: self.set_iris_mode(self.auto_iris_var.get()))
        self.auto_iris_check.grid(row=0, column=2, padx=5)
        
        # Auto Exposure button
        self.auto_exp_btn = tk.Button(basic_frame, text="Auto Exposure", font=("Arial", 10),
                                     width=12, height=1, bg="lightblue")
        self.auto_exp_btn.grid(row=0, column=3, padx=5)
        self.auto_exp_btn.bind("<Button-1>", lambda e: self.auto_exposure_on())
        
        # Row 2: White Balance
        wb_frame = ttk.Frame(settings_frame)
        wb_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(wb_frame, text="White Balance:").grid(row=0, column=0, padx=(0, 5))
        
        self.wb_var = tk.StringVar(value="auto")
        wb_options = ["auto", "indoor", "outdoor", "manual"]
        self.wb_combo = ttk.Combobox(wb_frame, textvariable=self.wb_var, values=wb_options, 
                                    state="readonly", width=10)
        self.wb_combo.grid(row=0, column=1, padx=5)
        self.wb_combo.bind("<<ComboboxSelected>>", lambda e: self.set_white_balance_mode(self.wb_var.get()))
        
        # Row 3: Preset Controls
        preset_frame = ttk.Frame(settings_frame)
        preset_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(preset_frame, text="Preset:").grid(row=0, column=0, padx=(0, 5))
        
        self.preset_var = tk.IntVar(value=1)
        self.preset_spin = tk.Spinbox(preset_frame, from_=1, to=10, textvariable=self.preset_var, 
                                     width=5)
        self.preset_spin.grid(row=0, column=1, padx=5)
        
        self.set_preset_btn = tk.Button(preset_frame, text="Set", font=("Arial", 10),
                                       width=8, height=1, bg="lightgreen")
        self.set_preset_btn.grid(row=0, column=2, padx=2)
        self.set_preset_btn.bind("<Button-1>", lambda e: self.set_preset_position(self.preset_var.get()))
        
        self.call_preset_btn = tk.Button(preset_frame, text="Call", font=("Arial", 10),
                                        width=8, height=1, bg="lightyellow")
        self.call_preset_btn.grid(row=0, column=3, padx=2)
        self.call_preset_btn.bind("<Button-1>", lambda e: self.call_preset_position(self.preset_var.get()))
        
        # Row 4: Utility Controls
        util_frame = ttk.Frame(settings_frame)
        util_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        self.reset_btn = tk.Button(util_frame, text="Reset Camera", font=("Arial", 10),
                                  width=12, height=1, bg="lightcoral")
        self.reset_btn.grid(row=0, column=0, padx=5)
        self.reset_btn.bind("<Button-1>", lambda e: self.reset_camera())
        
        # Picture Flip controls - separate for horizontal and vertical
        flip_frame = ttk.Frame(util_frame)
        flip_frame.grid(row=0, column=1, padx=5)
        
        self.flip_h_var = tk.BooleanVar(value=False)
        self.flip_h_check = ttk.Checkbutton(flip_frame, text="Flip H",
                                           variable=self.flip_h_var,
                                           command=lambda: self.flip_horizontal(self.flip_h_var.get()))
        self.flip_h_check.grid(row=0, column=0, padx=2)
        
        self.flip_v_var = tk.BooleanVar(value=False)
        self.flip_v_check = ttk.Checkbutton(flip_frame, text="Flip V",
                                           variable=self.flip_v_var,
                                           command=lambda: self.flip_vertical(self.flip_v_var.get()))
        self.flip_v_check.grid(row=0, column=1, padx=2)
        
        # Gain controls
        ttk.Label(util_frame, text="Gain:").grid(row=0, column=2, padx=(20, 5))
        
        self.gain_up_btn = tk.Button(util_frame, text="▲", font=("Arial", 8),
                                    width=3, height=1, bg="lightgray")
        self.gain_up_btn.grid(row=0, column=3, padx=2)
        self.gain_up_btn.bind("<Button-1>", lambda e: self.adjust_gain("up"))
        
        self.gain_down_btn = tk.Button(util_frame, text="▼", font=("Arial", 8),
                                      width=3, height=1, bg="lightgray")
        self.gain_down_btn.grid(row=0, column=4, padx=2)
        self.gain_down_btn.bind("<Button-1>", lambda e: self.adjust_gain("down"))
        
        # Close button
        self.close_btn = tk.Button(util_frame, text="Close App", font=("Arial", 10),
                                  width=10, height=1, bg="salmon", fg="white")
        self.close_btn.grid(row=0, column=5, padx=(20, 0))
        self.close_btn.bind("<Button-1>", lambda e: self.on_closing())
    
    def setup_status_panel(self, parent):
        """Create status display panel"""
        status_frame = ttk.LabelFrame(parent, text="Status", padding="10")
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack()
        
        # Connection status
        self.connection_var = tk.StringVar(value="Camera: Disconnected | NDI: Disconnected")
        self.connection_label = ttk.Label(status_frame, textvariable=self.connection_var, 
                                         font=("Arial", 9))
        self.connection_label.pack(pady=(5, 0))
    
    def update_speed_label(self, *args):
        """Update speed display label"""
        self.speed_label.config(text=str(self.speed_var.get()))
    
    def update_zoom_speed_label(self, *args):
        """Update zoom speed display label"""
        self.zoom_speed_label.config(text=str(self.zoom_speed_var.get()))
    
    def start_move(self, direction):
        """Start pan/tilt movement"""
        if self.is_moving:
            return
            
        self.is_moving = True
        speed = self.speed_var.get()
        
        direction_map = {
            "pan_left": (speed, 0, "L", "S"),
            "pan_right": (speed, 0, "R", "S"),
            "tilt_up": (0, speed, "S", "U"),
            "tilt_down": (0, speed, "S", "D")
        }
        
        if direction in direction_map:
            pan_vel, tilt_vel, pan_dir, tilt_dir = direction_map[direction]
            try:
                print(f"Sending VISCA command: {direction} - pan_vel={pan_vel}, tilt_vel={tilt_vel}, pan_dir={pan_dir}, tilt_dir={tilt_dir}")
                self.camera.move(pan_vel, tilt_vel, pan_dir, tilt_dir)
                self.status_var.set(f"Moving: {direction.replace('_', ' ').title()}")
            except Exception as e:
                print(f"Move command failed: {e}")
                self.status_var.set(f"Error: {str(e)}")
    
    def stop_move(self):
        """Stop all movement"""
        if not self.is_moving:
            return
            
        self.is_moving = False
        try:
            print("Sending VISCA stop command")
            self.camera.stop()
            self.status_var.set("Stopped")
        except Exception as e:
            print(f"Stop command failed: {e}")
            self.status_var.set(f"Error: {str(e)}")
    
    def start_zoom(self, direction):
        """Start zoom operation"""
        speed = self.zoom_speed_var.get()
        try:
            print(f"Sending zoom command: {direction} at speed {speed}")
            if direction == "in":
                self.camera.zoom_in(speed)
            else:
                self.camera.zoom_out(speed)
            self.status_var.set(f"Zooming {direction}")
        except Exception as e:
            print(f"Zoom command failed: {e}")
            self.status_var.set(f"Zoom error: {str(e)}")
    
    def stop_zoom(self):
        """Stop zoom operation"""
        try:
            print("Sending zoom stop command")
            self.camera.zoom_stop()
            self.status_var.set("Zoom stopped")
        except Exception as e:
            print(f"Zoom stop failed: {e}")
            self.status_var.set(f"Error: {str(e)}")
    
    def go_home(self):
        """Return camera to home position"""
        try:
            self.camera.pantilt_home()
            self.status_var.set("Returning to home position")
        except Exception as e:
            self.status_var.set(f"Home error: {str(e)}")
    
    def reset_camera(self):
        """Reset camera settings"""
        try:
            self.camera.pantilt_reset()
            self.status_var.set("Camera reset")
        except Exception as e:
            self.status_var.set(f"Reset error: {str(e)}")
    
    def set_backlight(self, enable):
        """Set backlight compensation"""
        try:
            self.camera.backlight(enable)
            self.status_var.set(f"Backlight {'enabled' if enable else 'disabled'}")
        except Exception as e:
            self.status_var.set(f"Backlight error: {str(e)}")
    
    def set_white_balance_mode(self, mode):
        """Set white balance mode"""
        try:
            if hasattr(self.camera.camera, 'white_balance_mode'):
                self.camera.camera.white_balance_mode(mode)
            self.status_var.set(f"White balance: {mode}")
        except Exception as e:
            self.status_var.set(f"WB error: {str(e)}")
    
    def set_focus_mode(self, auto_focus):
        """Set focus mode"""
        try:
            if auto_focus:
                self.camera.camera.set_autofocus_mode('auto')
            else:
                self.camera.camera.manual_focus()
            self.status_var.set(f"Focus: {'Auto' if auto_focus else 'Manual'}")
        except Exception as e:
            self.status_var.set(f"Focus error: {str(e)}")
    
    def set_iris_mode(self, auto_iris):
        """Set iris mode"""
        try:
            # Most visca_over_ip libraries handle iris through auto exposure
            if hasattr(self.camera.camera, 'iris_auto') and hasattr(self.camera.camera, 'iris_manual'):
                if auto_iris:
                    self.camera.camera.iris_auto()
                else:
                    self.camera.camera.iris_manual()
            self.status_var.set(f"Iris: {'Auto' if auto_iris else 'Manual'}")
        except Exception as e:
            self.status_var.set(f"Iris error: {str(e)}")
    
    def adjust_gain(self, direction):
        """Adjust camera gain"""
        try:
            # Gain adjustment through visca_over_ip library
            if hasattr(self.camera.camera, 'gain_up') and hasattr(self.camera.camera, 'gain_down'):
                if direction == "up":
                    self.camera.camera.gain_up()
                else:
                    self.camera.camera.gain_down()
            self.status_var.set(f"Gain adjusted {direction}")
        except Exception as e:
            self.status_var.set(f"Gain error: {str(e)}")
    
    def flip_horizontal(self, enable):
        """Enable/disable horizontal flip"""
        try:
            # Use the X1Visca wrapper method
            self.camera.flip_horizontal(enable)
            self.status_var.set(f"Horizontal flip {'enabled' if enable else 'disabled'}")
        except Exception as e:
            self.status_var.set(f"Horizontal flip error: {str(e)}")
    
    def flip_vertical(self, enable):
        """Enable/disable vertical flip"""
        try:
            # Use the X1Visca wrapper method
            self.camera.flip_vertical(enable)
            self.status_var.set(f"Vertical flip {'enabled' if enable else 'disabled'}")
        except Exception as e:
            self.status_var.set(f"Vertical flip error: {str(e)}")
    
    def set_preset_position(self, preset_num):
        """Set preset position"""
        try:
            self.camera.save_preset(preset_num)
            self.status_var.set(f"Preset {preset_num} saved")
        except Exception as e:
            self.status_var.set(f"Preset error: {str(e)}")
    
    def call_preset_position(self, preset_num):
        """Call preset position"""
        try:
            self.camera.recall_preset(preset_num)
            self.status_var.set(f"Called preset {preset_num}")
        except Exception as e:
            self.status_var.set(f"Preset error: {str(e)}")
    
    def set_pan_tilt_position(self, pan_speed, tilt_speed, pan_pos, tilt_pos):
        """Set absolute pan/tilt position"""
        try:
            # Most visca_over_ip libraries have absolute positioning
            if hasattr(self.camera.camera, 'pan_tilt_absolute'):
                self.camera.camera.pan_tilt_absolute(pan_pos, tilt_pos, pan_speed, tilt_speed)
            self.status_var.set(f"Moving to pan: {pan_pos}°, tilt: {tilt_pos}°")
        except Exception as e:
            self.status_var.set(f"Position error: {str(e)}")
    
    def auto_exposure_on(self):
        """Enable auto exposure"""
        try:
            if hasattr(self.camera.camera, 'autoexposure_mode'):
                self.camera.camera.autoexposure_mode('auto')
            self.status_var.set("Auto exposure enabled")
        except Exception as e:
            self.status_var.set(f"AE error: {str(e)}")
    
    def start_video_stream(self):
        """Start the NDI video stream in a separate thread"""
        self.video_running = True
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        self.status_var.set("Starting video stream...")
    
    def video_loop(self):
        """Main video streaming loop"""
        try:
            # Create OpenCV window
            cv2.namedWindow("BirdDog X1 Live Feed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("BirdDog X1 Live Feed", 960, 540)
            
            for frame in ndi_frames(NDI_NAME):
                if not self.video_running:
                    break
                
                # Display frame in OpenCV window
                cv2.imshow("BirdDog X1 Live Feed", frame)
                
                # Update connection status
                self.root.after(0, lambda: self.connection_var.set("Camera: Connected | NDI: Connected"))
                self.root.after(0, lambda: self.video_status_label.config(text="✅ Live video active in separate window", fg="green"))
                
                # Check for window close or ESC key
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                
                # Check if window was closed
                if cv2.getWindowProperty("BirdDog X1 Live Feed", cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.status_var.set(f"Video error: {error_msg}"))
            self.root.after(0, lambda: self.connection_var.set("Camera: Unknown | NDI: Disconnected"))
            self.root.after(0, lambda: self.video_status_label.config(text="❌ Video connection failed", fg="red"))
        finally:
            cv2.destroyWindow("BirdDog X1 Live Feed")
    
    def stop_video_stream(self):
        """Stop the video stream and close OpenCV windows"""
        print("Stopping video stream...")
        
        # Set flag to stop video loop
        self.video_running = False
        
        # Wait for video thread to finish
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.is_alive():
            print("Waiting for video thread to finish...")
            self.video_thread.join(timeout=3)  # Increased timeout
            if self.video_thread.is_alive():
                print("Video thread did not finish in time")
        
        # Force close all OpenCV windows
        try:
            cv2.destroyWindow("BirdDog X1 Live Feed")
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process any pending window events
            print("OpenCV windows closed")
        except Exception as e:
            print(f"Error closing OpenCV windows: {e}")
        
        # Update GUI status
        try:
            self.connection_var.set("Camera: Disconnected | NDI: Disconnected")
            self.video_status_label.config(text="❌ Video stream stopped", fg="red")
        except Exception as e:
            print(f"Error updating GUI status: {e}")
    
    def on_closing(self):
        """Handle application closing - gracefully close both GUI and video windows"""
        try:
            print("Closing application...")
            
            # Stop video stream first
            self.stop_video_stream()
            
            # Stop camera movement and close camera connection
            if hasattr(self, 'camera') and self.camera:
                try:
                    self.camera.stop()
                    self.camera.close()
                    print("Camera connection closed")
                except Exception as e:
                    print(f"Error closing camera: {e}")
            
            # Force close any remaining OpenCV windows
            cv2.destroyAllWindows()
            
            # Small delay to ensure cleanup
            import time
            time.sleep(0.1)
            
            print("Application closed successfully")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            # Destroy the main window
            self.root.destroy()

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    try:
        app = BirdDogGUI(root)
        
        # Connect the close handler to window close events
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Also handle Ctrl+C and other termination signals if possible
        def signal_handler():
            app.on_closing()
        
        # Bind escape key to close as well
        root.bind('<Escape>', lambda e: app.on_closing())
        
        # Set window to be resizable
        root.resizable(True, True)
        
        # Start the main loop
        root.mainloop()
        
    except KeyboardInterrupt:
        print("Application interrupted by user")
        if 'app' in locals():
            app.on_closing()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start application: {str(e)}")
        # Ensure cleanup even if there's an error
        if 'app' in locals():
            app.on_closing()

if __name__ == "__main__":
    main()
