"""
BirdDog X1 Camera Control GUI

A graphical interface for controlling the BirdDog X1 camera with live NDI feed display
and pan/tilt/zoom controls.

Requirements:
- All dependencies from birddog.py
- tkinter (usually included with Python)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageTk
import socket

# Import NDI and camera control from the main module
import ndi
from birddog import X1Visca, ndi_frames

### ------------------- GUI CONFIG -------------------
CAMERA_IP   = "192.168.100.150"       # X1 address
NDI_NAME    = "X1-Studio-Cam"         # NDI stream name
VISCA_PORT  = 52381                   # VISCA-IP port
WINDOW_WIDTH = 1280                   # GUI window width
WINDOW_HEIGHT = 800                   # GUI window height
VIDEO_WIDTH = 960                     # Video display width
VIDEO_HEIGHT = 540                    # Video display height
### -------------------------------------------------

class BirdDogGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BirdDog X1 Camera Control")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        
        # Initialize camera control
        self.camera = X1Visca(CAMERA_IP)
        
        # Video streaming variables
        self.video_running = False
        self.video_thread = None
        self.current_frame = None
        
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
        
        # Video display area
        self.video_frame = ttk.LabelFrame(main_frame, text="Live Feed", padding="5")
        self.video_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.video_label = tk.Label(self.video_frame, bg="black", width=VIDEO_WIDTH//8, height=VIDEO_HEIGHT//16)
        self.video_label.pack(expand=True)
        
        # Control panels
        self.setup_pan_tilt_controls(main_frame)
        self.setup_zoom_controls(main_frame)
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
    
    def setup_status_panel(self, parent):
        """Create status display panel"""
        status_frame = ttk.LabelFrame(parent, text="Status", padding="10")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
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
                self.camera.move(pan_vel, tilt_vel, pan_dir, tilt_dir)
                self.status_var.set(f"Moving: {direction.replace('_', ' ').title()}")
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
    
    def stop_move(self):
        """Stop all movement"""
        if not self.is_moving:
            return
            
        self.is_moving = False
        try:
            self.camera.stop()
            self.status_var.set("Stopped")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
    
    def start_zoom(self, direction):
        """Start zoom operation"""
        speed = self.zoom_speed_var.get()
        try:
            if direction == "in":
                # VISCA zoom in command
                payload = bytes([0x81, 0x01, 0x04, 0x07, 0x20 + speed, 0xFF])
            else:
                # VISCA zoom out command  
                payload = bytes([0x81, 0x01, 0x04, 0x07, 0x30 + speed, 0xFF])
            
            self.camera._send(payload)
            self.status_var.set(f"Zooming {direction}")
        except Exception as e:
            self.status_var.set(f"Zoom error: {str(e)}")
    
    def stop_zoom(self):
        """Stop zoom operation"""
        try:
            # VISCA zoom stop command
            payload = bytes([0x81, 0x01, 0x04, 0x07, 0x00, 0xFF])
            self.camera._send(payload)
            self.status_var.set("Zoom stopped")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
    
    def go_home(self):
        """Return camera to home position"""
        try:
            # VISCA home position command
            payload = bytes([0x81, 0x01, 0x06, 0x04, 0xFF])
            self.camera._send(payload)
            self.status_var.set("Returning to home position")
        except Exception as e:
            self.status_var.set(f"Home error: {str(e)}")
    
    def start_video_stream(self):
        """Start the NDI video stream in a separate thread"""
        self.video_running = True
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        self.status_var.set("Starting video stream...")
    
    def video_loop(self):
        """Main video streaming loop"""
        try:
            for frame in ndi_frames(NDI_NAME):
                if not self.video_running:
                    break
                
                # Resize frame for display
                frame_resized = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update the video display
                self.root.after(0, self.update_video_display, photo)
                
                # Update connection status
                self.root.after(0, lambda: self.connection_var.set("Camera: Connected | NDI: Connected"))
                
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.status_var.set(f"Video error: {error_msg}"))
            self.root.after(0, lambda: self.connection_var.set("Camera: Unknown | NDI: Disconnected"))
    
    def update_video_display(self, photo):
        """Update the video display with new frame"""
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # Keep a reference
        
        if self.status_var.get() == "Starting video stream...":
            self.status_var.set("Video stream active")
    
    def stop_video_stream(self):
        """Stop the video stream"""
        self.video_running = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2)
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_video_stream()
        try:
            self.camera.stop()
        except:
            pass
        self.root.destroy()

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    try:
        app = BirdDogGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()
