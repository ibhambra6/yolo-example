"""
birdtrack.py – drive a BirdDog X1 so it automatically keeps a YOLO-detected
object in the centre of frame.

▶ Prereqs  (Windows / macOS / Linux + Python ≥ 3.9)
---------------------------------------------------
pip install ultralytics opencv-python numpy visca_over_ip
# NewTek NDI runtime + Python wrapper:
pip install ndi-python               # thin wrapper over official NDI SDK
                                      #  ➜ https://github.com/buresu/ndi-python
"""
import cv2, numpy as np, time, math, threading
from ultralytics import YOLO
import NDIlib as ndi   # noqa:  pip install ndi-python
from visca_over_ip import Camera

### ------------------- USER CONFIG -------------------
CAMERA_IP   = "192.168.0.13"       #  X1 address
NDI_NAME    = "CAM2"         #  NDI stream name advertised by the camera
VISCA_PORT  = 52381                   #  fixed for BirdDog/Sony VISCA-IP
MODEL_PATH  = "best.pt"               #  your trained YOLO model
TRACK_CLASS = 0                       #  class-id you want to track
FOV_H_DEG   = 55.8                    #  X1 widest horizontal FoV; used for angle calc
DEADBAND    = 0.05                    #  centre dead-zone (±5 % of frame)
PT_SPEED_MIN, PT_SPEED_MAX = 0x01, 0x18  # VISCA velocity range (1-24)
### ----------------------------------------------------

# --------------------------------------------------------------------------- #
# 1. VISCA-over-IP wrapper using visca_over_ip library
# --------------------------------------------------------------------------- #
class X1Visca:
    """VISCA-IP client wrapper for BirdDog X1 camera using visca_over_ip library."""
    def __init__(self, ip:str=CAMERA_IP, port:int=VISCA_PORT):
        self.camera = Camera(ip)
        self.ip = ip
        self.port = port
        print(f"VISCA client initialized for {ip}:{port}")

    def move(self, pan_vel:int, tilt_vel:int, pan_dir:str, tilt_dir:str):
        """
        Continuous move using visca_over_ip library.
        pan_vel: 1-24, tilt_vel: 1-20   (X1 spec)
        """
        try:
            # Convert direction and velocity to pantilt parameters
            pan_speed = pan_vel if pan_dir in ["L", "R"] else 0
            tilt_speed = tilt_vel if tilt_dir in ["U", "D"] else 0
            
            # Convert directions to signed values
            pan_direction = -pan_speed if pan_dir == "L" else pan_speed if pan_dir == "R" else 0
            tilt_direction = tilt_speed if tilt_dir == "U" else -tilt_speed if tilt_dir == "D" else 0
            
            self.camera.pantilt(pan_direction, tilt_direction)
            print(f"VISCA move command sent: pan_dir={pan_dir}, tilt_dir={tilt_dir}, pan_vel={pan_vel}, tilt_vel={tilt_vel}")
        except Exception as e:
            print(f"VISCA move error: {e}")
            raise

    def stop(self):
        """Stop all pan/tilt movement."""
        try:
            self.camera.pantilt(0, 0)
            print("VISCA stop command sent")
        except Exception as e:
            print(f"VISCA stop error: {e}")
    
    def zoom_in(self, speed:int=5):
        """Zoom in with specified speed."""
        try:
            self.camera.zoom(speed)
            print(f"VISCA zoom in command sent: speed={speed}")
        except Exception as e:
            print(f"VISCA zoom in error: {e}")
    
    def zoom_out(self, speed:int=5):
        """Zoom out with specified speed."""
        try:
            self.camera.zoom(-speed)
            print(f"VISCA zoom out command sent: speed={speed}")
        except Exception as e:
            print(f"VISCA zoom out error: {e}")
    
    def zoom_stop(self):
        """Stop zoom operation."""
        try:
            self.camera.zoom(0)
            print("VISCA zoom stop command sent")
        except Exception as e:
            print(f"VISCA zoom stop error: {e}")
    
    def home(self):
        """Return to home position."""
        try:
            self.camera.pantilt_home()
            print("VISCA home command sent")
        except Exception as e:
            print(f"VISCA home error: {e}")
    
    def reset(self):
        """Reset camera."""
        try:
            # Use pantilt_reset if available, otherwise fall back to home
            if hasattr(self.camera, 'pantilt_reset'):
                self.camera.pantilt_reset()
            else:
                self.camera.pantilt_home()
            print("VISCA reset command sent")
        except Exception as e:
            print(f"VISCA reset error: {e}")
    
    def set_preset(self, preset_num:int):
        """Set preset position."""
        try:
            self.camera.save_preset(preset_num)
            print(f"VISCA set preset {preset_num} command sent")
        except Exception as e:
            print(f"VISCA set preset error: {e}")
    
    def call_preset(self, preset_num:int):
        """Call preset position."""
        try:
            self.camera.recall_preset(preset_num)
            print(f"VISCA call preset {preset_num} command sent")
        except Exception as e:
            print(f"VISCA call preset error: {e}")
    
    def set_backlight(self, enable:bool):
        """Set backlight compensation."""
        try:
            self.camera.backlight(enable)
            print(f"VISCA backlight {'enabled' if enable else 'disabled'}")
        except Exception as e:
            print(f"VISCA backlight error: {e}")
    
    def set_white_balance(self, mode:str):
        """Set white balance mode."""
        try:
            if hasattr(self.camera, 'white_balance_mode'):
                self.camera.white_balance_mode(mode)
            print(f"VISCA white balance set to {mode}")
        except Exception as e:
            print(f"VISCA white balance error: {e}")
    
    def set_focus_auto(self, auto:bool):
        """Set focus mode."""
        try:
            if auto:
                self.camera.set_autofocus_mode('auto')
            else:
                self.camera.manual_focus()
            print(f"VISCA focus set to {'auto' if auto else 'manual'}")
        except Exception as e:
            print(f"VISCA focus error: {e}")
    
    def auto_exposure(self):
        """Enable full auto exposure."""
        try:
            if hasattr(self.camera, 'autoexposure_mode'):
                self.camera.autoexposure_mode('auto')
            print("VISCA auto exposure enabled")
        except Exception as e:
            print(f"VISCA auto exposure error: {e}")

    def picture_flip(self, enable:bool):
        """Enable/disable picture flip."""
        try:
            # Use the correct flip method signature: flip(horizontal: bool, vertical: bool)
            if hasattr(self.camera, 'flip'):
                # Enable both horizontal and vertical flip when enable=True
                self.camera.flip(horizontal=enable, vertical=enable)
                print(f"VISCA picture flip {'enabled' if enable else 'disabled'} (horizontal and vertical)")
            elif hasattr(self.camera, 'flip_vertical'):
                # Use vertical flip as fallback
                self.camera.flip_vertical(flip_mode=enable)
                print(f"VISCA vertical flip {'enabled' if enable else 'disabled'}")
            else:
                print("VISCA flip methods not available")
        except Exception as e:
            print(f"VISCA picture flip error: {e}")
    
    def flip_horizontal(self, enable:bool):
        """Enable/disable horizontal flip only."""
        try:
            if hasattr(self.camera, 'flip_horizontal'):
                self.camera.flip_horizontal(flip_mode=enable)
                print(f"VISCA horizontal flip {'enabled' if enable else 'disabled'}")
            elif hasattr(self.camera, 'flip'):
                self.camera.flip(horizontal=enable, vertical=False)
                print(f"VISCA horizontal flip {'enabled' if enable else 'disabled'}")
        except Exception as e:
            print(f"VISCA horizontal flip error: {e}")
    
    def flip_vertical(self, enable:bool):
        """Enable/disable vertical flip only."""
        try:
            if hasattr(self.camera, 'flip_vertical'):
                self.camera.flip_vertical(flip_mode=enable)
                print(f"VISCA vertical flip {'enabled' if enable else 'disabled'}")
            elif hasattr(self.camera, 'flip'):
                self.camera.flip(horizontal=False, vertical=enable)
                print(f"VISCA vertical flip {'enabled' if enable else 'disabled'}")
        except Exception as e:
            print(f"VISCA vertical flip error: {e}")

    def close(self):
        """Close the connection"""
        try:
            if hasattr(self.camera, 'close_connection'):
                self.camera.close_connection()
            print("VISCA connection cleanup")
        except Exception as e:
            print(f"Error during VISCA cleanup: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()

# --------------------------------------------------------------------------- #
# 2. NDI receiver  ➜ OpenCV BGR frame
# --------------------------------------------------------------------------- #
def ndi_frames(source_name:str):
    finder   = ndi.find_create_v2()
    recv     = ndi.recv_create_v3()
    source   = None
    while True:
        if source is None:
            # wait until the announced name appears on the network
            for s in ndi.find_get_current_sources(finder):
                ndi_name = s.ndi_name.decode() if hasattr(s.ndi_name, 'decode') else str(s.ndi_name)
                if source_name in ndi_name:
                    source = s
                    ndi.recv_connect(recv, source)
                    break
            time.sleep(0.25)
            continue

        frame_type, video_frame, audio_frame, metadata = ndi.recv_capture_v2(recv, timeout_in_ms=1000)
        if frame_type != ndi.FRAME_TYPE_VIDEO or video_frame.data is None:
            continue
        
        try:
            h, w = video_frame.yres, video_frame.xres
            line_stride = video_frame.line_stride_in_bytes
            
            # Get the raw frame data
            frame_data = np.frombuffer(video_frame.data, dtype=np.uint8)
            
            # Calculate expected dimensions
            total_bytes = len(frame_data)
            expected_bytes_4ch = h * w * 4  # BGRA/RGBA
            expected_bytes_3ch = h * w * 3  # BGR/RGB
            expected_bytes_2ch = h * w * 2  # UYVY/YUV422
            
            print(f"Frame: {w}x{h}, stride: {line_stride}, data: {total_bytes} bytes")
            
            # Try different formats based on data size
            if total_bytes == expected_bytes_4ch:
                # 4-channel format (likely BGRA from NDI)
                img = frame_data.reshape((h, w, 4))
                # NDI typically uses BGRA format, convert to BGR for OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif total_bytes == expected_bytes_3ch:
                # 3-channel format 
                img = frame_data.reshape((h, w, 3))
                # Check if it's RGB and convert to BGR for OpenCV
                # NDI might send RGB, but OpenCV expects BGR
                if video_frame.FourCC == ndi.FOURCC_TYPE_RGBX or video_frame.FourCC == ndi.FOURCC_TYPE_RGBA:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # Otherwise assume it's already BGR
            elif total_bytes == expected_bytes_2ch:
                # UYVY format (common in NDI) - convert properly to color
                img = frame_data.reshape((h, w, 2))
                # Convert UYVY to BGR using OpenCV
                img_uyvy = np.zeros((h, w, 2), dtype=np.uint8)
                img_uyvy[:, :, 0] = img[:, :, 0]  # U
                img_uyvy[:, :, 1] = img[:, :, 1]  # Y
                # Create a proper UYVY image and convert to BGR
                uyvy_reshaped = img.astype(np.uint8)
                try:
                    img = cv2.cvtColor(uyvy_reshaped, cv2.COLOR_YUV2BGR_UYVY)
                except:
                    # Fallback: convert to grayscale then to color
                    y_channel = img[:, :, 1]  # Y (luminance) 
                    img = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
            elif line_stride > 0 and w > 0:
                # Use line stride to determine format
                bytes_per_pixel = line_stride // w
                if bytes_per_pixel > 0 and total_bytes >= h * line_stride:
                    img = frame_data[:h * line_stride].reshape((h, line_stride))
                    # Extract the actual pixel data
                    img = img[:, :w*bytes_per_pixel].reshape((h, w, bytes_per_pixel))
                    if bytes_per_pixel == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    elif bytes_per_pixel == 3:
                        # Assume RGB and convert to BGR
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif bytes_per_pixel == 2:
                        # UYVY handling with proper color conversion
                        try:
                            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_YUV2BGR_UYVY)
                        except:
                            y_channel = img[:, :, 1]
                            img = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
                    else:
                        img = cv2.cvtColor(img[:,:,0], cv2.COLOR_GRAY2BGR)
                else:
                    raise ValueError("Cannot determine frame format from line stride")
            else:
                # Last resort: try different possibilities
                possible_formats = [
                    (h, w, 4),  # BGRA
                    (h, w, 3),  # BGR/RGB
                    (h, w, 2),  # UYVY
                    (h, w, 1),  # Grayscale
                ]
                
                img = None
                for ph, pw, pc in possible_formats:
                    if total_bytes >= ph * pw * pc:
                        try:
                            temp_img = frame_data[:ph*pw*pc].reshape((ph, pw, pc))
                            if pc == 4:
                                img = cv2.cvtColor(temp_img, cv2.COLOR_BGRA2BGR)
                            elif pc == 3:
                                # Try RGB to BGR conversion
                                img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
                            elif pc == 2:
                                # Proper UYVY to BGR conversion
                                try:
                                    img = cv2.cvtColor(temp_img.astype(np.uint8), cv2.COLOR_YUV2BGR_UYVY)
                                except:
                                    y_channel = temp_img[:, :, 1]
                                    img = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
                            elif pc == 1:
                                img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
                            break
                        except:
                            continue
                
                if img is None:
                    raise ValueError(f"Cannot process frame: {total_bytes} bytes for {h}x{w} image")
            
            # Final validation
            if len(img.shape) != 3 or img.shape[2] != 3:
                print(f"Warning: Unexpected image shape {img.shape}, attempting conversion")
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = img[:, :, :3]  # Take first 3 channels
                else:
                    img = img[:, :, :3]  # Take first 3 channels if more than 3
            
        except Exception as e:
            print(f"Video frame processing error: {e}")
            print(f"Frame info: {h}x{w}, stride: {line_stride}, data size: {len(frame_data)}")
            ndi.recv_free_video_v2(recv, video_frame)
            continue
        
        yield img
        ndi.recv_free_video_v2(recv, video_frame)

# --------------------------------------------------------------------------- #
# 3. Main loop – detection ➜ PTZ correction
# --------------------------------------------------------------------------- #
def track():
    detector = YOLO(MODEL_PATH)
    cam      = X1Visca()
    for frame in ndi_frames(NDI_NAME):
        h, w, _ = frame.shape
        # run detector (no tracker → lowest latency)
        result = detector(frame, verbose=False, conf=0.25)[0]
        boxes  = [b for b in result.boxes if int(b.cls)==TRACK_CLASS]
        if not boxes:
            cam.stop()
            cv2.imshow("BirdTrack", frame); cv2.waitKey(1)
            continue
        # pick the highest-confidence box
        box = max(boxes, key=lambda b: float(b.conf))
        x0,y0,x1,y1 = map(int, box.xyxy[0])
        cx, cy = (x0+x1)//2, (y0+y1)//2
        # normalised offset from centre (-1 … +1)
        off_x, off_y = ( (cx - w/2)/(w/2), (cy - h/2)/(h/2) )
        # draw HUD
        cv2.circle(frame, (cx,cy), 6, (0,255,0), 2)
        cv2.line(frame, (w//2, h//2), (cx,cy), (0,255,0), 1)
        cv2.rectangle(frame, (x0,y0), (x1,y1), (255,0,0), 2)
        cv2.putText(frame, f"off=({off_x:+.2f},{off_y:+.2f})", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # decide move
        if abs(off_x) < DEADBAND and abs(off_y) < DEADBAND:
            cam.stop()
        else:
            # speed proportional to angle error (clamped)
            #   FoV→deg: angle ≈ off_x * (FoV/2)
            ang_x = off_x * (FOV_H_DEG/2)
            speed = min(max(int(abs(ang_x) / (FOV_H_DEG/2) * PT_SPEED_MAX), 1), 20)
            pan_dir = "R" if off_x>0 else "L"
            tilt_dir= "U" if off_y<0 else "D"
            cam.move(speed, speed, pan_dir, tilt_dir)

        cv2.imshow("BirdTrack", frame)
        if cv2.waitKey(1) in (27, ord('q')):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        track()
    finally:
        X1Visca().stop()
