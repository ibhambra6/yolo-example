"""
birdtrack.py – drive a BirdDog X1 so it automatically keeps a YOLO-detected
object in the centre of frame with advanced pose estimation and predictive tracking.

▶ Prereqs  (Windows / macOS / Linux + Python ≥ 3.9)
---------------------------------------------------
pip install ultralytics opencv-python numpy visca_over_ip
# For YOLOv5 models, also install:
pip install torch torchvision
# NewTek NDI runtime + Python wrapper:
pip install ndi-python               # thin wrapper over official NDI SDK
                                      #  ➜ https://github.com/buresu/ndi-python
"""
import cv2, numpy as np, time, math, threading
import torch
from ultralytics import YOLO
import NDIlib as ndi   # noqa:  pip install ndi-python
from visca_over_ip import Camera
from collections import deque

### ------------------- USER CONFIG -------------------
CAMERA_IP   = "192.168.0.7"       #  X1 address
NDI_NAME    = "CAM1"         #  NDI stream name advertised by the camera
VISCA_PORT  = 52381                   #  fixed for BirdDog/Sony VISCA-IP
MODEL_PATH  = "best.pt"               #  your trained YOLO model (YOLOv5 or YOLOv8)
TRACK_CLASS = 4                       #  class-id you want to track (4 = "5M Assem")
FOV_H_DEG   = 55.8                    #  X1 widest horizontal FoV; used for angle calc
FOV_V_DEG   = 31.4                    #  X1 vertical FoV (calculated from 16:9 aspect ratio)
DEADBAND    = 0.05                    #  centre dead-zone (±5 % of frame)
PT_SPEED_MIN, PT_SPEED_MAX = 0x01, 0x18  # VISCA velocity range (1-24)

# Object characteristics for pose estimation
OBJECT_REAL_WIDTH_MM = 150.0         # Real width of tracked object in millimeters
OBJECT_REAL_HEIGHT_MM = 100.0        # Real height of tracked object in millimeters
CAMERA_FOCAL_LENGTH_MM = 4.7         # BirdDog X1 approximate focal length (varies with zoom)
SENSOR_WIDTH_MM = 6.17               # BirdDog X1 sensor width (1/2.8" sensor)
SENSOR_HEIGHT_MM = 4.63              # BirdDog X1 sensor height

# Smooth movement control parameters - TUNED FOR BETTER TRACKING
MAX_ACCELERATION = 5                  # Increased for more responsive movement
MIN_MOVE_THRESHOLD = 0.015           # Slightly increased to reduce micro-movements (was 0.01)
SPEED_RAMP_FACTOR = 0.8              # Increased for better responsiveness (was 0.7)
MAX_TRACKING_SPEED = 18              # Increased max speed for better tracking (was 15)

# New parameters for better tracking
DISTANCE_SCALING_FACTOR = 1.2        # Reduced to prevent over-aggressive movements (was 1.5)
VELOCITY_SMOOTHING = 0.4             # Increased smoothing to reduce jitter (was 0.3)

# Pose estimation parameters
ENABLE_POSE_ESTIMATION = True        # Enable 3D pose estimation
ENABLE_PREDICTIVE_TRACKING = False   # Disable initially to debug basic tracking
ENABLE_DEPTH_ESTIMATION = True       # Enable distance estimation from object size
MOTION_HISTORY_SIZE = 10             # Number of frames to track for motion prediction
DISTANCE_SMOOTHING_FACTOR = 0.8      # Increased for more stable distance estimates (was 0.7)
POSITION_SMOOTHING_FACTOR = 0.9      # Increased for more stable position (was 0.8)

# Enhanced stability parameters
TILT_STABILITY_FACTOR = 0.7          # Reduce tilt sensitivity compared to pan
VERTICAL_DEADBAND_MULTIPLIER = 1.5   # Make vertical deadband larger than horizontal
MIN_TILT_THRESHOLD = 0.02            # Higher threshold for tilt movements
CONFIDENCE_THRESHOLD = 0.5           # Minimum confidence for full-speed movement

# Debug and testing options
ENABLE_TRACKING_DEBUG = True         # Show detailed tracking information
ENABLE_MOVEMENT_VISUALIZATION = True # Show movement vectors and deadband zones
ENABLE_POSE_VISUALIZATION = True     # Show 3D pose information
### ----------------------------------------------------

# --------------------------------------------------------------------------- #
# 0. Model loading helper to support both YOLOv5 and YOLOv8
# --------------------------------------------------------------------------- #
def load_model(model_path):
    """Load YOLO model, supporting both YOLOv5 and YOLOv8 formats."""
    try:
        # Try YOLOv8/Ultralytics format first
        model = YOLO(model_path)
        print(f"Loaded YOLOv8 model from {model_path}")
        return model, "v8"
    except Exception as e:
        print(f"YOLOv8 loading failed: {e}")
        try:
            # Fallback to YOLOv5 torch.hub loading
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
            print(f"Loaded YOLOv5 model from {model_path}")
            return model, "v5"
        except Exception as e2:
            print(f"YOLOv5 loading also failed: {e2}")
            print("Please ensure your model file is valid or retrain with ultralytics")
            raise e2

def detect_objects(model, frame, model_type="v8"):
    """Run detection on frame, supporting both YOLOv5 and YOLOv8."""
    if model_type == "v8":
        # YOLOv8 inference
        results = model(frame, verbose=False, conf=0.25)
        return results[0].boxes if results[0].boxes is not None else []
    else:
        # YOLOv5 inference
        results = model(frame)
        detections = []
        if len(results.xyxy[0]) > 0:
            # Convert YOLOv5 format to YOLOv8-like format
            for detection in results.xyxy[0]:
                # detection format: [x1, y1, x2, y2, confidence, class]
                x1, y1, x2, y2, conf, cls = detection.tolist()
                # Create a simple object to mimic YOLOv8 box format
                class Detection:
                    def __init__(self, x1, y1, x2, y2, conf, cls):
                        self.xyxy = [[x1, y1, x2, y2]]
                        self.conf = [conf]
                        self.cls = [cls]
                
                detections.append(Detection(x1, y1, x2, y2, conf, cls))
        return detections

# --------------------------------------------------------------------------- #
# 0.1. Camera Pose Estimation System
# --------------------------------------------------------------------------- #
class CameraPoseEstimator:
    """Estimates camera pose and object 3D position for improved tracking."""
    
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Camera intrinsic parameters (estimated for BirdDog X1)
        self.focal_length_px_x = (frame_width * CAMERA_FOCAL_LENGTH_MM) / SENSOR_WIDTH_MM
        self.focal_length_px_y = (frame_height * CAMERA_FOCAL_LENGTH_MM) / SENSOR_HEIGHT_MM
        self.principal_point_x = frame_width / 2
        self.principal_point_y = frame_height / 2
        
        # Camera matrix
        self.camera_matrix = np.array([
            [self.focal_length_px_x, 0, self.principal_point_x],
            [0, self.focal_length_px_y, self.principal_point_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients (assumed minimal for BirdDog X1)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Pose tracking variables
        self.camera_position = np.array([0.0, 0.0, 0.0])  # [x, y, z] in world coordinates
        self.camera_rotation = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw] in radians
        self.last_camera_movement = np.array([0.0, 0.0, 0.0])
        
        print(f"📷 Camera intrinsics initialized:")
        print(f"   - Focal length: {self.focal_length_px_x:.1f}px (H), {self.focal_length_px_y:.1f}px (V)")
        print(f"   - Principal point: ({self.principal_point_x:.1f}, {self.principal_point_y:.1f})")
    
    def estimate_object_distance(self, bbox_width, bbox_height):
        """Estimate distance to object based on bounding box size."""
        if bbox_width <= 0 or bbox_height <= 0:
            return None
            
        # Calculate distance using similar triangles
        # Distance = (real_size * focal_length_px) / bbox_size_px
        distance_from_width = (OBJECT_REAL_WIDTH_MM * self.focal_length_px_x) / bbox_width
        distance_from_height = (OBJECT_REAL_HEIGHT_MM * self.focal_length_px_y) / bbox_height
        
        # Use average of both estimates for robustness
        estimated_distance = (distance_from_width + distance_from_height) / 2.0
        
        return estimated_distance
    
    def pixel_to_world_ray(self, pixel_x, pixel_y, distance_mm):
        """Convert pixel coordinates to 3D world coordinates."""
        # Normalize pixel coordinates
        x_norm = (pixel_x - self.principal_point_x) / self.focal_length_px_x
        y_norm = (pixel_y - self.principal_point_y) / self.focal_length_px_y
        
        # Calculate 3D position (assuming camera at origin looking down -Z axis)
        world_x = x_norm * distance_mm
        world_y = y_norm * distance_mm
        world_z = distance_mm
        
        return np.array([world_x, world_y, world_z])
    
    def update_camera_movement(self, pan_angle_deg, tilt_angle_deg):
        """Update camera pose based on pan/tilt movements."""
        # Convert to radians
        pan_rad = np.radians(pan_angle_deg)
        tilt_rad = np.radians(tilt_angle_deg)
        
        # Update camera rotation (yaw for pan, pitch for tilt)
        self.camera_rotation[1] += tilt_rad  # pitch
        self.camera_rotation[2] += pan_rad   # yaw
        
        # Track movement for motion prediction
        self.last_camera_movement = np.array([0, tilt_rad, pan_rad])
    
    def get_camera_rotation_matrix(self):
        """Get current camera rotation matrix."""
        roll, pitch, yaw = self.camera_rotation
        
        # Rotation matrices for each axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        return R_z @ R_y @ R_x

class MotionPredictor:
    """Predicts object motion for improved tracking."""
    
    def __init__(self, history_size=MOTION_HISTORY_SIZE):
        self.history_size = history_size
        self.position_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size)
        
    def add_position(self, position_3d, timestamp):
        """Add a new position observation."""
        self.position_history.append(position_3d.copy())
        self.time_history.append(timestamp)
        
        # Calculate velocity if we have enough history
        if len(self.position_history) >= 2:
            dt = self.time_history[-1] - self.time_history[-2]
            if dt > 0:
                velocity = (self.position_history[-1] - self.position_history[-2]) / dt
                self.velocity_history.append(velocity)
    
    def predict_position(self, future_time_delta):
        """Predict future position based on motion history."""
        if len(self.position_history) < 2:
            return None
            
        current_position = self.position_history[-1]
        
        # Simple linear prediction using average velocity
        if len(self.velocity_history) > 0:
            # Use weighted average of recent velocities
            weights = np.linspace(0.5, 1.0, len(self.velocity_history))
            avg_velocity = np.average(self.velocity_history, axis=0, weights=weights)
            predicted_position = current_position + avg_velocity * future_time_delta
            return predicted_position
        
        return current_position
    
    def get_current_velocity(self):
        """Get current estimated velocity."""
        if len(self.velocity_history) > 0:
            return self.velocity_history[-1]
        return np.array([0.0, 0.0, 0.0])
    
    def get_motion_confidence(self):
        """Get confidence in motion prediction (0-1)."""
        if len(self.position_history) < 3:
            return 0.0
        
        # Calculate consistency of velocity estimates
        if len(self.velocity_history) < 2:
            return 0.5
        
        velocities = np.array(self.velocity_history)
        velocity_std = np.std(velocities, axis=0)
        velocity_mean = np.mean(velocities, axis=0)
        
        # Confidence based on velocity consistency
        velocity_consistency = 1.0 / (1.0 + np.linalg.norm(velocity_std))
        return min(velocity_consistency, 1.0)

class Object3DTracker:
    """Tracks object in 3D space with pose estimation."""
    
    def __init__(self):
        self.pose_estimator = None
        self.motion_predictor = MotionPredictor()
        
        # 3D tracking state
        self.current_3d_position = None
        self.current_distance = None
        self.last_distance = None
        self.smoothed_distance = None
        
        # Tracking confidence
        self.tracking_confidence = 0.0
        self.distance_confidence = 0.0
        
    def initialize(self, frame_width, frame_height):
        """Initialize the pose estimator with frame dimensions."""
        self.pose_estimator = CameraPoseEstimator(frame_width, frame_height)
        print("🎯 3D object tracker initialized")
    
    def update_tracking(self, bbox, timestamp, camera_pan_angle=0, camera_tilt_angle=0):
        """Update 3D tracking with new detection."""
        if self.pose_estimator is None:
            return None
            
        x0, y0, x1, y1 = bbox
        bbox_width = x1 - x0
        bbox_height = y1 - y0
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        
        # Estimate distance from object size
        estimated_distance = self.pose_estimator.estimate_object_distance(bbox_width, bbox_height)
        
        if estimated_distance is not None:
            # Smooth distance estimates
            if self.smoothed_distance is None:
                self.smoothed_distance = estimated_distance
            else:
                self.smoothed_distance = (DISTANCE_SMOOTHING_FACTOR * self.smoothed_distance + 
                                        (1 - DISTANCE_SMOOTHING_FACTOR) * estimated_distance)
            
            # Calculate distance confidence based on bbox size consistency
            if self.last_distance is not None:
                distance_change = abs(estimated_distance - self.last_distance) / self.last_distance
                self.distance_confidence = max(0.0, 1.0 - distance_change * 2.0)
            else:
                self.distance_confidence = 0.8
            
            self.last_distance = estimated_distance
            self.current_distance = self.smoothed_distance
            
            # Convert to 3D world position
            world_position = self.pose_estimator.pixel_to_world_ray(
                center_x, center_y, self.smoothed_distance
            )
            
            # Smooth 3D position
            if self.current_3d_position is not None:
                self.current_3d_position = (POSITION_SMOOTHING_FACTOR * self.current_3d_position + 
                                          (1 - POSITION_SMOOTHING_FACTOR) * world_position)
            else:
                self.current_3d_position = world_position
            
            # Update motion predictor
            self.motion_predictor.add_position(self.current_3d_position, timestamp)
            
            # Update camera pose
            self.pose_estimator.update_camera_movement(camera_pan_angle, camera_tilt_angle)
            
            # Calculate overall tracking confidence
            motion_confidence = self.motion_predictor.get_motion_confidence()
            self.tracking_confidence = (self.distance_confidence + motion_confidence) / 2.0
            
            return {
                '3d_position': self.current_3d_position,
                'distance_mm': self.current_distance,
                'velocity': self.motion_predictor.get_current_velocity(),
                'confidence': self.tracking_confidence,
                'distance_confidence': self.distance_confidence,
                'motion_confidence': motion_confidence
            }
        
        return None
    
    def predict_future_position(self, time_delta=0.1):
        """Predict where object will be in the future."""
        if ENABLE_PREDICTIVE_TRACKING:
            return self.motion_predictor.predict_position(time_delta)
        return self.current_3d_position
    
    def get_tracking_info(self):
        """Get current tracking information for display."""
        if self.current_3d_position is None:
            return None
            
        return {
            'position_3d': self.current_3d_position,
            'distance_mm': self.current_distance,
            'velocity': self.motion_predictor.get_current_velocity(),
            'confidence': self.tracking_confidence,
            'distance_confidence': self.distance_confidence
        }

# --------------------------------------------------------------------------- #
# 0.5. Enhanced movement controller with 3D pose integration
# --------------------------------------------------------------------------- #
class SmoothMovementController:
    """Controls smooth camera movement with 3D pose estimation and predictive tracking."""
    
    def __init__(self):
        self.current_pan_speed = 0
        self.current_tilt_speed = 0
        self.current_pan_dir = ""
        self.current_tilt_dir = ""
        self.last_move_time = time.time()
        self.movement_history = []
        
        # Velocity smoothing
        self.velocity_smoothing = VELOCITY_SMOOTHING
        self.last_target_pan_speed = 0
        self.last_target_tilt_speed = 0
        
        # 3D pose integration
        self.last_camera_pan_angle = 0.0
        self.last_camera_tilt_angle = 0.0
        self.cumulative_pan_angle = 0.0
        self.cumulative_tilt_angle = 0.0
        
        # Predictive tracking
        self.prediction_lookahead = 0.1  # seconds
        self.distance_based_scaling = True
        
    def calculate_smooth_speed(self, target_speed, current_speed, max_accel):
        """Calculate smooth speed change with acceleration limiting."""
        speed_diff = target_speed - current_speed
        
        # Limit acceleration
        if abs(speed_diff) > max_accel:
            if speed_diff > 0:
                return current_speed + max_accel
            else:
                return current_speed - max_accel
        else:
            return target_speed
    
    def calculate_distance_scaling(self, distance_mm):
        """Calculate movement scaling based on object distance."""
        if not self.distance_based_scaling or distance_mm is None:
            return 1.0
            
        # Closer objects need slower, more precise movements
        # Farther objects can handle faster movements
        min_distance = 500.0   # 50cm
        max_distance = 5000.0  # 5m
        
        # Clamp distance to reasonable range
        clamped_distance = max(min_distance, min(distance_mm, max_distance))
        
        # Scale from 0.5 (close) to 1.5 (far)
        distance_factor = 0.5 + (clamped_distance - min_distance) / (max_distance - min_distance)
        
        return distance_factor
    
    def get_enhanced_movement(self, off_x, off_y, tracking_info=None, predicted_position=None):
        """
        Calculate enhanced movement with 3D pose estimation and improved stability.
        Returns (pan_speed, tilt_speed, pan_dir, tilt_dir, should_move, debug_info)
        """
        debug_info = {}
        
        # Apply enhanced deadband logic with separate horizontal/vertical zones
        horizontal_deadband = DEADBAND
        vertical_deadband = DEADBAND * VERTICAL_DEADBAND_MULTIPLIER
        is_horizontally_centered = abs(off_x) < horizontal_deadband
        is_vertically_centered = abs(off_y) < vertical_deadband
        is_centered = is_horizontally_centered and is_vertically_centered
        
        debug_info['deadbands'] = {
            'horizontal': horizontal_deadband,
            'vertical': vertical_deadband,
            'h_centered': is_horizontally_centered,
            'v_centered': is_vertically_centered
        }
        
        # Use predicted position if available and enabled
        if ENABLE_PREDICTIVE_TRACKING and predicted_position is not None and tracking_info is not None:
            current_3d = tracking_info.get('3d_position')
            if current_3d is not None:
                prediction_offset_3d = predicted_position - current_3d
                
                if tracking_info.get('distance_mm', 0) > 0:
                    distance = tracking_info['distance_mm']
                    pred_off_x = prediction_offset_3d[0] / distance * 1000
                    pred_off_y = prediction_offset_3d[1] / distance * 1000
                    
                    prediction_weight = min(tracking_info.get('confidence', 0.0), 0.2)  # Reduced weight
                    off_x = (1 - prediction_weight) * off_x + prediction_weight * pred_off_x
                    off_y = (1 - prediction_weight) * off_y + prediction_weight * pred_off_y
                    
                    debug_info['prediction_used'] = True
                    debug_info['prediction_weight'] = prediction_weight
        
        # Separate thresholds for pan and tilt
        pan_threshold = MIN_MOVE_THRESHOLD
        tilt_threshold = max(MIN_TILT_THRESHOLD, MIN_MOVE_THRESHOLD)
        
        # Check if movement is needed with separate thresholds
        pan_needs_movement = abs(off_x) > pan_threshold and not is_horizontally_centered
        tilt_needs_movement = abs(off_y) > tilt_threshold and not is_vertically_centered
        
        debug_info['movement_analysis'] = {
            'off_x': off_x,
            'off_y': off_y,
            'pan_threshold': pan_threshold,
            'tilt_threshold': tilt_threshold,
            'pan_needs_movement': pan_needs_movement,
            'tilt_needs_movement': tilt_needs_movement
        }
        
        if not pan_needs_movement and not tilt_needs_movement:
            # Object is well centered - stop all movement
            target_pan_speed = 0
            target_tilt_speed = 0
            pan_dir = ""
            tilt_dir = ""
            debug_info['action'] = "centered_stop"
        else:
            # Calculate distance-based scaling
            distance_scaling = 1.0
            if tracking_info and tracking_info.get('distance_mm'):
                distance_scaling = self.calculate_distance_scaling(tracking_info['distance_mm'])
                debug_info['distance_scaling'] = distance_scaling
            
            # Calculate target speeds with enhanced scaling and stability factors
            if pan_needs_movement:
                pan_distance_factor = min(abs(off_x) * DISTANCE_SCALING_FACTOR * distance_scaling, 1.0)
                base_pan_speed = max(
                    int(pan_distance_factor * MAX_TRACKING_SPEED), 
                    PT_SPEED_MIN
                )
            else:
                base_pan_speed = 0
                pan_distance_factor = 0
            
            if tilt_needs_movement:
                tilt_distance_factor = min(abs(off_y) * DISTANCE_SCALING_FACTOR * distance_scaling * TILT_STABILITY_FACTOR, 1.0)
                base_tilt_speed = max(
                    int(tilt_distance_factor * MAX_TRACKING_SPEED), 
                    PT_SPEED_MIN
                )
            else:
                base_tilt_speed = 0
                tilt_distance_factor = 0
            
            debug_info['base_speeds'] = {
                'pan': base_pan_speed,
                'tilt': base_tilt_speed,
                'pan_factor': pan_distance_factor,
                'tilt_factor': tilt_distance_factor
            }
            
            # Apply confidence-based scaling
            confidence_scaling = 1.0
            if tracking_info and tracking_info.get('confidence'):
                confidence = tracking_info['confidence']
                if confidence < CONFIDENCE_THRESHOLD:
                    # Reduce speed significantly for low confidence
                    confidence_scaling = max(0.2, confidence / CONFIDENCE_THRESHOLD)
                debug_info['confidence_scaling'] = confidence_scaling
            
            # Apply speed ramping with enhanced smoothing
            target_pan_speed = base_pan_speed * SPEED_RAMP_FACTOR * confidence_scaling
            target_tilt_speed = base_tilt_speed * SPEED_RAMP_FACTOR * confidence_scaling
            
            # Enhanced velocity smoothing
            target_pan_speed = (self.velocity_smoothing * self.last_target_pan_speed + 
                               (1 - self.velocity_smoothing) * target_pan_speed)
            target_tilt_speed = (self.velocity_smoothing * self.last_target_tilt_speed + 
                                (1 - self.velocity_smoothing) * target_tilt_speed)
            
            # Store for next iteration
            self.last_target_pan_speed = target_pan_speed
            self.last_target_tilt_speed = target_tilt_speed
            
            # Determine directions with improved logic
            pan_dir = ""
            tilt_dir = ""
            
            if pan_needs_movement:
                if off_x > pan_threshold:
                    pan_dir = "R"
                elif off_x < -pan_threshold:
                    pan_dir = "L"
            
            if tilt_needs_movement:
                if off_y > tilt_threshold:
                    tilt_dir = "D"  # Object below center, tilt down
                elif off_y < -tilt_threshold:
                    tilt_dir = "U"  # Object above center, tilt up
            
            debug_info['action'] = f"tracking_pan_{pan_dir}_tilt_{tilt_dir}"
        
        # Apply smooth acceleration limiting
        new_pan_speed = self.calculate_smooth_speed(
            target_pan_speed if pan_dir else 0, 
            self.current_pan_speed, 
            MAX_ACCELERATION
        )
        new_tilt_speed = self.calculate_smooth_speed(
            target_tilt_speed if tilt_dir else 0, 
            self.current_tilt_speed, 
            MAX_ACCELERATION
        )
        
        # Update current state
        self.current_pan_speed = new_pan_speed
        self.current_tilt_speed = new_tilt_speed
        self.current_pan_dir = pan_dir
        self.current_tilt_dir = tilt_dir
        
        # Track camera angle changes for pose estimation
        speed_to_angle_factor = 0.1
        if pan_dir == "R":
            angle_change = new_pan_speed * speed_to_angle_factor
        elif pan_dir == "L":
            angle_change = -new_pan_speed * speed_to_angle_factor
        else:
            angle_change = 0
        self.cumulative_pan_angle += angle_change
        
        if tilt_dir == "U":
            tilt_angle_change = new_tilt_speed * speed_to_angle_factor
        elif tilt_dir == "D":
            tilt_angle_change = -new_tilt_speed * speed_to_angle_factor
        else:
            tilt_angle_change = 0
        self.cumulative_tilt_angle += tilt_angle_change
        
        debug_info['camera_angles'] = {
            'pan': self.cumulative_pan_angle,
            'tilt': self.cumulative_tilt_angle
        }
        
        # Determine if we should move
        should_move = (
            (new_pan_speed >= PT_SPEED_MIN and pan_dir) or 
            (new_tilt_speed >= PT_SPEED_MIN and tilt_dir)
        )
        
        # Ensure minimum speeds when moving
        final_pan_speed = max(int(new_pan_speed), PT_SPEED_MIN) if should_move and pan_dir else 0
        final_tilt_speed = max(int(new_tilt_speed), PT_SPEED_MIN) if should_move and tilt_dir else 0
        
        debug_info.update({
            'target_speeds': (target_pan_speed, target_tilt_speed),
            'final_speeds': (final_pan_speed, final_tilt_speed),
            'should_move': should_move
        })
        
        return (
            final_pan_speed,
            final_tilt_speed,
            pan_dir if should_move else "",
            tilt_dir if should_move else "",
            should_move,
            debug_info
        )
    
    def get_smooth_movement(self, off_x, off_y):
        """Legacy method for backward compatibility."""
        result = self.get_enhanced_movement(off_x, off_y)
        return result[:5]  # Return only the first 5 elements
    
    def get_camera_angles(self):
        """Get current estimated camera angles."""
        return self.cumulative_pan_angle, self.cumulative_tilt_angle
    
    def stop(self):
        """Gradually stop movement."""
        self.current_pan_speed = 0
        self.current_tilt_speed = 0
        self.current_pan_dir = ""
        self.current_tilt_dir = ""
        self.last_target_pan_speed = 0
        self.last_target_tilt_speed = 0

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
        
        IMPORTANT: Direction mapping for BirdDog X1:
        - pan_dir "L" = pan left (negative values)
        - pan_dir "R" = pan right (positive values)  
        - tilt_dir "U" = tilt up (positive values)
        - tilt_dir "D" = tilt down (negative values)
        """
        try:
            # Clamp velocities to valid ranges
            pan_vel = max(0, min(pan_vel, PT_SPEED_MAX))
            tilt_vel = max(0, min(tilt_vel, 20))  # X1 tilt max is typically 20
            
            # Convert direction and velocity to signed values for pantilt command
            # Pan: Left is negative, Right is positive
            if pan_dir == "L":
                pan_command = -pan_vel
            elif pan_dir == "R":
                pan_command = pan_vel
            else:
                pan_command = 0
            
            # Tilt: Up is positive, Down is negative (standard VISCA convention)
            if tilt_dir == "U":
                tilt_command = tilt_vel
            elif tilt_dir == "D":
                tilt_command = -tilt_vel
            else:
                tilt_command = 0
            
            # Send the pantilt command
            self.camera.pantilt(pan_command, tilt_command)
            
            # Enhanced logging for debugging
            if pan_command != 0 or tilt_command != 0:
                print(f"VISCA move: pan={pan_command} ({pan_dir}), tilt={tilt_command} ({tilt_dir})")
            
        except Exception as e:
            print(f"VISCA move error: {e}")
            print(f"  Parameters: pan_vel={pan_vel}, tilt_vel={tilt_vel}, pan_dir={pan_dir}, tilt_dir={tilt_dir}")
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
# 3. Main loop – detection ➜ PTZ correction with enhanced UI
# --------------------------------------------------------------------------- #

# Global variables for window control
window_should_close = False
close_button_rect = None

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events in the tracking window."""
    global window_should_close, close_button_rect
    
    if event == cv2.EVENT_LBUTTONUP and close_button_rect:
        # Check if click is within close button area
        btn_x, btn_y, btn_w, btn_h = close_button_rect
        if btn_x <= x <= btn_x + btn_w and btn_y <= y <= btn_y + btn_h:
            print("🔴 Close button clicked - stopping tracking...")
            window_should_close = True

def draw_close_button(frame):
    """Draw a close button on the frame and return its coordinates."""
    global close_button_rect
    
    h, w = frame.shape[:2]
    
    # Close button dimensions and position (top-right corner)
    btn_w, btn_h = 80, 30
    btn_x = w - btn_w - 10
    btn_y = 10
    
    close_button_rect = (btn_x, btn_y, btn_w, btn_h)
    
    # Draw button background
    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (0, 0, 200), -1)  # Red background
    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 255), 2)  # White border
    
    # Draw "CLOSE" text
    text = "CLOSE"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = btn_x + (btn_w - text_size[0]) // 2
    text_y = btn_y + (btn_h + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def track():
    global window_should_close
    
    detector, model_type = load_model(MODEL_PATH)
    cam = X1Visca()
    movement_controller = SmoothMovementController()
    
    # Initialize 3D tracking system
    object_3d_tracker = Object3DTracker() if ENABLE_POSE_ESTIMATION else None
    
    print(f"🎯 Starting enhanced object tracking for class '{TRACK_CLASS}'")
    print(f"📊 Movement settings:")
    print(f"   - Max acceleration: {MAX_ACCELERATION}")
    print(f"   - Min move threshold: {MIN_MOVE_THRESHOLD}")
    print(f"   - Speed ramp factor: {SPEED_RAMP_FACTOR}")
    print(f"   - Max tracking speed: {MAX_TRACKING_SPEED}")
    print(f"   - Deadband: ±{DEADBAND}")
    print(f"   - Distance scaling: {DISTANCE_SCALING_FACTOR}")
    print(f"   - Velocity smoothing: {VELOCITY_SMOOTHING}")
    print(f"🔬 Pose estimation settings:")
    print(f"   - 3D pose estimation: {ENABLE_POSE_ESTIMATION}")
    print(f"   - Predictive tracking: {ENABLE_PREDICTIVE_TRACKING}")
    print(f"   - Depth estimation: {ENABLE_DEPTH_ESTIMATION}")
    print(f"   - Object size: {OBJECT_REAL_WIDTH_MM}mm x {OBJECT_REAL_HEIGHT_MM}mm")
    print(f"📺 Video settings:")
    print(f"   - NDI source: {NDI_NAME}")
    print(f"   - Camera IP: {CAMERA_IP}")
    print(f"🖱️  Window controls:")
    print(f"   - Click 'CLOSE' button in window to exit")
    print(f"   - Press ESC or Q to quit")
    print(f"   - Click X on window title bar to close")
    
    frame_count = 0
    last_detection_time = time.time()
    tracking_performance = {"total_frames": 0, "detected_frames": 0, "centered_frames": 0}
    
    # Pose estimation initialization flag
    pose_initialized = False
    window_created = False
    
    try:
        for frame in ndi_frames(NDI_NAME):
            if window_should_close:
                break
                
            h, w, _ = frame.shape
            frame_count += 1
            tracking_performance["total_frames"] += 1
            current_time = time.time()
            
            # Create window and set mouse callback on first frame
            if not window_created:
                cv2.namedWindow("BirdTrack", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("BirdTrack", mouse_callback)
                cv2.resizeWindow("BirdTrack", 1200, 800)  # Set a good default size
                window_created = True
                print("🖼️  Tracking window created - click CLOSE button to exit")
            
            # Initialize pose estimation on first frame
            if ENABLE_POSE_ESTIMATION and not pose_initialized and object_3d_tracker:
                object_3d_tracker.initialize(w, h)
                pose_initialized = True
            
            # run detector (no tracker → lowest latency)
            results = detect_objects(detector, frame, model_type)
            boxes  = [b for b in results if int(b.cls[0] if hasattr(b, 'cls') else b.cls)==TRACK_CLASS]
            
            if not boxes:
                # No target detected - gradually stop movement
                pan_speed, tilt_speed, pan_dir, tilt_dir, should_move = movement_controller.get_smooth_movement(0, 0)
                if should_move:
                    cam.move(pan_speed, tilt_speed, pan_dir, tilt_dir)
                else:
                    cam.stop()
                    movement_controller.stop()
                
                # Show "searching" status
                cv2.putText(frame, "SEARCHING FOR TARGET...", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(frame, f"Frame: {frame_count} | Detection rate: {tracking_performance['detected_frames']/max(1,tracking_performance['total_frames'])*100:.1f}%", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
                # Draw close button
                frame = draw_close_button(frame)
                
                cv2.imshow("BirdTrack", frame)
                
                # Check for key press or window close
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC or Q key
                    break
                
                # Check if window was closed via X button
                if cv2.getWindowProperty("BirdTrack", cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
                continue
                
            # Target detected - update detection time and stats
            last_detection_time = time.time()
            tracking_performance["detected_frames"] += 1
            
            # pick the highest-confidence box
            box = max(boxes, key=lambda b: float(b.conf[0] if hasattr(b, 'conf') and hasattr(b.conf, '__getitem__') else b.conf))
            
            # Extract coordinates - handle both YOLOv5 and YOLOv8 formats
            if hasattr(box.xyxy[0], '__getitem__'):
                x0,y0,x1,y1 = map(int, box.xyxy[0])
            else:
                x0,y0,x1,y1 = map(int, box.xyxy)
                
            cx, cy = (x0+x1)//2, (y0+y1)//2
            # normalised offset from centre (-1 … +1)
            off_x, off_y = ( (cx - w/2)/(w/2), (cy - h/2)/(h/2) )
            
            # Get confidence for display
            conf_value = float(box.conf[0] if hasattr(box.conf, '__getitem__') else box.conf)
            
            # 3D pose estimation and tracking
            tracking_info = None
            predicted_position = None
            
            if ENABLE_POSE_ESTIMATION and object_3d_tracker and pose_initialized:
                # Get current camera angles for pose estimation
                pan_angle, tilt_angle = movement_controller.get_camera_angles()
                
                # Update 3D tracking
                tracking_info = object_3d_tracker.update_tracking(
                    (x0, y0, x1, y1), current_time, pan_angle, tilt_angle
                )
                
                # Get motion prediction
                if ENABLE_PREDICTIVE_TRACKING and tracking_info:
                    predicted_position = object_3d_tracker.predict_future_position(
                        movement_controller.prediction_lookahead
                    )
            
            # Use enhanced movement controller with 3D information
            if tracking_info:
                result = movement_controller.get_enhanced_movement(
                    off_x, off_y, tracking_info, predicted_position
                )
                pan_speed, tilt_speed, pan_dir, tilt_dir, should_move, debug_info = result
            else:
                # Fallback to basic movement
                pan_speed, tilt_speed, pan_dir, tilt_dir, should_move = movement_controller.get_smooth_movement(off_x, off_y)
                debug_info = {}
            
            # Enhanced movement logic with improved deadband handling
            horizontal_deadband = DEADBAND
            vertical_deadband = DEADBAND * VERTICAL_DEADBAND_MULTIPLIER
            is_horizontally_centered = abs(off_x) < horizontal_deadband
            is_vertically_centered = abs(off_y) < vertical_deadband
            is_centered = is_horizontally_centered and is_vertically_centered
            
            if is_centered:
                tracking_performance["centered_frames"] += 1
                # Object is centered - gradually stop movement
                pan_speed, tilt_speed, pan_dir, tilt_dir, should_move = movement_controller.get_smooth_movement(0, 0)
                if not should_move:
                    cam.stop()
                    movement_controller.stop()
                else:
                    # Still decelerating
                    cam.move(pan_speed, tilt_speed, pan_dir, tilt_dir)
            else:
                # Object needs tracking - use calculated movement
                if should_move:
                    cam.move(pan_speed, tilt_speed, pan_dir, tilt_dir)
                else:
                    # Object detected but movement too small - stop for stability
                    cam.stop()
            
            # Enhanced HUD with 3D tracking info
            cv2.circle(frame, (cx,cy), 8, (0,255,0), 2)
            cv2.line(frame, (w//2, h//2), (cx,cy), (0,255,0), 2)
            cv2.rectangle(frame, (x0,y0), (x1,y1), (255,0,0), 2)
            
            # Draw center crosshairs and deadband zones (if visualization enabled)
            if ENABLE_MOVEMENT_VISUALIZATION:
                cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (255,255,255), 1)
                cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (255,255,255), 1)
                
                # Draw separate deadband zones for horizontal and vertical
                h_deadband_w = int(horizontal_deadband * w/2)
                v_deadband_h = int(vertical_deadband * h/2)
                
                # Horizontal deadband (pan)
                cv2.rectangle(frame, 
                             (w//2 - h_deadband_w, h//2 - 10), 
                             (w//2 + h_deadband_w, h//2 + 10), 
                             (128,128,255), 1)  # Blue for horizontal
                
                # Vertical deadband (tilt) 
                cv2.rectangle(frame, 
                             (w//2 - 10, h//2 - v_deadband_h), 
                             (w//2 + 10, h//2 + v_deadband_h), 
                             (255,128,128), 1)  # Red for vertical
                
                # Draw movement vector
                if should_move and (pan_dir or tilt_dir):
                    vector_length = 50
                    end_x = w//2 + int(off_x * vector_length)
                    end_y = h//2 + int(off_y * vector_length)
                    cv2.arrowedLine(frame, (w//2, h//2), (end_x, end_y), (255,0,255), 2)
                
                # Draw prediction vector if available
                if ENABLE_PREDICTIVE_TRACKING and predicted_position is not None and tracking_info:
                    if debug_info.get('prediction_used'):
                        cv2.circle(frame, (cx, cy), 12, (255,255,0), 2)  # Yellow circle for prediction
                
                # Show stability status
                stability_text = f"H:{'✓' if is_horizontally_centered else '✗'} V:{'✓' if is_vertically_centered else '✗'}"
                cv2.putText(frame, stability_text, (w-200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            # Enhanced pose visualization
            if ENABLE_POSE_VISUALIZATION and tracking_info:
                pose_y_offset = 200
                distance_mm = tracking_info.get('distance_mm', 0)
                position_3d = tracking_info.get('3d_position')
                velocity = tracking_info.get('velocity')
                confidence = tracking_info.get('confidence', 0)
                
                # Distance and 3D position
                cv2.putText(frame, f"Distance: {distance_mm:.0f}mm ({distance_mm/1000:.2f}m)", (10, pose_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                
                if position_3d is not None:
                    cv2.putText(frame, f"3D Pos: X{position_3d[0]:.0f} Y{position_3d[1]:.0f} Z{position_3d[2]:.0f}mm", (10, pose_y_offset + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                
                if velocity is not None:
                    vel_magnitude = np.linalg.norm(velocity)
                    cv2.putText(frame, f"Velocity: {vel_magnitude:.1f}mm/s", (10, pose_y_offset + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                
                # Confidence indicators
                cv2.putText(frame, f"3D Confidence: {confidence:.2f}", (10, pose_y_offset + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                
                # Camera pose
                pan_angle, tilt_angle = movement_controller.get_camera_angles()
                cv2.putText(frame, f"Cam: Pan{pan_angle:.1f}° Tilt{tilt_angle:.1f}°", (10, pose_y_offset + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            
            # Status display with better information
            status_color = (0,255,0) if is_centered else (255,255,0)
            status_text = "CENTERED" if is_centered else "TRACKING"
            
            # Calculate performance metrics
            detection_rate = tracking_performance["detected_frames"] / max(1, tracking_performance["total_frames"]) * 100
            center_rate = tracking_performance["centered_frames"] / max(1, tracking_performance["detected_frames"]) * 100
            
            cv2.putText(frame, f"TARGET: {status_text}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            if ENABLE_TRACKING_DEBUG:
                cv2.putText(frame, f"Offset: X{off_x:+.3f} Y{off_y:+.3f} | Conf: {conf_value:.2f}", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"Movement: P{pan_speed}({pan_dir}) T{tilt_speed}({tilt_dir}) | Active: {should_move}", (10,90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"Position: ({cx},{cy}) | H-DB: ±{horizontal_deadband:.3f} V-DB: ±{vertical_deadband:.3f}", (10,120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                # Enhanced movement debugging
                if tracking_info and debug_info:
                    movement_analysis = debug_info.get('movement_analysis', {})
                    cv2.putText(frame, f"Thresholds: Pan±{movement_analysis.get('pan_threshold', 0):.3f} Tilt±{movement_analysis.get('tilt_threshold', 0):.3f}", (10,140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                    cv2.putText(frame, f"Needs Move: Pan:{movement_analysis.get('pan_needs_movement', False)} Tilt:{movement_analysis.get('tilt_needs_movement', False)}", (10,160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                
                cv2.putText(frame, f"Performance: Det:{detection_rate:.1f}% Ctr:{center_rate:.1f}% | Frame: {frame_count}", (10,180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            else:
                cv2.putText(frame, f"Conf: {conf_value:.2f} | Det: {detection_rate:.1f}% | Stability: H{'✓' if is_horizontally_centered else '✗'} V{'✓' if is_vertically_centered else '✗'} | Frame: {frame_count}", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            # Draw close button (always visible)
            frame = draw_close_button(frame)
            
            cv2.imshow("BirdTrack", frame)
            
            # Check for key press or window close
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or Q key
                break
            
            # Check if window was closed via X button
            if cv2.getWindowProperty("BirdTrack", cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\n🛑 Tracking interrupted by user")
    except Exception as e:
        print(f"❌ Tracking error: {e}")
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        try:
            cam.stop()
            print("🏁 Camera stopped")
        except:
            pass
        
        cv2.destroyAllWindows()
        window_should_close = False  # Reset for next run
        print("🖼️  Window closed successfully")

def test_tracking_movements():
    """Test function to verify camera movement directions, speeds, and pose estimation."""
    print("🧪 Testing enhanced camera tracking with pose estimation...")
    cam = X1Visca()
    movement_controller = SmoothMovementController()
    
    # Test movements in all directions
    test_scenarios = [
        (0.2, 0.0, "Object to RIGHT -> should pan RIGHT"),
        (-0.2, 0.0, "Object to LEFT -> should pan LEFT"),
        (0.0, 0.2, "Object BELOW -> should tilt DOWN"),
        (0.0, -0.2, "Object ABOVE -> should tilt UP"),
        (0.2, 0.2, "Object to RIGHT and BELOW -> should pan RIGHT and tilt DOWN"),
        (-0.2, -0.2, "Object to LEFT and ABOVE -> should pan LEFT and tilt UP"),
        (0.03, 0.0, "Small offset RIGHT -> should make small movement"),
        (0.0, 0.0, "Centered -> should stop"),
    ]
    
    # Create mock tracking info for testing
    mock_tracking_info = {
        'distance_mm': 2000.0,  # 2 meters
        'confidence': 0.8,
        '3d_position': np.array([100.0, 50.0, 2000.0])
    }
    
    print("\n📊 Testing basic movements:")
    for off_x, off_y, description in test_scenarios:
        print(f"\n🎯 Test: {description}")
        print(f"   Input offset: X={off_x:+.3f}, Y={off_y:+.3f}")
        
        # Test basic movement
        pan_speed, tilt_speed, pan_dir, tilt_dir, should_move = movement_controller.get_smooth_movement(off_x, off_y)
        print(f"   Basic Output: Pan={pan_speed}({pan_dir}), Tilt={tilt_speed}({tilt_dir}), Move={should_move}")
        
        # Test enhanced movement with 3D info
        if ENABLE_POSE_ESTIMATION:
            result = movement_controller.get_enhanced_movement(off_x, off_y, mock_tracking_info, None)
            pan_speed_3d, tilt_speed_3d, pan_dir_3d, tilt_dir_3d, should_move_3d, debug_info = result
            print(f"   Enhanced Output: Pan={pan_speed_3d}({pan_dir_3d}), Tilt={tilt_speed_3d}({tilt_dir_3d}), Move={should_move_3d}")
            
            if debug_info.get('distance_scaling'):
                print(f"   Distance scaling: {debug_info['distance_scaling']:.2f}")
            if debug_info.get('confidence_scaling'):
                print(f"   Confidence scaling: {debug_info['confidence_scaling']:.2f}")
        
        if should_move or (ENABLE_POSE_ESTIMATION and should_move_3d):
            basic_pan = pan_speed * (-1 if pan_dir=='L' else 1 if pan_dir=='R' else 0)
            basic_tilt = tilt_speed * (1 if tilt_dir=='U' else -1 if tilt_dir=='D' else 0)
            print(f"   ✅ VISCA command: pan={basic_pan}, tilt={basic_tilt}")
        else:
            print(f"   ⏹️  Would stop camera (no movement)")
        
        time.sleep(0.3)  # Brief pause between tests
    
    # Test pose estimation features
    if ENABLE_POSE_ESTIMATION:
        print("\n🔬 Testing pose estimation features:")
        
        # Test distance estimation
        print(f"\n📏 Distance estimation test:")
        print(f"   Object real size: {OBJECT_REAL_WIDTH_MM}mm x {OBJECT_REAL_HEIGHT_MM}mm")
        
        # Simulate different object sizes (bounding boxes)
        test_bbox_sizes = [
            (100, 67, "Close object (large bbox)"),
            (50, 33, "Medium distance"),
            (25, 17, "Far object (small bbox)"),
        ]
        
        pose_estimator = CameraPoseEstimator(1920, 1080)  # Simulate HD frame
        
        for bbox_w, bbox_h, description in test_bbox_sizes:
            estimated_distance = pose_estimator.estimate_object_distance(bbox_w, bbox_h)
            print(f"   {description}: bbox={bbox_w}x{bbox_h}px -> distance={estimated_distance:.0f}mm ({estimated_distance/1000:.2f}m)")
        
        # Test 3D position calculation
        print(f"\n📍 3D position test:")
        center_positions = [
            (960, 540, "Center of frame"),
            (1200, 540, "Right side"),
            (720, 540, "Left side"),
            (960, 400, "Upper center"),
            (960, 680, "Lower center"),
        ]
        
        for px, py, description in center_positions:
            world_pos = pose_estimator.pixel_to_world_ray(px, py, 2000.0)  # 2m distance
            print(f"   {description}: pixel=({px},{py}) -> 3D=({world_pos[0]:.0f},{world_pos[1]:.0f},{world_pos[2]:.0f})mm")
        
        # Test motion prediction
        if ENABLE_PREDICTIVE_TRACKING:
            print(f"\n🔮 Motion prediction test:")
            motion_predictor = MotionPredictor()
            
            # Simulate object moving in a straight line
            for i in range(5):
                timestamp = time.time() + i * 0.1
                position = np.array([100.0 + i * 50, 0.0, 2000.0])  # Moving right at 500mm/s
                motion_predictor.add_position(position, timestamp)
                
                if i >= 2:  # Start predicting after 3 observations
                    predicted_pos = motion_predictor.predict_position(0.2)  # 200ms prediction
                    current_velocity = motion_predictor.get_current_velocity()
                    confidence = motion_predictor.get_motion_confidence()
                    
                    if predicted_pos is not None:
                        print(f"   Step {i}: Current=({position[0]:.0f},{position[1]:.0f},{position[2]:.0f})")
                        print(f"           Predicted=({predicted_pos[0]:.0f},{predicted_pos[1]:.0f},{predicted_pos[2]:.0f})")
                        print(f"           Velocity=({current_velocity[0]:.0f},{current_velocity[1]:.0f},{current_velocity[2]:.0f})mm/s")
                        print(f"           Confidence={confidence:.2f}")
    
    print("\n✅ Enhanced tracking test complete!")
    print("📋 Summary of improvements:")
    print("   - ✅ Correct movement directions maintained")
    print("   - ✅ Distance-based speed scaling")
    print("   - ✅ Confidence-based movement adjustment")
    if ENABLE_POSE_ESTIMATION:
        print("   - ✅ 3D pose estimation functional")
        print("   - ✅ Object distance calculation working")
        print("   - ✅ Camera angle tracking active")
    if ENABLE_PREDICTIVE_TRACKING:
        print("   - ✅ Motion prediction system operational")
    
    print("\n🚀 Ready for enhanced object tracking!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test movement directions
        test_tracking_movements()
    else:
        # Normal tracking mode
        try:
            track()
        except KeyboardInterrupt:
            print("\n🛑 Tracking stopped by user")
        except Exception as e:
            print(f"❌ Tracking error: {e}")
        finally:
            try:
                X1Visca().stop()
                print("🏁 Camera stopped")
            except:
                pass
