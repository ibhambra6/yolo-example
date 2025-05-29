"""
birdtrack.py – drive a BirdDog X1 so it automatically keeps a YOLO-detected
object in the centre of frame.

▶ Prereqs  (Windows / macOS / Linux + Python ≥ 3.9)
---------------------------------------------------
pip install ultralytics opencv-python numpy
# NewTek NDI runtime + Python wrapper:
pip install ndi-python               # thin wrapper over official NDI SDK
                                      #  ➜ https://github.com/buresu/ndi-python
"""
import cv2, numpy as np, socket, time, math, threading
from ultralytics import YOLO
import ndi   # noqa:  pip install ndi-python

### ------------------- USER CONFIG -------------------
CAMERA_IP   = "192.168.100.150"       #  X1 address
NDI_NAME    = "X1-Studio-Cam"         #  NDI stream name advertised by the camera
VISCA_PORT  = 52381                   #  fixed for BirdDog/Sony VISCA-IP
MODEL_PATH  = "best.pt"               #  your trained YOLO model
TRACK_CLASS = 0                       #  class-id you want to track
FOV_H_DEG   = 55.8                    #  X1 widest horizontal FoV; used for angle calc
DEADBAND    = 0.05                    #  centre dead-zone (±5 % of frame)
PT_SPEED_MIN, PT_SPEED_MAX = 0x01, 0x18  # VISCA velocity range (1-24)
### ----------------------------------------------------

# --------------------------------------------------------------------------- #
# 1. VISCA-over-IP helpers
# --------------------------------------------------------------------------- #
class X1Visca:
    """Minimal VISCA-IP client: only pan/tilt/stop."""
    def __init__(self, ip:str, port:int=VISCA_PORT):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.lock = threading.Lock()

    def _send(self, payload:bytes):
        """Send a raw VISCA packet with BirdDog header (8 dummy bytes)."""
        packet = b"\x01\x00\x00" + bytes([len(payload)+1]) + b"\x00\x00\x00\x00" + payload
        with self.lock:
            self.sock.sendto(packet, self.addr)

    def move(self, pan_vel:int, tilt_vel:int, pan_dir:str, tilt_dir:str):
        """
        Continuous move. Directions:  'L','R','S'  (stop) ; 'U','D','S'.
        pan_vel: 1-24, tilt_vel: 1-20   (X1 spec)
        """
        dmap = {"L":0x02, "R":0x01, "S":0x03,
                "U":0x01, "D":0x02}
        payload = bytes([
            0x81, 0x01, 0x06, 0x01,
            pan_vel, tilt_vel,
            dmap.get(pan_dir,0x03),
            dmap.get(tilt_dir,0x03),
            0xFF
        ])
        self._send(payload)

    def stop(self):
        self.move(0x00, 0x00, "S", "S")    #  stop cmd (all zero / 0x03)

# --------------------------------------------------------------------------- #
# 2. NDI receiver  ➜ OpenCV BGR frame
# --------------------------------------------------------------------------- #
def ndi_frames(source_name:str):
    finder   = ndi.finder_create()
    recv     = ndi.recv_create(v=True, a=False)
    source   = None
    while True:
        if source is None:
            # wait until the announced name appears on the network
            for s in ndi.finder_get_sources(finder):
                if source_name in s.ndi_name.decode():
                    source = s
                    ndi.recv_connect(recv, source)
                    break
            time.sleep(0.25)
            continue

        frame = ndi.recv_capture_v2(recv, timeout_ms=1000)
        if frame.data is None:
            continue
        h, w = frame.yres, frame.xres
        img  = np.frombuffer(frame.data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        yield img
        ndi.recv_free_video_v2(recv, frame)

# --------------------------------------------------------------------------- #
# 3. Main loop – detection ➜ PTZ correction
# --------------------------------------------------------------------------- #
def track():
    detector = YOLO(MODEL_PATH)
    cam      = X1Visca(CAMERA_IP)
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
        X1Visca(CAMERA_IP).stop()
