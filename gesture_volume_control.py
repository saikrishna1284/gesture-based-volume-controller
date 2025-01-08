import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from IPython.display import display, Image, clear_output
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set a higher resolution for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam successfully opened.")

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraws = mp.solutions.drawing_utils

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

def process_frame():
    success, img = cap.read()
    if not success:
        return None, None

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []
    volPer = 0

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                mpDraws.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                z1, z2 = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (z1, z2), 15, (255, 0, 0), cv2.FILLED)
                length = math.hypot(x2 - x1, y2 - y1)

                if length < 50:
                    cv2.circle(img, (z1, z2), 15, (255, 255, 255), cv2.FILLED)
                
                vol = np.interp(length, [50, 300], [minVol, maxVol])
                volBar = np.interp(length, [50, 300], [400, 150])
                volPer = np.interp(length, [50, 300], [0, 100])

                volume.SetMasterVolumeLevel(vol, None)
                
                # Improve text visibility
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 255), 5, cv2.LINE_AA)
    
    return img, volPer

def resize_image(img, scale_percent=50):
    """Resize the image by a scale percent."""
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    resized_img = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)
    return resized_img

try:
    frame_rate = 15  # Reduce frame rate to help with stability
    prev_time = time.time()

    while True:
        current_time = time.time()
        if current_time - prev_time >= 1.0 / frame_rate:
            img, volPer = process_frame()
            if img is not None:
                # Resize image to make it smaller
                img_resized = resize_image(img, scale_percent=50)
                # Convert image to RGB for display
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                _, img_encoded = cv2.imencode('.jpg', img_rgb)
                display(Image(data=img_encoded))
                clear_output(wait=True)
            prev_time = current_time
        time.sleep(0.01)  # Small sleep to prevent tight loop
finally:
    cap.release()
    cv2.destroyAllWindows()
