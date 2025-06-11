# app_threaded_mac_safe_final.py

import cv2
import threading
import time
from detectors.face_detector import FaceDetector
from detectors.face_recognizer import FaceRecognizer
from detectors.face_landmarks import FaceLandmarks
from detectors.eye_state import EyeStateDetector
from detectors.yawn_detector import YawnDetector

# Initialize detectors
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
face_landmarks = FaceLandmarks()
eye_state_detector = EyeStateDetector()
yawn_detector = YawnDetector()

# Drowsy detection params
DROWSY_FRAME_THRESHOLD = 15
closed_frames_counter = 0
recent_yawn_counter = 0
YAWN_RECENT_FRAMES = 100

# Shared variables
shared_frame = None
shared_boxes = []
shared_driver_name = "Unknown"
shared_left_eye_state = "Open"
shared_right_eye_state = "Open"
shared_yawn_state = "No Yawn"
shared_landmarks = None

# Thread control
running = True

# Thread lock for shared_frame
frame_lock = threading.Lock()

# FPS counter
fps = 0
last_time = time.time()

# ───────────────────────────────────────────────
# Utility functions for drawing
# ───────────────────────────────────────────────

def draw_face_box(frame, box, driver_name):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'{driver_name}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def draw_eye_states(frame, landmarks, left_eye_state, right_eye_state):
    left_eye_box, right_eye_box = face_landmarks.get_eye_boxes(landmarks, scale_factor=1.2)

    lx1, ly1, lx2, ly2 = left_eye_box
    rx1, ry1, rx2, ry2 = right_eye_box

    lx1, ly1, lx2, ly2 = clamp_box(frame, lx1, ly1, lx2, ly2)
    rx1, ry1, rx2, ry2 = clamp_box(frame, rx1, ry1, rx2, ry2)

    cv2.putText(frame, f'Left Eye: {left_eye_state}', (lx1, ly2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f'Right Eye: {right_eye_state}', (rx1, ry2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def draw_drowsy_alert(frame):
    cv2.putText(frame, 'DROWSY DRIVER!', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

def draw_mouth_box(frame, landmarks, yawn_state):
    mouth_box = face_landmarks.get_mouth_box(landmarks)
    mx1, my1, mx2, my2 = mouth_box
    mx1, my1, mx2, my2 = clamp_box(frame, mx1, my1, mx2, my2)

    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 255), 2)
    cv2.putText(frame, f'Mouth: {yawn_state}', (mx1, my2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def clamp_box(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return x1, y1, x2, y2

# ───────────────────────────────────────────────
# Threads
# ───────────────────────────────────────────────

def capture_thread():
    global shared_frame, running
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            shared_frame = frame.copy()

    cap.release()

def inference_thread():
    global shared_boxes, shared_driver_name, shared_left_eye_state, shared_right_eye_state, shared_yawn_state
    global shared_landmarks, closed_frames_counter, recent_yawn_counter, running

    while running:
        with frame_lock:
            frame_local = shared_frame.copy() if shared_frame is not None else None

        if frame_local is None:
            time.sleep(0.01)
            continue

        # Detect faces
        boxes = face_detector.detect_faces(frame_local)
        shared_boxes = boxes

        for box in boxes:
            x1, y1, x2, y2 = box
            face_img = frame_local[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            shared_driver_name = face_recognizer.recognize(face_img)

            try:
                landmarks = face_landmarks.get_landmarks(frame_local, (x1, y1, x2, y2))

                if landmarks is not None:
                    shared_landmarks = landmarks

                    # Eyes
                    left_eye_box, right_eye_box = face_landmarks.get_eye_boxes(landmarks, scale_factor=1.2)
                    lx1, ly1, lx2, ly2 = clamp_box(frame_local, *left_eye_box)
                    rx1, ry1, rx2, ry2 = clamp_box(frame_local, *right_eye_box)

                    left_eye_crop = frame_local[ly1:ly2, lx1:lx2]
                    right_eye_crop = frame_local[ry1:ry2, rx1:rx2]

                    if left_eye_crop.size != 0 and right_eye_crop.size != 0:
                        left_eye_img = cv2.resize(left_eye_crop, (24, 24))
                        right_eye_img = cv2.resize(right_eye_crop, (24, 24))

                        shared_left_eye_state = eye_state_detector.predict(left_eye_img)
                        shared_right_eye_state = eye_state_detector.predict(right_eye_img)

                    # Drowsy counter
                    if shared_left_eye_state == 'Closed' and shared_right_eye_state == 'Closed':
                        closed_frames_counter += 1
                    else:
                        closed_frames_counter = 0

                    # Yawn
                    mouth_box = face_landmarks.get_mouth_box(landmarks)
                    mx1, my1, mx2, my2 = clamp_box(frame_local, *mouth_box)

                    mouth_img = frame_local[my1:my2, mx1:mx2]
                    if mouth_img.size != 0:
                        shared_yawn_state = yawn_detector.predict(mouth_img)

                    # Update recent_yawn_counter
                    if shared_yawn_state == 'Yawn':
                        recent_yawn_counter = YAWN_RECENT_FRAMES
                    elif recent_yawn_counter > 0:
                        recent_yawn_counter -= 1

            except Exception as e:
                print(f"❌ Landmark / EyeState / Yawn error: {e}")

        time.sleep(0.2)  # Inference rate

# ───────────────────────────────────────────────
# Main Program
# ───────────────────────────────────────────────

# Start threads
capture_t = threading.Thread(target=capture_thread)
inference_t = threading.Thread(target=inference_thread)

capture_t.start()
inference_t.start()

# Main display loop (safe for macOS)
while True:
    with frame_lock:
        frame_disp = shared_frame.copy() if shared_frame is not None else None

    if frame_disp is not None:
        # Draw overlays
        for box in shared_boxes:
            draw_face_box(frame_disp, box, shared_driver_name)

        if shared_landmarks is not None:
            draw_eye_states(frame_disp, shared_landmarks, shared_left_eye_state, shared_right_eye_state)

            # Improved Drowsy Logic
            is_driver_drowsy = False
            if shared_left_eye_state == 'Closed' and shared_right_eye_state == 'Closed':
                if closed_frames_counter >= DROWSY_FRAME_THRESHOLD:
                    is_driver_drowsy = True
                elif recent_yawn_counter > 0 and closed_frames_counter >= 5:
                    is_driver_drowsy = True

            if is_driver_drowsy:
                draw_drowsy_alert(frame_disp)

            draw_mouth_box(frame_disp, shared_landmarks, shared_yawn_state)

        # Update FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time

        # Draw FPS and Logs
        cv2.putText(frame_disp, f'FPS: {fps:.1f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame_disp, f'Eyes Closed Frames: {closed_frames_counter}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame_disp, f'Recent Yawn Counter: {recent_yawn_counter}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Driver Monitoring System (FINAL)', frame_disp)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# Wait for threads to finish
capture_t.join()
inference_t.join()

print("✅ App exited cleanly.")
