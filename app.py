# app.py

import cv2
import os
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

# Landmark optimization
LANDMARKS_EVERY_N_FRAMES = 5
frame_counter = 0

# Initialize camera
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Detect faces
        boxes = face_detector.detect_faces(frame)

        for box in boxes:
            x1, y1, x2, y2 = box

            # Crop face
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # Recognize driver
            driver_name = face_recognizer.recognize(face_img)

            # Draw face box and name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{driver_name}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Run landmarks every N frames
            if frame_counter % LANDMARKS_EVERY_N_FRAMES == 0:
                try:
                    landmarks = face_landmarks.get_landmarks(frame, (x1, y1, x2, y2))

                    if landmarks is not None:
                        # Eye state detection
                        left_eye_box, right_eye_box = face_landmarks.get_eye_boxes(landmarks, scale_factor=1.2)

                        # Clamp eye boxes to frame
                        lx1, ly1, lx2, ly2 = left_eye_box
                        rx1, ry1, rx2, ry2 = right_eye_box

                        lx1, ly1, lx2, ly2 = max(0,lx1), max(0,ly1), min(frame.shape[1],lx2), min(frame.shape[0],ly2)
                        rx1, ry1, rx2, ry2 = max(0,rx1), max(0,ry1), min(frame.shape[1],rx2), min(frame.shape[0],ry2)

                        # Crop eyes
                        left_eye_crop = frame[ly1:ly2, lx1:lx2]
                        right_eye_crop = frame[ry1:ry2, rx1:rx2]

                        if left_eye_crop.size == 0 or right_eye_crop.size == 0:
                            continue

                        left_eye_img = cv2.resize(left_eye_crop, (24, 24))
                        right_eye_img = cv2.resize(right_eye_crop, (24, 24))

                        # Eye state prediction
                        left_eye_state = eye_state_detector.predict(left_eye_img)
                        right_eye_state = eye_state_detector.predict(right_eye_img)

                        # Draw eye states
                        cv2.putText(frame, f'Left Eye: {left_eye_state}', (lx1, ly2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        cv2.putText(frame, f'Right Eye: {right_eye_state}', (rx1, ry2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # Drowsy logic (eyes closed)
                        if left_eye_state == 'Closed' and right_eye_state == 'Closed':
                            closed_frames_counter += 1
                        else:
                            closed_frames_counter = 0

                        if closed_frames_counter >= DROWSY_FRAME_THRESHOLD:
                            cv2.putText(frame, 'DROWSY DRIVER!', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                        # Yawn detection
                        mouth_box = face_landmarks.get_mouth_box(landmarks)
                        mx1, my1, mx2, my2 = mouth_box
                        mx1, my1, mx2, my2 = max(0,mx1), max(0,my1), min(frame.shape[1],mx2), min(frame.shape[0],my2)

                        mouth_img = frame[my1:my2, mx1:mx2]

                        if mouth_img.size != 0:
                            yawn_state = yawn_detector.predict(mouth_img)

                            # Draw mouth box and yawn state
                            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 255), 2)
                            cv2.putText(frame, f'Mouth: {yawn_state}', (mx1, my2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                except Exception as e:
                    print(f"❌ Landmark / EyeState / Yawn error: {e}")

        # Show main frame
        cv2.imshow('Driver Monitoring System', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera released, window closed. App exited safely.")
