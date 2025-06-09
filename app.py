# app.py

import cv2
import os
from detectors.face_detector import FaceDetector
from detectors.face_recognizer import FaceRecognizer
from detectors.face_landmarks import FaceLandmarks

# Initialize components
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
face_landmarks = FaceLandmarks()

# Initialize camera
cap = cv2.VideoCapture(0)

# Main loop
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1

    # Only run landmarks every 5 frames
    run_landmarks = (frame_count % 5 == 0)

    boxes = face_detector.detect_faces(frame)

    for box in boxes:
        x1, y1, x2, y2 = box
        face_img = frame[y1:y2, x1:x2]

        driver_name = face_recognizer.recognize(face_img)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{driver_name}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Run landmarks only every N frames
        if run_landmarks:
            landmarks = face_landmarks.get_landmarks(frame, (x1, y1, x2, y2))
            if landmarks is not None:
                left_eye_box, right_eye_box = face_landmarks.get_eye_boxes(landmarks)

                lx1, ly1, lx2, ly2 = left_eye_box
                rx1, ry1, rx2, ry2 = right_eye_box

                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)


    # Show frame
    cv2.imshow('Driver Monitoring System', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
