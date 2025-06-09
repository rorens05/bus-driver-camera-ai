# detectors/face_landmarks.py

import face_alignment
import cv2
import numpy as np

class FaceLandmarks:
    def __init__(self, device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'):
      self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

    def get_landmarks(self, frame, face_box):
      x1, y1, x2, y2 = face_box
      face_img = frame[y1:y2, x1:x2]

      # Check minimum size (face must be large enough for landmarks)
      h, w, _ = face_img.shape
      if h < 100 or w < 100:
          # Skip small face → no landmarks
          return None

      # Convert BGR to RGB
      face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

      # Detect landmarks
      landmarks_list = self.fa.get_landmarks(face_rgb)

      if landmarks_list is None or len(landmarks_list) == 0:
          return None

      landmarks = landmarks_list[0]

      # Offset back to full frame coords
      landmarks[:, 0] += x1
      landmarks[:, 1] += y1

      return landmarks


    def get_eye_boxes(self, landmarks, padding=5):
        # Eye points (68-point model):
        # Left eye → points 36-41
        # Right eye → points 42-47

        left_eye_points = landmarks[36:42]
        right_eye_points = landmarks[42:48]

        # Compute bounding boxes for both eyes
        def points_to_box(pts):
            x_min = int(np.min(pts[:, 0])) - padding
            y_min = int(np.min(pts[:, 1])) - padding
            x_max = int(np.max(pts[:, 0])) + padding
            y_max = int(np.max(pts[:, 1])) + padding
            return (x_min, y_min, x_max, y_max)

        left_eye_box = points_to_box(left_eye_points)
        right_eye_box = points_to_box(right_eye_points)

        return left_eye_box, right_eye_box
