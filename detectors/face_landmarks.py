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
        
    def get_mouth_box(self, landmarks, scale_factor=1.5):
        # Mouth landmarks (points 48–67)
        mouth_points = landmarks[48:68]

        x_min = np.min(mouth_points[:, 0])
        y_min = np.min(mouth_points[:, 1])
        x_max = np.max(mouth_points[:, 0])
        y_max = np.max(mouth_points[:, 1])

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        w = (x_max - x_min) * scale_factor
        h = (y_max - y_min) * scale_factor
        size = max(w, h)

        x1 = int(cx - size / 2)
        y1 = int(cy - size / 2)
        x2 = int(cx + size / 2)
        y2 = int(cy + size / 2)

        return (x1, y1, x2, y2)

    def get_eye_boxes(self, landmarks, scale_factor=1.2):
      """
      Dynamic tight crop around eye center, with square box
      scale_factor controls how big the box is:
      - 1.0 → exact eye bounding box
      - 1.2 → small padding
      """

      def compute_eye_box(eye_points):
          # Eye bounding box
          x_min = np.min(eye_points[:, 0])
          y_min = np.min(eye_points[:, 1])
          x_max = np.max(eye_points[:, 0])
          y_max = np.max(eye_points[:, 1])

          # Eye center
          cx = (x_min + x_max) / 2
          cy = (y_min + y_max) / 2

          # Eye size
          w = (x_max - x_min)
          h = (y_max - y_min)
          size = max(w, h) * scale_factor  # force square box with padding

          # Final box
          x1 = int(cx - size / 2)
          y1 = int(cy - size / 2)
          x2 = int(cx + size / 2)
          y2 = int(cy + size / 2)

          return (x1, y1, x2, y2)

      left_eye_points = landmarks[36:42]
      right_eye_points = landmarks[42:48]

      left_eye_box = compute_eye_box(left_eye_points)
      right_eye_box = compute_eye_box(right_eye_points)

      return left_eye_box, right_eye_box
