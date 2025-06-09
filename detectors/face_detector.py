# detectors/face_detector.py

from facenet_pytorch import MTCNN
import cv2
import torch

class FaceDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.detector = MTCNN(keep_all=True, device=device)

    def detect_faces(self, frame):
        h, w, _ = frame.shape

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, _ = self.detector.detect(rgb_frame)

        result_boxes = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]

                # Clip to frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Only accept valid box (area > 0)
                if x2 > x1 and y2 > y1:
                    result_boxes.append((x1, y1, x2, y2))

        return result_boxes
