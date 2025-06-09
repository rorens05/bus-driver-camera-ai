# detectors/eye_state.py

import onnxruntime as ort
import numpy as np
import cv2
import os

class EyeStateDetector:
    def __init__(self, model_path='data/eye_state_model.onnx'):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def preprocess_eye(self, eye_img):
        # Resize to 24x24, convert to grayscale
        eye_img = cv2.resize(eye_img, (24, 24))
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        eye_img = eye_img.astype(np.float32) / 255.0

        # Add batch and channel dimension → (1, 1, 24, 24)
        eye_img = np.expand_dims(eye_img, axis=(0, 1))
        return eye_img

    def predict(self, eye_img):
        input_eye = self.preprocess_eye(eye_img)

        output = self.session.run(None, {'input': input_eye})[0]
        pred = np.argmax(output)

        if pred == 0:
            return 'Closed'
        else:
            return 'Open'
