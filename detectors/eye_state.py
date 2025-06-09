# detectors/eye_state.py

import onnxruntime as ort
import numpy as np
import cv2

class EyeStateDetector:
    def __init__(self, model_path='data/eye_state_model.onnx'):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_eye(self, eye_img):
        # Resize to 24x24
        eye_img = cv2.resize(eye_img, (24, 24))
        # Convert to grayscale
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        # Normalize
        eye_img = eye_img.astype(np.float32) / 255.0
        # Add correct batch and channel dimensions (1, 24, 24, 1)
        eye_img = np.expand_dims(eye_img, axis=(0, -1))
        return eye_img

    def predict(self, eye_img):
        input_eye = self.preprocess_eye(eye_img)
        output = self.session.run(None, {self.input_name: input_eye})[0]
        pred = np.argmax(output)
        return 'Closed' if pred == 0 else 'Open'
