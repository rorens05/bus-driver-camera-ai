# detectors/eye_state.py

import onnxruntime as ort
import numpy as np
import cv2
import os

class EyeStateDetector:
    def __init__(self, model_path='data/open_closed_eye.onnx'):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        # Get actual input name
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_eye(self, eye_img):
        # Resize to 32x32 → required by Intel model
        eye_img = cv2.resize(eye_img, (32, 32))

        # Convert to float32 and normalize to [0,1]
        eye_img = eye_img.astype(np.float32) / 255.0

        # Change to CHW format
        eye_img = np.transpose(eye_img, (2, 0, 1))  # HWC → CHW

        # Add batch dimension → (1, 3, 32, 32)
        eye_img = np.expand_dims(eye_img, axis=0)

        return eye_img

    def predict(self, eye_img):
        input_eye = self.preprocess_eye(eye_img)

        # Use correct input name
        output = self.session.run(None, {self.input_name: input_eye})[0]

        # Output is softmax → 2 values → take argmax
        pred = np.argmax(output)
        print(f'Predicted class: {pred} (0=Closed, 1=Open)')
        print("Eye img shape before ONNX:", eye_img.shape)

        if pred == 0:
            return 'Closed'
        else:
            return 'Open'
