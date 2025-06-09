import onnxruntime as ort
import numpy as np
import cv2

class YawnDetector:
    def __init__(self, model_path='data/yawn_model.onnx'):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, mouth_img):
        img = cv2.resize(mouth_img, (64, 64)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1,64,64,3)
        return img

    def predict(self, mouth_img):
        input_img = self.preprocess(mouth_img)
        prediction = self.session.run(None, {self.input_name: input_img})[0][0][0]
        return 'Yawn' if prediction > 0.5 else 'No Yawn'
