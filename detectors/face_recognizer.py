# detectors/face_recognizer.py

from facenet_pytorch import InceptionResnetV1
import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import glob

class FaceRecognizer:
    def __init__(self, driver_images_path='data/driver_images'):
        # Load embedding model
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

        # Load known driver faces
        self.known_embeddings = []
        self.known_names = []

        image_paths = glob.glob(os.path.join(driver_images_path, '*.jpg'))

        print(f'Loading {len(image_paths)} driver faces...')

        for path in image_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            img = Image.open(path).convert('RGB')
            embedding = self.get_embedding(img)

            self.known_embeddings.append(embedding)
            self.known_names.append(name)

        if len(self.known_embeddings) > 0:
            self.known_embeddings = np.stack(self.known_embeddings)

    def get_embedding(self, img):
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        img_tensor = transform(img).unsqueeze(0)  # 1x3x160x160
        with torch.no_grad():
            embedding = self.model(img_tensor)
        embedding = embedding.squeeze().numpy()
        embedding = embedding / np.linalg.norm(embedding)  # L2 norm
        return embedding

    def recognize(self, face_img):
        # Convert OpenCV image to PIL
        img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

        # Get embedding
        embedding = self.get_embedding(img)
        # Compare to known
        if len(self.known_embeddings) == 0:
            return 'Unknown'

        similarities = np.dot(self.known_embeddings, embedding)
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        # Threshold (tuneable)
        if best_score > 0.5:
            return "{} ({:.2f})".format(self.known_names[best_idx], best_score)
        else:
            return 'Unknown'
