from facenet_pytorch import InceptionResnetV1
import torch
import torchvision.transforms as transforms
from typing import Tuple, List
import numpy as np


class FaceEncoder():
    def __init__(self):
        '''Initialize FaceEncoder with Pytorch FaceNet'''
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def get_embedding(self, face_conf: List[Tuple[np.ndarray, float]]):
        '''
        Extract face embeddings with PyTorch FaceNet

        Args:
            face_conf (List[(face, conf)]): Has tuple (face, conf)

        Returns:
            embeddings: List[Tuple(embedding, conf)]
                embedding: Normalized Torch Tensor
                conf: float
        '''
        # Unzip faces and confidences
        faces, confs = zip(*face_conf)

        # Process each face
        embeddings = []
        for face, conf in zip(faces, confs):
            # Convert to tensor and normalize
            face_tensor = self.transform(face).unsqueeze(0)  # Add batch dimension

            # Get embedding
            with torch.no_grad():
                embedding = self.model(face_tensor).squeeze(0)  # Remove batch dimension

            embeddings.append((embedding, conf))

        return embeddings


'''
TEST FOR SIMILARITY AND HOW IT SHOULD BE DONE


# Read the image file
im0 = cv2.cvtColor(cv2.imread('../../data/unknown_faces/face_0.jpg'), cv2.COLOR_BGR2RGB)
im1 = cv2.cvtColor(cv2.imread('../../data/unknown_faces/face_1.jpg'), cv2.COLOR_BGR2RGB)
imgs = [im0, im1]
conf = [0.9, 0.8]
inp = zip(imgs, conf)
fac = FaceEncoder()
results = fac.get_embedding(inp)
# Extract embeddings
embedding1 = results[0][0].unsqueeze(0)  # [1, 512]
embedding2 = results[1][0].unsqueeze(0)  # [1, 512]

# Calculate cosine similarity
r = cosine_similarity(embedding1, embedding2).item()
print(f"Cosine similarity: {r:.4f}")
'''
