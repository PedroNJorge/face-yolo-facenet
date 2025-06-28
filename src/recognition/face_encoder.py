from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms


class FaceEncoder():
    def __init__(self):
        '''
        Initialize FaceEncoder with Pytorch FaceNet
        '''
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

    def get_embedding(self, faces_lst):
        '''
        Extract face embeddings with PyTorch FaceNet

        Args:
            faces_lst (List[ndarray]): List of extracted faces

        Returns:
            List of face embeddings in Tensor form
        '''
        embeddings = map(self.model, faces_lst)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        return map(transform, embeddings)
