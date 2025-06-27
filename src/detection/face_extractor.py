from .yolo_detector import YOLODetector
import cv2


class FaceExtractor():
    def __init__(self):
        '''
        Initialize FaceExtractor with YOLODetector
        '''
        self.detector = YOLODetector()

    def extract_faces(self, image, required_size=(160, 160)):
        '''
        Extract faces from image and resizes them for recognition later

        Args:
            image (ndarray): Target Image
            required_size (Tuple(Int)): Size required for Pytorch FaceNet

        Return:
            List of resized faces
        '''
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = self.detector.detect(image)
        faces = [rgb_image[y1:y2, x1:x2] for (x1, y1, x2, y2, _) in detections]
        resized_faces = map(lambda x: cv2.resize(x, required_size), faces)

        return resized_faces
