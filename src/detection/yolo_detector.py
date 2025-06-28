from ultralytics import YOLO


class YOLODetector():
    def __init__(self, model_path='../../data/models/yolov8n.pt', conf_thresh=0.5):
        '''
        Initialize YOLO face detector

        Args:
            model_path (str): path to the model to be used
            conf_thresh (float): threshold for detection face
        '''
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, image):
        '''
        Detect faces in image

        Args:
            image (ndarray): image to detect faces

        Return:
            detections (tuple): (box_tensor, conf)
                    box_tensor (Torch Tensor): Contains face coord in xyxy format
                    conf (float): confidence_threshold
            None: if it couldn't detect a face
        '''
        results = self.model(image, conf=self.conf_thresh, verbose=False)
        if not results:
            return None

        detections = []
        for r in results:
            box_tensor = r.boxes.xyxy[0]
            conf = r.boxes.conf

            detections.append((box_tensor, conf))
        return detections
