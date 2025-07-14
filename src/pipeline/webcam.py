import cv2
import questionary
import numpy as np
from .processor import Processor


class InteractiveWebcam:
    def __init__(self, processor: Processor):
        self.processor = processor

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Press 'd' to interact, 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):
                self._handle_frame_interactively(frame)
            elif key == ord('q'):
                break

    def _handle_frame_interactively(self, frame: np.ndarray):
        """Unified Questionary workflow for both known/unknown faces."""
        for result in self.processor.process_frame(frame):
            if result["identity"]:  # Known face
                choice = questionary.select(
                    f"Recognized: {result['identity']}. Action:",
                    choices=["Update last seen", "Ignore"]
                ).ask()
                if choice == "Update last seen":
                    self._update_redis(result["identity"])
            else:  # Unknown face
                choice = questionary.select(
                    "Unknown face. Action:",
                    choices=["Save as unknown", "Register new person", "Ignore"]
                ).ask()
                if choice == "Register new person":
                    self._register_person(frame, result["face"])

    def _update_redis(self, person_id):
        """Update last_seen timestamp in Redis."""
        self.processor.face_matcher.update_last_seen(person_id)

    def _register_person(self, frame, bbox):
        """Replace CLI registration logic."""
        name = questionary.text("Enter name:").ask()
        self.processor.add_person(name, frame, bbox)
