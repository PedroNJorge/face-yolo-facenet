import cv2
import questionary
import numpy as np
from torch import Tensor
from ..pipeline.processor import Processor


class InteractiveWebcam:
    def __init__(self):
        self.processor = Processor()

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Press 'd' to interact, 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):
                self._handle_frame_interactively(frame)
            elif key == ord('d'):
                self._handle_person_deletion()
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _handle_frame_interactively(self, frame: np.ndarray):
        """Unified Questionary workflow for both known/unknown faces."""
        for result in self.processor.process_frame(frame):
            if result["identity"]:  # Known face
                person_metadata = result["profile"].metadata
                cv2.imshow(result["face"])
                questionary.print(
                        f"""
                        ✔ Found: {person_metadata.name}
                        First Seen: {person_metadata.first_seen}
                        Last Seen: {person_metadata.last_seen}
                        """,
                        style="bold fg:green"
                )
                questionary.print(
                        f"""
                            Snapshot Detection Confidence: {result["detection_confidence"]:.2%}
                            Database Match Confidence: {result["match_confidence"]:.2%}
                        """,
                        style="fg:blue"
                )
            else:  # Unknown face
                choice = questionary.select(
                    "Unknown face. Action:",
                    choices=["Save as unknown", "Register new person"]
                ).ask()
                if choice == "Register new person":
                    self._register_person(result["face"],
                                          result["detection_confidence"],
                                          result["bbox"],
                                          result["embedding"]
                                          )

    def _handle_person_deletion(self):
        pass

    def _register_person(self, face: np.ndarray, dconf: float, bbox: float, embedding: Tensor):
        """Implement new known person logic"""
        name = questionary.text("Enter name:").ask()
        self.processor.add_person(name, face, dconf, bbox, embedding)
