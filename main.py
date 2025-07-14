from src import InteractiveWebcam, Processor, FaceExtractor, FaceEncoder


def main():
    processor = Processor(FaceExtractor(), FaceEncoder())
    app = InteractiveWebcam(processor)
    app.run()


if __name__ == "__main__":
    main()

"""
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Webcam
    pipeline = Processor(FaceExtractor(), FaceEncoder())

    Q_KEY = ord('q')
    D_KEY = ord('d')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Face Recognition', frame)

        key = cv2.waitKey(1)

        if key == Q_KEY:
            break
        elif key == D_KEY:
            print("Processing frame")
            pipeline.process_frame(frame)
            print("Done!")

    cap.release()
    cv2.destroyAllWindows()
"""
