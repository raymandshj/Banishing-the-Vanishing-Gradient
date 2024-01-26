from keras.models import load_model
import numpy as np
import cv2


class Camera:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.video_capture.read()
        return frame

    def release(self):
        self.video_capture.release()

class SmileDetector(Camera):
    def __init__(self):
        super().__init__()
        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = load_model('SmileDetector.h5')

    def smile_process(self, frame, gray):
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)

        for (fX, fY, fW, fH) in faces:
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype('float') / 255
            roi = np.expand_dims(roi, axis=0)

            (notSmiling, Smiling) = self.model.predict(roi)[0]

            label = 'Smiling' if Smiling > 0.01 else "Not Smiling"

            color = (0, 0, 255) if label == 'Smiling' else (0, 255, 0)
            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), color, 2)

        return frame

if __name__ == "__main__":
    smile_detector = SmileDetector()

    try:
        while True:
            frame = smile_detector.get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            processed_frame = smile_detector.smile_process(frame, gray)

            cv2.imshow("Smile Detection", processed_frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        smile_detector.release()
        cv2.destroyAllWindows()
