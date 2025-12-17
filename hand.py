import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandJob:
    def __init__(self, developerMode):
        self.base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(self.options)
        self.developerMode = developerMode
        self.pinching_threshold = 0.04

    def __draw_landmarks(self, image, detection_result):
        for hand in detection_result.hand_landmarks:
            for landmark in hand:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        return image

    def detect(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.detector.detect(mp_image)
        return result

    def show_img(self, frame, result=None):
        if self.developerMode and result is not None:
            frame = self.__draw_landmarks(frame, result)
        cv2.imshow("Hands", frame)
        return None

    def is_thumb_touching_index(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        return distance < self.pinching_threshold

    def is_thumb_touching_middle(self, landmarks):
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]
        distance = math.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)
        return distance < self.pinching_threshold

    def is_thumb_touching_ring(self, landmarks):
        thumb_tip = landmarks[4]
        ring_tip = landmarks[16]
        distance = math.sqrt((thumb_tip.x - ring_tip.x)**2 + (thumb_tip.y - ring_tip.y)**2)
        return distance < self.pinching_threshold

    def is_thumb_touching_pinky(self, landmarks):
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        distance = math.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)
        return distance < self.pinching_threshold

# Example code
if __name__ == "__main__":
    handjob = HandJob(developerMode = True)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = handjob.detect(frame)
        handjob.show_img(frame, result)
        cv2.waitKey(1)

