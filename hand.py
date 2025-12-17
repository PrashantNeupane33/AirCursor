import cv2
import math
import webbrowser
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

link = "https://google.com" #here is what u need

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

    def __finger_states(self, landmarks):
        states = {}
        states['thumb'] = landmarks[4].x > landmarks[2].x
        states['index'] = landmarks[8].y < landmarks[6].y
        states['middle'] = landmarks[12].y < landmarks[10].y
        states['ring'] = landmarks[16].y < landmarks[14].y
        states['pinky'] = landmarks[20].y < landmarks[18].y
        return states

    def detect_gesture(self, landmarks):
        states = self.__finger_states(landmarks)
        if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
            return True, "V_SIGN"

        if states["thumb"] and not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            return True, "THUMBS_UP"

        if states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            return True, "Index Pointing"

        if not states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
            self.open_link()
            return True, "Middle Pointing"

        if not states["index"] and not states["middle"] and states["ring"] and not states["pinky"]:
            return True, "Ring Pointing"

        if not states["index"] and not states["middle"] and not states["ring"] and states["pinky"]:
            return True, "Pinky Pointing"

        if states["index"] and not states["middle"] and not states["ring"] and states["pinky"]:
            return True, "Rock On!!!"

        if not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            return True, "Fist"

        return False, None
    
    def check_for_gesture(self, landmark, frame):
        if self.is_thumb_touching_index(landmark):
            cv2.putText(frame, "Index and thumb pinched",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        if self.is_thumb_touching_middle(landmark):
            cv2.putText(frame, "Middle and thumb pinched",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        if self.is_thumb_touching_ring(landmark):
            cv2.putText(frame, "Ring and thumb pinched",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        if self.is_thumb_touching_pinky(landmark):
            cv2.putText(frame, "Pinky and thumb pinched",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        gesture_detected, gesture = self.detect_gesture(landmark)
        if gesture_detected and gesture is not None:
            cv2.putText(frame, gesture,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

    def open_link(self):
        webbrowser.open(link)
        return

# Example code
if __name__ == "__main__":
    handjob = HandJob(developerMode = True)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = handjob.detect(frame)
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            handjob.check_for_gesture(landmarks, frame)
        handjob.show_img(frame, result)
        cv2.waitKey(1)

