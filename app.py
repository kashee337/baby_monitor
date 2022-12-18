import statistics
import time
from datetime import datetime

import cv2
import mediapipe as mp
import streamlit as st

# camera setting
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)


class NegaeriDetector:

    def __init__(self, window_size=5, is_draw=True):
        # face detector
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
        self.is_draw = is_draw
        if is_draw:
            self.mp_drawing = mp.solutions.drawing_utils

        # for filtering
        self.window = [False for _ in range(window_size)]
        self.idx = 0
        
    def __del__(self):
        self.face_detection.close()
        
    def __call__(self, img):

        results = self.face_detection.process(img)

        if results.detections:
            self.window[self.idx] = True
            if self.is_draw:
                for detection in results.detections:
                    self.mp_drawing.draw_detection(img, detection)
        else:
            self.window[self.idx] = False

        self.idx += 1
        self.idx %= len(self.window)

        return statistics.mode(self.window)


negaeri_detector = NegaeriDetector()

st.title('Baby Monitor')

# place holder
baby_status = st.empty()
monitor_canvas = st.empty()

while cap.isOpened:
    ret, img = cap.read()
    time.sleep(0.01)

    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, -1)

        if negaeri_detector(img):
            baby_status.success("Baby is here!", icon='🐥')
        else:
            baby_status.warning('Where is baby?', icon='🚨')

        cv2.putText(img,
                    text=f'{datetime.now()}',
                    org=(10, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_4)

        monitor_canvas.image(img)
    else:
        print("missing")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
