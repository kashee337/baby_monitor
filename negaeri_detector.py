import statistics
from abc import ABC, abstractmethod

import numpy as np
import cv2
import mediapipe as mp


class Detector(ABC):

    def __init__(self, is_ir=False):
        pass

    @abstractmethod
    def detect(self, img):
        pass

    @abstractmethod
    def draw(self, img):
        pass

    @abstractmethod
    def close(self):
        pass


class PoseBase(Detector):

    def __init__(self, is_ir=False):
        # pose detector
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.is_ir = is_ir

    def negaeri_check(self, results):
        if not results.pose_landmarks:
            return False
        try:
            RIGHT_SHOULDER = 12
            LEFT_SHOULDER = 11
            NOSE = 0

            nose = results.pose_landmarks.landmark[NOSE]
            right = results.pose_landmarks.landmark[RIGHT_SHOULDER]
            left = results.pose_landmarks.landmark[LEFT_SHOULDER]

            center_x = (nose.x+right.x+left.x)/3
            center_y = (nose.y+right.y+left.y)/3

            u = np.array([right.x-center_x, right.y-center_y])
            v = np.array([left.x-center_x, left.y-center_y])

            return np.cross(u, v) >= 0
        except:
            return False

    def detect(self, img):
        if self.is_ir:
            gray_img = np.stack(
                (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),)*3, axis=-1)
            results = self.pose_detector.process(gray_img)
        else:
            results = self.pose_detector.process(img)

        ret = self.negaeri_check(results)

        return ret, results

    def draw(self, img, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

    def close(self):
        self.pose_detector.close()


class FaceBase(Detector):

    def __init__(self, is_ir=False):
        # face detector
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils

        self.is_ir = is_ir

    def detect(self, img):

        if self.is_ir:
            gray_img = np.stack(
                (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),)*3, axis=-1)
            results = self.face_detection.process(gray_img)
        else:
            results = self.face_detection.process(img)

        ret = results.detections is not None
        return ret, results

    def draw(self, img, results):
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(img, detection)

    def close(self):
        self.face_detctor.close()


class NegaeriDetector:

    def __init__(self, detector, window_size=5, is_draw=True):

        self.detector = detector
        self.is_draw = is_draw

        # for filtering
        self.window = [False for _ in range(window_size)]
        self.idx = 0

    def __del__(self):
        self.detector.close()

    def __call__(self, img):

        ret, results = self.detector.detect(img)
        self.window[self.idx] = ret
        if self.is_draw:
            self.detector.draw(img, results)
        self.idx += 1
        self.idx %= len(self.window)

        return statistics.mode(self.window)
