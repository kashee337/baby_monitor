import statistics
from abc import ABC, abstractmethod

import numpy as np
import cv2
import mediapipe as mp
from enum import Enum


class DetectResult(Enum):
    LOST = 1
    OK = 2
    NG = 3


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


def calc_yaw_rate(r):
    l1 = np.linalg.norm([r[0], r[2]])
    l2 = r[1]
    return np.rad2deg(np.arctan2(l2, l1))


class RingBuffer():

    def __init__(self, buf_size=10):
        self.idx = 0
        self.buf = []
        self.buf_size = buf_size

    def append(self, v):
        if(len(self.buf) < self.buf_size):
            self.buf.append(v)
        else:
            self.buf[self.idx] = v
            self.idx = (self.idx+1) % self.buf_size
        return

    def mean(self):
        return np.mean(self.buf)

    def median(self):
        return np.median(self.buf)


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

        self.YAW_THRESHOLD = 70
        self.yaw_buf = RingBuffer(10)

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
            u = np.array(
                [right.x-nose.x, -(right.y-nose.y), (right.z-nose.z)])
            v = np.array([left.x-nose.x, -(left.y-nose.y), (left.z-nose.z)])

            yaw_rate = calc_yaw_rate(np.cross(u, v))

            self.yaw_buf.append(yaw_rate)
            print(self.yaw_buf.median())

            return self.yaw_buf.median() < self.YAW_THRESHOLD
        except Exception as e:
            print(e)
            return False

    def detect(self, img):
        if self.is_ir:
            gray_img = np.stack(
                (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),)*3, axis=-1)
            results = self.pose_detector.process(gray_img)
        else:
            results = self.pose_detector.process(img)

        if results.pose_landmarks:
            if self.negaeri_check(results):
                ret = DetectResult.OK
            else:
                ret = DetectResult.NG
        else:
            ret = DetectResult.LOST

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

        if results.detections:
            ret = DetectResult.OK
        else:
            ret = DetectResult.LOST
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
        self.window = [None for _ in range(window_size)]
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
