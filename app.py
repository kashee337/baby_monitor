import time
from datetime import datetime

import cv2
import streamlit as st

from negaeri_detector import DetectResult, FaceBase, NegaeriDetector, PoseBase

# camera setting
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 5)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)


def check_refresh(refresh_minute_rate=1):
    minute = datetime.now().minute
    second = datetime.now().second
    return minute % refresh_minute_rate == 0 and second == 0


is_ir = True
# detector = FaceBase(is_ir)
detector = PoseBase(is_ir)
negaeri_detector = NegaeriDetector(detector)

st.title('Baby Monitor')
# place holder
baby_status = st.empty()
monitor_canvas = st.empty()

print("start app")
while cap.isOpened:
    ret, img = cap.read()
    time.sleep(0.01)

    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.flip(img, -1)

        ret_detect = negaeri_detector(img)

        if ret_detect == DetectResult.OK:
            baby_status.success("Baby is here!", icon='🐥')
        elif ret_detect == DetectResult.LOST:
            baby_status.warning('Where is baby?', icon='👀')
        elif ret_detect == DetectResult.NG:
            baby_status.error('Help me!', icon='🚨')
        else:
            pass

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

    # FIX: temporary solution
    if check_refresh(1):
        del negaeri_detector
        cap.release()
        st.experimental_rerun()

cap.release()
