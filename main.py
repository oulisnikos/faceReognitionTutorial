# Import necessary packages
from typing import NamedTuple

import cv2
import mediapipe as mp

# Define mediapipe Face detector
face_detection = mp.solutions.face_detection.FaceDetection(0.8)


def detector(frame: any, result: NamedTuple):
    count = 0
    height, width, channel = frame.shape
    # Convert frame BGR to RGB colorspace
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect results from the frame
    result = face_detection.process(imgRGB)
    print(result)
    return frame


def mainProc():
    # Reading image
    img = cv2.imread("image.png")
    result: NamedTuple
    detector(img, result)
    for count, detection in enumerate(result.detections):
        print(detection)
    count += 1
    print(count)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mainProc()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
