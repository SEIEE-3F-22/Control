import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)  # 前置摄像头

counter = 0
fps_avg = 0
timeStamp = time.time() - 0.5  # 避免后续运算出现NaN

while True:
    lastStamp = timeStamp
    timeStamp = time.time()
    
    _, frame = cap.read()
    
    fps = 1 / (timeStamp - lastStamp)
    counter += 1
    fps_avg += fps
    print("\rFPS: {0}\tAverage FPS: {1}".format(
        fps, fps_avg / counter), end='')



