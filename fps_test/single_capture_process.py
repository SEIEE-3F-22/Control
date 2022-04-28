import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)  # 前置摄像头

# 鱼眼摄像头参数
K = np.array([
    [265.26929481493994, 0.0, 336.2310483246026],
    [0.0, 265.5493364501116, 227.57455725938897],
    [0.0, 0.0, 1.0]
])
D = np.array([
    [-0.04460482096772315], [0.12979934182124128],
    [-0.20708138065438658], [0.1025151811608204]
])
Knew = K.copy()

counter = 0
fps_avg = 0
timeStamp = time.time() - 0.5  # 避免后续运算出现NaN

while True:
    lastStamp = timeStamp
    timeStamp = time.time()
    
    _, frame = cap.read()
    # 前置鱼眼摄像头校正
    frame = cv2.fisheye.undistortImage(frame, K, D=D, Knew=Knew)
    
    fps = 1 / (timeStamp - lastStamp)
    counter += 1
    fps_avg += fps
    print("\rFPS: {0}\tAverage FPS: {1}".format(
        fps, fps_avg / counter), end='')
