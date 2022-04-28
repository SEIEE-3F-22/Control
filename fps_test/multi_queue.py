import cv2
import numpy as np
import time
from multiprocessing import Process, Queue, Pipe


def capture(camera, queue):
    cap = cv2.VideoCapture(camera)

    while True:
        ret, frame = cap.read()
        # if not queue.empty():
        #     try:
        #         _ = queue.get(block=False)
        #     except:
        #         pass
        
        if ret:
            queue.put(frame)


if __name__ == '__main__':
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

    q = Queue()  # 传输前置摄像头图像
    p = Process(target=capture, args=(0, q))
    p.daemon = True
    p.start()

    timeStamp = time.time() - 0.5  # 避免后续运算出现NaN
    counter = 0
    fps_avg = 0

    while True:
        lastStamp = timeStamp
        timeStamp = time.time()
        
        frame = q.get()
        # 前置鱼眼摄像头校正
        frame = cv2.fisheye.undistortImage(frame, K, D=D, Knew=Knew)

        fps = 1 / (timeStamp - lastStamp)
        counter += 1
        fps_avg += fps
        print("\rFPS: {0}\tAverage FPS: {1}".format(
            fps, fps_avg / counter), end='')
