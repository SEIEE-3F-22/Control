import cv2
import numpy as np
import time
from multiprocessing import Process, Pipe


def capture(camera, send_conn, recv_conn):
    cap = cv2.VideoCapture(camera)

    while True:
        ret, frame = cap.read()
        if ret:
            try:
                _ = recv_conn.recv()
            except:
                pass
            send_conn.send(frame)


if __name__ == '__main__':
    recv_conn, send_conn = Pipe()
    p = Process(target=capture, args=(0, send_conn, recv_conn))  # 获取前置摄像头图像进程
    p.daemon = True
    p.start()

    while True:
        frame = recv_conn.recv()

        cv2.imshow('frame', frame)
        cv2.waitKey(1)