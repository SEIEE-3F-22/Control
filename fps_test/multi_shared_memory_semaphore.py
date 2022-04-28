import cv2
import numpy as np
from multiprocessing import Process, RawArray, Semaphore
from ctypes import c_uint8


def capture(camera, frame_shared, frame_shape, semaphore):
    cap = cv2.VideoCapture(camera)
    frame_shared_np = np.frombuffer(frame_shared, dtype='uint8').reshape(frame_shape)
    while True:
        ret, frame = cap.read()
        if ret:
            with semaphore:
                np.copyto(frame_shared_np, np.uint8(frame))


if __name__ == '__main__':
    semaphore = Semaphore(1)
    frame_shape = (480, 640, 3)
    frame_shared = RawArray(c_uint8, frame_shape[0] * frame_shape[1] * frame_shape[2])

    p = Process(target=capture, args=(0, frame_shared, frame_shape, semaphore))
    p.daemon = True
    p.start()

    while True:
        with semaphore:
            frame = np.frombuffer(frame_shared, dtype='uint8').reshape(frame_shape)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)