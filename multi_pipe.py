import cv2
import numpy as np
from multiprocessing import Process, Pipe
from ctypes import c_uint8
import time
from lane_detect import *
from driver import driver

car = driver()

HEIGHT = 480
WIDTH = 640
window_start = 170
window_end = 470
window_L = window_end - window_start
NUM_AP = 23

class camera:
    def __init__(self):
        self.cap1 = cv2.VideoCapture(1)
        
    def capture(self, cap, send_conn, recv_conn):
        while True:
            ret, frame = cap.read()
            if ret:
                send_conn.send(frame)
                
    def __del__(self):
        self.cap1.release()
        print('camera shut')


if __name__ == '__main__':
    
    cam = camera()
    send_conn, recv_conn = Pipe()

    p = Process(target=cam.capture, args=(cam.cap1, send_conn, recv_conn))
    p.daemon = True
    p.start()
    
    # frame test
    timeStamp = time.time() - 0.5  # 避免后续运算出现NaN
    counter = 0
    fps_avg = 0

    N=0
    while True:
        
        t_start = time.time()
        
        img = recv_conn.recv()
        
        binary_img = get_binary(img)
        
        start_point, lane_base_flag = get_start_point(binary_img)
        aim_point, flag_info = get_aim_point(binary_img, start_point, 200, math.pi / 2, 10)  # aim_point 坐标左下角为原点
        num_ap = len(aim_point)  # 预瞄点数量. 正常应为NUM_AP.
        # print(cross_flag)
        
        k_avg = get_K(aim_point, num_ap)  # 曲率。用这个变量衡量赛道的弯曲程度。
        if num_ap > 0:
            target_speed = fuzzy_controller(k_avg, num_ap) / 2  # map to 0-40
            vl, vr, theta_ap, lambda_ = get_steer(aim_point, num_ap, target_speed)
        else:
            vl, vr = 0, 0
        w = 100 * (vr - vl)
        
        N +=1
        string_info = ['normal', 'error angle', 'there is cross', 'lines are broken', 'on the cross']
        if not flag_info:
            flag_info = 0
        if lane_base_flag == 2:
            flag_info = 4
        log_info = string_info[flag_info]
        
        print(N, target_speed, vl, vr, theta_ap, lambda_,flag_info,'\n')
        
        if flag_info==4:
            vl,vr = 0,0
        # set speed to hardware
        vl, vr = 0,0
        for i in range(20): car.set_speed(vr, vl)

        t_end = time.time()
        fps = 1 / (t_end - t_start)
        counter += 1
        fps_avg += fps
        
        print("\rFPS: {0}\tAverage FPS: {1} \n".format(fps, fps_avg / counter), end='')
        
        show = 1
        if show == 1:
            binary_3chanl = img
            binary_3chanl[:, :, 0], binary_3chanl[:, :, 1], binary_3chanl[:, :, 2] = binary_img, binary_img, binary_img
            if len(aim_point) > 0:
                for j in range(len(aim_point)):
                    aim_point[j][1] = new_y(aim_point[j][1])  # 绘图时转换为原图像坐标系
                    x_, y_ = aim_point[j][0], aim_point[j][1]
                    cv2.circle(binary_3chanl, (x_, y_), 2, (0, 0, 255))
            cv2.rectangle(binary_3chanl, (window_start, HEIGHT - 40), (window_end, HEIGHT - 1), (0, 255, 0), 1)
            cv2.putText(binary_3chanl, 'k:{:+.3f} speed:{:+.3f} steer:{:+.3f}'.format(k_avg, target_speed, w),
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(binary_3chanl, 'vl:{:+.3f} vr:{:+.3f} '.format(vl, vr),
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(binary_3chanl, 'log: {}'.format(log_info),
                        (20, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # cv2.imwrite('./debug_fig/' + str(N) + '.jpg', binary_3chanl)
            cv2.imshow('img', binary_3chanl)
            
        k = cv2.waitKey(1)
        if k == ord('q'):  # 按q结束检测
            break
