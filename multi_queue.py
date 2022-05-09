import cv2
import numpy as np
from multiprocessing import Process, Pipe, Queue
from ctypes import c_uint8
import time
from lane_detect import *
from driver import driver

car = driver()

HEIGHT = 480
WIDTH = 640
window_start = 130
window_end = 450
window_L = window_end - window_start
NUM_AP = 18

class camera:
    def __init__(self):
        self.cap1 = cv2.VideoCapture(1)

    def capture(self, cap, queue):
        while True:
            ret, frame = cap.read()
            if ret:
                try:
                    _ = queue.get(False)
                except:
                    pass
                queue.put(frame)

    def __del__(self):
        self.cap1.release()
        print('camera shut')


if __name__ == '__main__':
    cam = camera()
    # queue
    q = Queue(1)
    p = Process(target=cam.capture, args=(cam.cap1, q))
    p.daemon = True
    p.start()

    # frame test
    timeStamp = time.time() - 0.5  # 避免后续运算出现NaN
    counter = 0
    fps_avg = 0

    N = 0
    for i in range(5): car.set_speed(0.5, 0.5)
    while True:
        t_start = time.time()
        img = q.get()
        binary_img = get_binary(img)
        binary_img = get_warped_binary(binary_img)
        start_point, lane_base_flag = get_start_point(binary_img, HEIGHT-1)
        aim_point, flag_info = get_aim_point(binary_img, start_point, 200, math.pi / 2, 15)  # aim_point 坐标左下角为原点
        num_ap = len(aim_point)  # 预瞄点数量. 正常应为NUM_AP.
        k_avg, k_var = get_K(aim_point, num_ap)  # 曲率。用这个变量衡量赛道的弯曲程度。
        if num_ap > 0:
            target_speed = fuzzy_controller(k_avg, num_ap) / 4  # map to 0-40
            vl, vr, theta_ap, lambda_, steer_case = get_steer(aim_point, num_ap, target_speed, k_avg, k_var)
        else:
            vl, vr = 0, 0
            theta_ap, target_speed, lambda_ ,steer_case= 0,0,0,5
        w = lambda_

        ####################### 可视化与调试 #########################
        N += 1
        string_info = ['normal', 'error angle', 'there is cross', 'lines are broken', 'on the cross']
        if not flag_info:
            flag_info = 0
        if lane_base_flag == 2:
            flag_info = 4
        log_info = string_info[flag_info]

        print(N, target_speed, vl, vr, theta_ap, lambda_, k_avg, '\n')

        show = 0
        if show == 1:
            binary_3chanl = img
            binary_3chanl[:, :, 0], binary_3chanl[:, :, 1], binary_3chanl[:, :, 2] = binary_img, binary_img, binary_img
            if len(aim_point) > 0:
                for j in range(len(aim_point)):
                    aim_point[j][1] = new_y(aim_point[j][1])  # 绘图时转换为原图像坐标系
                    x_, y_ = aim_point[j][0], aim_point[j][1]
                    cv2.circle(binary_3chanl, (x_, y_), 2, (0, 0, 255))
            cv2.circle(binary_3chanl, (324, 479), 2, (213, 155, 91))
            cv2.rectangle(binary_3chanl, (window_start, HEIGHT - 40), (window_end, HEIGHT - 1), (0, 255, 0), 1)
            cv2.putText(binary_3chanl, 'k:{:+.3f} k_var:{:+.3f} theta:{:+.3f} steer:{:+.3f}'.format(k_avg,k_var,theta_ap,w),
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(binary_3chanl, 'speed:{:+.3f} vl:{:+.3f} vr:{:+.3f} steer_case:{:+.3f}'.format(target_speed, vl, vr, steer_case),
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(binary_3chanl, 'log: {}'.format(log_info),
                        (20, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # cv2.imwrite('./debug_fig/now/' + str(N) + '.jpg', binary_3chanl)
            # cv2.imshow('img', binary_3chanl)
            # cv2.waitKey(1)

        ###################### 决策与下发指令 ####################
        # set speed to hardware
        # vl, vr = 0, 0
        # 状态机
        obj_class = 0  # 罗兄的检测结果
        if flag_info != 4:
            for i in range(5): car.set_speed(vr, vl)

        # on the cross
        if flag_info == 4:
            t0 = time.time()
            # 跳过十字线看前方的路况
            start_point, _ = get_start_point(binary_img, HEIGHT - 70)
            aim_point, _ = get_aim_point(binary_img, start_point, 200, math.pi / 2, 15)
            num_ap = len(aim_point)
            '''
            for i in range(5): car.set_speed(0, 0)
            binary_3chanl = img
            binary_3chanl[:, :, 0], binary_3chanl[:, :, 1], binary_3chanl[:, :, 2] = binary_img, binary_img, binary_img
            if len(aim_point) > 0:
                for j in range(len(aim_point)):
                    aim_point[j][1] = new_y(aim_point[j][1])  # 绘图时转换为原图像坐标系
                    x_, y_ = aim_point[j][0], aim_point[j][1]
                    cv2.circle(binary_3chanl, (x_, y_), 2, (0, 0, 255))
            cv2.imshow('img', binary_3chanl)
            cv2.waitKey(0)
            '''
            target_speed = 18
            vl, vr, w, theta_far = get_cross_steer(aim_point, num_ap, target_speed)
            print("cross_control:", vl, vr, lambda_, theta_far)
            counter = 0
            while counter < 40:  # 阈值，开环
                for i in range(5): car.set_speed(vr, vl)
                counter += 1
            t1 = time.time()
            print("cross", t1 - t0)
            vl, vr = 0, 0
            for i in range(20): car.set_speed(vr, vl)
            # while True:
            #     pass

            # 在十字线处做不同动作
            '''
            if obj_class == 0:  # 向右转
                vr, vl = -10, 10
                counter0 = 0
                while counter0 < 500:
                    for i in range(20): car.set_speed(vr, vl)
                    counter0 += 1
                    print('counter0:', counter0)

            if obj_class == 1:  # 向左转
                vr, vl = 10, -10
                counter1 = 0
                while counter1 < 500:
                    for i in range(20): car.set_speed(vr, vl)
                    counter1 += 1
                    print('counter1:', counter1)
            '''

        t_end = time.time()
        # print('total_time:', t_end - t_start)
        fps = 1 / (t_end - t_start)
        counter += 1
        fps_avg += fps

        # print("\rFPS: {0}\tAverage FPS: {1} \n".format(fps, fps_avg / counter), end='')

