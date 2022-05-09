import time
import cv2
import numpy as np
import math
from fuzzy_control import *
# from driver import driver


# car = driver()
HEIGHT = 480
WIDTH = 640
window_start = 130
window_end = 450
window_L = window_end - window_start
NUM_AP = 18 # in fact is 19. 1 point less than fuzzy NUM_AP

'''
def homomorphic_filter(src, d0=10, rl=0.6, rh=2, c=4, h=4.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - rl) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + rl  # 衰减低频，增强高频
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst
'''

def get_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)  # 00 00
    return opening

def get_start_point(binary, aim_y):
    global window_start, window_end, HEIGHT
    piece = binary[aim_y, window_start: window_end]
    list_ = []
    lane_base_ = []
    last_unit = 0
    flag_start = 0
    lane_base_flag_ = 1  # 0: 没有lane_base  1: 正常lane_base  2:在十字线上


    # piece11 = binary[aim_y - 40, window_start: window_end]
    # piece12 = binary[aim_y, window_start: window_end]
    # black_detected = 2*(window_end - window_start) - (np.sum(piece11)+np.sum(piece12))/255
    # print(black_detected)
    # if black_detected > 220:
    #     start_point = [320, aim_y]
    #     lane_base_flag_ = 2
    piece1 = binary[aim_y - 40:aim_y, window_start: window_end]
    histogram_y = np.sum(piece1, axis=1)
    area = np.sum(histogram_y)/255
    total_area = 40 * (window_end - window_start)
    ratio = 1 - area/total_area
    # print(ratio)
    if ratio > 0.6:  # 看到十字
        start_point = [324, aim_y]
        lane_base_flag_ = 2
    else:
        for i in range(len(piece)):
            if piece[i] == 0:  # black
                if last_unit == 0:
                    list_.append(i)
                    flag_start = 1
                else:
                    list_.append(i)
                last_unit = 1
                # if i == len(piece):
                #     lane_base_.append(int(np.average(list_)))
            else:
                last_unit = 0
                # print(list_)
                if len(list_) > 40 and flag_start == 1:  # 40 是赛道宽度阈值， 防止误识别其他的黑色。
                    # print(list_)
                    lane_base_.append(int(np.average(list_)))
                    list_ = []
                else:
                    list_ = []
                flag_start = 0
        if len(lane_base_) < 1:
            print('no base line detected')
            lane_base_flag_ = 0
            start_point = [320, aim_y]
        else:
            # print(lane_base_)
            midpoint = lane_base_[0] + window_start
            start_point = [midpoint, aim_y]
        # print(start_point)

    return start_point, lane_base_flag_

def new_y(y):
    global HEIGHT
    if y < 0:
        y = 0
    if y > HEIGHT-1:
        y = HEIGHT-1
    _new_y = HEIGHT-1 - y
    return _new_y

# 只需要0°-180°，选择cos.  start_point = [x, y]
def get_aim_point(binary_img, start_point, window, start_angle, step):  # window = window_L, angle = 90, step = 5
    global NUM_AP
    aim_point = [[start_point[0], new_y(start_point[1])]]  # 转为笛卡尔坐标系
    x, y = start_point[0], new_y(start_point[1])
    angle = start_angle
    # print(x, y, angle)
    flag_info_ = 0
    num_point = 0
    max_dist = 0
    while num_point < NUM_AP - 1:
        break_flag = 0
        next_x = int(x + step * math.cos(angle))
        next_y = int(y + step * math.sin(angle))
        left_limit = right_limit = []

        if angle == 0:
            break_flag = 1
            flag_info_ = 1
            # print("error angle!")
            break

        for x_ in range(next_x, int(next_x - 1 / 2 * window), -1):
            y_ = int(- 1 / math.tan(angle) * (x_ - next_x) + next_y)
            left_limit = [x_, y_]
            if binary_img[int(new_y(y_))][int(x_)] == 255:  # 判断是否达到边界
                break
        for x_ in range(next_x + 1, int(next_x + 1 / 2 * window)):
            y_ = int(- 1 / math.tan(angle) * (x_ - next_x) + next_y)
            right_limit = [x_, y_]
            if binary_img[int(new_y(y_))][int(x_)] == 255:  # 判断是否达到边界
                break

        # 判断是否有十字线：
        if math.sqrt((right_limit[0] - left_limit[0]) ** 2 + (right_limit[1] - left_limit[1]) ** 2) > 150:
            break_flag = 1
            flag_info_ = 2
            print("there is cross1")
        # print(math.sqrt((right_limit[0] - left_limit[0])**2 + (right_limit[1] - left_limit[1])**2))
        # print(left_limit, right_limit, angle, num_point)

        _midpoint = [1 / 2 * (left_limit[0] + right_limit[0]), 1 / 2 * (left_limit[1] + right_limit[1])]
        x_, y_ = int(_midpoint[0]), int(_midpoint[1])

        if binary_img[int(new_y(_midpoint[1]))][int(_midpoint[0])] == 255:  # 求得的预瞄点是白色，非法
            break_flag = 1
            flag_info_ = 3
            # print("lines are broken")

        # dist_ = math.sqrt((x_-x)**2 + (y_-y)**2)
        # if dist_ > max_dist:
        #     max_dist = dist_

        if math.sqrt((x_-x)**2 + (y_-y)**2) > 28:  # 前后两个点距离太远，有十字线
            break_flag = 1
            flag_info_ = 2
            print("there is cross2")

        if break_flag == 1:
            break

        # 更新下一次遍历的角度，起始点
        angle = math.acos((_midpoint[0] - x) / math.sqrt((_midpoint[0] - x) ** 2 + (_midpoint[1] - y) ** 2))
        x, y = x_, y_
        aim_point.append([x, y])
        num_point = num_point + 1

    # print("max_dist is :", max_dist)
    return aim_point, flag_info_

def k_compute(pt1, pt2):
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    k = (x1 - x2)/(y1 - y2)

    return abs(k)

#  计算平均曲率
def get_K(_aim_point, _num_ap):
    if _num_ap > 13:
        k1 = k_compute(_aim_point[0], _aim_point[5])
        k2 = k_compute(_aim_point[7], _aim_point[13])
        if _num_ap == 14:
            k3 = k2
        else:
            k3 = k_compute(_aim_point[13], _aim_point[_num_ap-1])
        _k_avg = 0.5 * k1 + 0.3 * k2 + 0.2 * k3
        _k_var = (k1-k2)**2 + (k2-k3)**2 + (k1-k3)**2
    elif _num_ap > 4:
        k1 = k_compute(_aim_point[0], _aim_point[4])
        if _num_ap == 5:
            k2 = k1
        else:
            k2 = k_compute(_aim_point[4], _aim_point[_num_ap - 1])
        _k_avg = 0.6 * k1 + 0.4 * k2
        _k_var = 3 * (k1 - k2) ** 2
    else:  # 点数过少，认为有十字
        _k_avg = 0.1
        _k_var = 0

    if abs(_k_avg) > 1.5:
        _k_avg = 1.5

    return abs(_k_avg), _k_var


def get_warpPerspective(apts):
    M = np.matrix([[-3.14848462e+00, -2.60509133e+00, 1.38597252e+03],
                   [-3.59759442e-02, -5.75615107e+00, 1.26991485e+03],
                   [-6.70390691e-05, -8.06154144e-03, 1.00000000e+00]])
    new_apts = []
    for k in range(len(apts)):
        x, y = apts[k][0], new_y(apts[k][1])  # 转为原图x y
        # x, y = apts[k][0], apts[k][1]
        vector = np.matrix([x, y, 1]).T
        new_vector = M * vector
        warped_x, warped_y = int(new_vector[0] / new_vector[2]), int(new_vector[1] / new_vector[2])
        new_apts.append([warped_x, warped_y])
    # print(new_apts)
    # print('warp')
    return new_apts

def get_warped_binary(img):
    h, w = img.shape
    M = np.matrix([[-3.14848462e+00, -2.60509133e+00, 1.38597252e+03],
                   [-3.59759442e-02, -5.75615107e+00, 1.26991485e+03],
                   [-6.70390691e-05, -8.06154144e-03, 1.00000000e+00]])
    binary_img = cv2.warpPerspective(img, M, (w, h))  # 投影过后的二值图

    return binary_img