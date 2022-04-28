import time
import cv2
import numpy as np
import math
from fuzzy_control import *
# from driver import driver


# car = driver()
HEIGHT = 480
WIDTH = 640
window_start = 170
window_end = 470
window_L = window_end - window_start
NUM_AP = 23 # in fact is 19. 1 point less than fuzzy NUM_AP

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

def get_start_point(binary):
    global window_start, window_end, HEIGHT
    aim_y = HEIGHT-1
    piece = binary[aim_y, window_start: window_end]
    list_ = []
    lane_base_ = []
    last_unit = 0
    flag_start = 0
    lane_base_flag_ = 1  # 0: 没有lane_base  1: 正常lane_base  2:在十字线上
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
    if lane_base_flag_ == 0:
        piece1 = binary[aim_y-8:aim_y+3, window_start: window_end]
        histogram_y = np.sum(piece1, axis=1)
        area = np.sum(histogram_y)
        total_area = 10 * len(piece)
        if area > total_area * 0.7:
            lane_base_flag_ = 2
            print('on the cross')

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
    while num_point < NUM_AP - 1:  # 加上base_lane的点，共13个点
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
        if math.sqrt((right_limit[0] - left_limit[0]) ** 2 + (right_limit[1] - left_limit[1]) ** 2) > 100:
            break_flag = 1
            flag_info_ = 2
            # print("there is cross")
        # print(math.sqrt((right_limit[0] - left_limit[0])**2 + (right_limit[1] - left_limit[1])**2))
        # print(left_limit, right_limit, angle, num_point)

        _midpoint = [1 / 2 * (left_limit[0] + right_limit[0]), 1 / 2 * (left_limit[1] + right_limit[1])]
        if binary_img[int(new_y(_midpoint[1]))][int(_midpoint[0])] == 255:  # 求得的预瞄点是白色，非法
            break_flag = 1
            flag_info_ = 3
            # print("lines are broken")

        if break_flag == 1:
            break

        # 更新下一次遍历的角度，起始点
        angle = math.acos((_midpoint[0] - x) / math.sqrt((_midpoint[0] - x) ** 2 + (_midpoint[1] - y) ** 2))
        x, y = int(_midpoint[0]), int(_midpoint[1])
        aim_point.append([x, y])
        num_point = num_point + 1

    return aim_point, flag_info_

#  计算平均曲率
def get_K(_aim_point, _num_ap):
    k_list = []
    if _num_ap > 13:
        '''
        for index in range(_num_ap - 6):
            k = circle(_aim_point[0 + index], _aim_point[3 + index], _aim_point[6 + index]).PJcurvatur()
            k_list.append(k)
        n = len(k_list)
        weight = []
        for j in range(n):
            wj = 2 / (n + 1) - j * 2 / ((n + 1) * n)
            weight.append(wj)
        _k_avg = np.dot(k_list, weight)
        '''
        mid = int(_num_ap/2)
        _k_avg = circle(_aim_point[0], _aim_point[mid], _aim_point[_num_ap-1]).PJcurvatur()
        # print(k_list)
        # print(weight)
        # print(k_avg)
    else:  # 点数过少，认为有弯道，该减速
        _k_avg = 3.0

    if abs(_k_avg) > 5:
        _k_avg = 5

    return abs(_k_avg)

''' 
def get_warpPerspective(apts):
    M = np.matrix([[-3.26100577e+00, -1.82335807e+00,  1.37277824e+03],
                   [0.00000000e+00, -4.46562242e+00,  1.20571805e+03],
                   [-2.83615880e-18, -5.88073647e-03,  1.00000000e+00]])
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
'''
'''
class camera:
    def __init__(self):
        self.N = 0
        self.camMat = []
        self.camDistortion = []

        self.cap = cv2.VideoCapture(1)
        # self.cap = cv2.VideoCapture('test1.mp4')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # self.out = cv2.VideoWriter('result_test.mp4', self.fourcc, 10.0, (640, 368))  # 倒数第二个参数是帧率，每秒钟的帧数。10: 慢， 20：快
        # self.out_warped = cv2.VideoWriter('result_warp.mp4', self.fourcc, 10.0, (640, 368))

    def __del__(self):
        self.cap.release()
        self.out.release()

    def calibrateAndTrans(self):
        cameraMatrix = np.array([[1.15777930e+03, 0, 6.67111054e+02], [0, 1.15282291e+03, 3.86128937e+02], [0, 0, 1]])
        cameraDistortion = np.array([[-0.24688775, -0.02373133, -0.00109842, 0.00035108, -0.00258571]])

        if cameraMatrix != []:
            self.camMat = cameraMatrix
            self.camDistortion = cameraDistortion
            print('CALIBRATION SUCCEEDED!')
        else:
            print('CALIBRATION FAILED!')
        return 0

    def spin(self):
        global HEIGHT
        ret, img = self.cap.read()
        # if j == 202:
        #     cv2.imwrite('./test_code/' + str(i) + '.jpg', img)
        if ret == True:
            t0 = time.time()
            ####################################### img pre_processing ########################################
            # blur = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
            # print("shape:", img.shape)  # 480*640
            homo_img = homomorphic_filter(img)
            binary_img = get_binary(homo_img)

            # debug
            # cv2.rectangle(binary_img, (window_start, 320), (window_end, 367), (0, 255, 0), 1)
            # cv2.imshow('binary', homo_img)
            # cv2.waitKey(0)

            start_point, lane_base_flag = get_start_point(binary_img)
            aim_point, flag_info = get_aim_point(binary_img, start_point, 200, math.pi / 2, 18)  # aim_point 坐标左下角为原点
            num_ap = len(aim_point)  # 预瞄点数量. 正常应为NUM_AP.
            # print(cross_flag)

            """
            ######################### project test ######################
            projected = np.ones_like(binary_img)
            projected = np.stack((projected,) * 3, axis=-1).astype(np.uint8)  # 二维转3维,三通道灰度图
            warped_ap = get_warpPerspective(aim_point)
            
            for j in range(len(aim_point)):
                # warped_ap[j][1] = new_y(warped_ap[j][1])
                cv2.circle(projected, warped_ap[j], 2, [0, 0, 255])
            self.out_warped.write(projected)
            """

            ################### after getting ap, velocity control with fuzzy control ############################
            k_avg = get_K(aim_point, num_ap)  # 曲率。用这个变量衡量赛道的弯曲程度。
            if num_ap > 0:
                target_speed = fuzzy_controller(k_avg, num_ap) / 4  # map to 0-40
                vl, vr, theta_ap, lambda_ = get_steer(aim_point, num_ap, target_speed)
            else:
                vl, vr = 0, 0

            w = 100 * (vr - vl)
            
            # set speed to hardware
            vl, vr = 20, 0
            for i in range(20): car.set_speed(vr, vl)

            t1 = time.time()
            # print('time:', t1 - t0)
            
            self.N +=1
            
            fps = 1/(t1-t0)
            print(self.N, target_speed, vl, vr, theta_ap, lambda_, fps)

            
            ######################### to restore and visualization ###############################
            show = 0
            if show == 1:
                string_info = ['normal', 'error angle', 'there is cross', 'lines are broken', 'on the cross']
                if not flag_info:
                    flag_info = 0
                if lane_base_flag == 2:
                    flag_info = 4
                log_info = string_info[flag_info]
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
                
                #cv2.imwrite('./debug_fig/' + str(self.N) + '.jpg', binary_3chanl)
                cv2.imshow('img', binary_3chanl)
            # """

            # cv2.imwrite('./test_imgs/result_' + str(i) + '.jpg', binary_3chanl)
            # self.out.write(binary_3chanl)

if __name__ == '__main__':
    cam = camera()
    cam.calibrateAndTrans()
    while True:
        cam.spin()
        k = cv2.waitKey(1)
        if k == ord('q'):  # 按q结束检测
            break

    print('end of file')
'''