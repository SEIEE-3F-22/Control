import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import math
import numpy.linalg as la

d = 0.1  # car width/2, 代调
# k_steer = 100  # 方向调节增益, 代调
NUM_AP = 19


def fuzzy_controller(ke, num_pt):
    global NUM_AP
    # 定义变量范围
    x_ke = np.arange(0, 1.6, 0.1)  # [-5 -4 -3 -2 -1 0 1 2 3 4 5]
    x_num = np.arange(0, NUM_AP + 1, 1)  # [0 1 2 .. 13]
    x_speed = np.arange(0, 100, 1)  # 速度输出
    # 定义模糊控制变量
    error = ctrl.Antecedent(x_ke, 'Error')
    num_of_point = ctrl.Antecedent(x_num, 'NOP')
    u_speed = ctrl.Consequent(x_speed, 'speed')

    # 生成模糊隶属函数
    error['Z'] = fuzz.trimf(x_ke, [0, 0, 1.5])
    error['M'] = fuzz.trimf(x_ke, [0, 0.7, 1.5])
    error['B'] = fuzz.trimf(x_ke, [0, 1.5, 1.5])
    num_of_point['S'] = fuzz.trimf(x_num, [0, 0, 9])
    num_of_point['M'] = fuzz.trimf(x_num, [6, 11, 16])
    num_of_point['B'] = fuzz.trimf(x_num, [12, 15, NUM_AP])
    u_speed['S'] = fuzz.trimf(x_speed, [0, 25, 50])
    u_speed['M'] = fuzz.trimf(x_speed, [30, 50, 70])
    u_speed['B'] = fuzz.trimf(x_speed, [50, 80, 100])
    # 规则
    rule1 = ctrl.Rule(
        antecedent=((error['B'] & num_of_point['S']) |
                    (error['M'] & num_of_point['S']) |
                    (error['B'] & num_of_point['M']) |
                    (error['B'] & num_of_point['B'])
                    ),
        consequent=u_speed['S'], label='S')

    rule2 = ctrl.Rule(
        antecedent=((error['Z'] & num_of_point['S']) |
                    (error['Z'] & num_of_point['M']) |
                    (error['M'] & num_of_point['M']) |
                    (error['M'] & num_of_point['B'])
                    ),
        consequent=u_speed['M'], label='M')

    rule3 = ctrl.Rule(
        antecedent=((error['Z'] & num_of_point['B'])
        ),
        consequent=u_speed['B'], label='B')

    speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    output_speed = ctrl.ControlSystemSimulation(speed_ctrl)
    output_speed.input['Error'] = ke
    output_speed.input['NOP'] = num_pt
    output_speed.compute()

    return output_speed.output['speed']


def get_steer(aim_point, num_pt, vc, k, k_var):
    global NUM_AP
    lane_start_x = aim_point[0][0]
    lane_start_y = aim_point[0][1]
    start_x = 324  # 对应的道路正中心点
    start_y = 0

    mid_ap = 2
    if num_pt > mid_ap + 1:
        target_x = (aim_point[mid_ap - 1][0] + aim_point[mid_ap][0] + aim_point[mid_ap + 1][0]) / 3
        target_y = (aim_point[mid_ap - 1][1] + aim_point[mid_ap][1] + aim_point[mid_ap + 1][1]) / 3
    else:  # theta is 0, coming to a cross
        target_x = lane_start_x + 1
        target_y = lane_start_y + 30

    theta = math.atan((target_x - lane_start_x) / (target_y - lane_start_y))
    l = math.sqrt((target_x - lane_start_x) ** 2 + (target_y - lane_start_y) ** 2)

    # 直道->弯道的过渡
    if (abs(k) < 0.7 and abs(theta) < 0.5 and k_var > 0.5) or theta == 0:
        vl = vc
        vr = vc * 1.03
        lambda_ = 0
        flag_case = 1

    # 直道
    elif (abs(k) < 0.12 and abs(theta) < 0.12) or num_pt < 4:
        vl = vc
        vr = vc * 1.03
        lambda_ = 0
        flag_case = 4

    # 弯道->直道的过渡
    elif abs(k) < 0.9 and abs(theta) < 0.6 and k_var < 0.5:
        if start_x < lane_start_x - 10:  # 如果车道线基线靠右，则减速直走直到基线在左侧
            deccelerate = 1
            vl = vc * deccelerate
            vr = vc * deccelerate
            lambda_ = 0
        else:  # 车道线基线靠左，使用远预瞄点
            apx, apy = aim_point[num_pt - 1][0], aim_point[num_pt - 1][1]
            theta_far = math.atan((apx - start_x) / (apy - start_y))
            k_correction = 60
            R = l / 2 / math.sin(theta_far)
            vl = vc + d / R * vc * k_correction
            vr = vc - d / R * vc * k_correction
            lambda_ = d / R * k_correction
        flag_case = 3

        # print("straight correction")

    # 过弯
    else:
        k_correction = 115
        R = l / 2 / math.sin(theta)
        vl = vc + d / R * vc * k_correction
        vr = vc - d / R * vc * k_correction
        if vr < 0:
            vr = 0
        lambda_ = d / R * k_correction
        flag_case = 2

    # print("flag_case:", flag_case)

    return vl, vr, theta, lambda_, flag_case


def get_cross_steer(aim_point, num_pt, vc):
    apx, apy = aim_point[num_pt - 1][0], aim_point[num_pt - 1][1]
    start_x, start_y = aim_point[0][0], aim_point[0][1]
    # print(apx, apy, start_x, start_y)
    theta_far = math.atan((apx - start_x) / (apy - start_y))
    k_correction = 60
    l = 30
    lambda_ = d / l * 2 * math.sin(theta_far) * k_correction
    vl = vc + lambda_ * vc
    vr = vc - lambda_ * vc

    return vl, vr, lambda_, theta_far
