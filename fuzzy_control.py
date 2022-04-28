import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import math
import numpy.linalg as la

d = 0.1  # car width/2, 代调
# k_steer = 100  # 方向调节增益, 代调
NUM_AP = 24


class circle:
    def __init__(self, pt1, pt2, pt3):  # 3个point。 2, 5, 8
        # self.R, self.sign, self.R_rate = self.getR(pt1, pt2, pt3, pt4, pt5, pt6)
        self.x = [pt1[0], pt2[0], pt3[0]]
        self.y = [pt1[1], pt2[1], pt3[1]]

    def constrain(self, min_, max_, output):
        if output < min_:
            output = min_
        elif output > max_:
            output = max_
        else:
            output = output
        return output

    def PJcurvatur(self):
        t_a = la.norm([self.x[1] - self.x[0], self.y[1] - self.y[0]])
        t_b = la.norm([self.x[2] - self.x[1], self.y[2] - self.y[1]])

        M = np.array([
            [1, -t_a, t_a ** 2],
            [1, 0, 0],
            [1, t_b, t_b ** 2]
        ])

        a = np.matmul(la.inv(M), self.x)
        b = np.matmul(la.inv(M), self.y)

        kappa = 1000 * 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2. + b[1] ** 2.) ** (1.5)  # 曲率

        return kappa


def fuzzy_controller(ke, num_pt):
    global NUM_AP
    # 定义变量范围
    x_ke = np.arange(0, 6, 1)  # [-5 -4 -3 -2 -1 0 1 2 3 4 5]
    x_num = np.arange(0, NUM_AP+1, 1)  # [0 1 2 .. 13]
    x_speed = np.arange(0, 100, 1)  # 速度输出
    # 定义模糊控制变量
    error = ctrl.Antecedent(x_ke, 'Error')
    num_of_point = ctrl.Antecedent(x_num, 'NOP')
    u_speed = ctrl.Consequent(x_speed, 'speed')
    
    # 生成模糊隶属函数
    error['Z'] = fuzz.trimf(x_ke, [0, 0, 5])
    error['M'] = fuzz.trimf(x_ke, [0, 2.5, 5])
    error['B'] = fuzz.trimf(x_ke, [0, 5, 5])
    num_of_point['S'] = fuzz.trimf(x_num, [0, 0, 11])
    num_of_point['M'] = fuzz.trimf(x_num, [10, 15, 20])
    num_of_point['B'] = fuzz.trimf(x_num, [18, 22, 24])
    u_speed['S'] = fuzz.trimf(x_speed, [0, 20, 40])
    u_speed['M'] = fuzz.trimf(x_speed, [20, 40, 60])
    u_speed['B'] = fuzz.trimf(x_speed, [40, 60, 100])
    # 规则
    rule1 = ctrl.Rule(
        antecedent=((error['Z'] & num_of_point['S']) |
                    (error['B'] & num_of_point['S']) |
                    (error['M'] & num_of_point['S']) |
                    (error['B'] & num_of_point['M']) |
                    (error['B'] & num_of_point['B'])
                    ),
        consequent=u_speed['S'], label='S')

    rule2 = ctrl.Rule(
        antecedent=((error['Z'] & num_of_point['M']) |
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

def get_steer(aim_point, num_pt, vc):
    global NUM_AP 
    start_x = aim_point[0][0]
    start_y = aim_point[0][1]
    
    mid_ap = 15
    if num_pt > mid_ap+1:
        target_x = (aim_point[mid_ap-2][0] + aim_point[mid_ap][0] + aim_point[mid_ap+2][0]) / 3
        target_y = (aim_point[mid_ap-2][1] + aim_point[mid_ap][1] + aim_point[mid_ap+2][1]) / 3
    else:  # theta is 0
        target_x = start_x
        target_y = start_y + 1

    theta = math.atan((target_x - start_x) / (target_y - start_y))
    l = math.sqrt((target_x - start_x) ** 2 + (target_y - start_y) ** 2)

    k_correction = 800  # 修正系数
    if abs(theta) < 0.1:
        vl = vc * 0.95
        vr = vc
        lambda_ = 0
    else:
        R = l / 2 / math.sin(theta)
        vl = vc + d/R*vc * k_correction
        vr = vc - d/R*vc * k_correction
        lambda_ = d/R*k_correction

    return vl, vr, theta, lambda_
