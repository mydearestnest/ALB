import boundary
from base import System
import numpy as np


# 纯动压轴承计算
def hydrodynamic_film_bearing(system):
    if isinstance(system, System):
        system.creat_cal_dom()
        system.cal_h()
        system.cal_kf()
        system.boundary_setting_with_lagrange()
        system.cal_p_direct()
        system.plot_p_3d()
    else:
        print('错误！请将系统对象传入')


# 小孔节流静压轴承计算，直接求解加比例分割法
def hydrostatic_film_bearing_with_orifice(system):
    if isinstance(system, System):
        system.creat_cal_dom()
        system.cal_h()
        system.cal_kf()
        system.setting_orifce_position(system.orifiec_x, system.orifiec_y)  # 设置油源位置
        for i in range(100):
            system.boundary_setting_with_lagrange()  # 组装拉格朗日乘子阵
            # system.add_fq()
            system.add_fqs()
            system.cal_p_direct()
            # 判断收敛条件
            # 至少需要算两次迭代
            if len(system.p_in_ans) >= 2:
                # 针对静压孔压力值计算是否收敛
                if abs((sum(system.p_in_ans[-1]) - sum(system.p_in_ans[-2])) / sum(system.p_in_ans[-1])) < 1e-8:
                    break
        system.plot_p_3d()
    else:
        print('错误！请将系统对象传入')


# 使用牛顿迭代并引入雷诺边界条件
def hydrostatic_with_newton_and_reynold(system):
    if isinstance(system, System):
        system.setting_orifce_position(system.orifiec_x, system.orifiec_y)  # 设置油源位置
        system.creat_cal_dom()
        system.cal_h()
        system.cal_kf()
        system.boundary_setting_with_lagrange()
        system.cal_p_direct()
        for i in range(200):
            system.boundarys_setting()  # 组装拉格朗日乘子阵
            system.add_fqs()
            system.cal_p_iter_newton()
            boundary.reynold_boundary(system)
            p_in_ans = np.array(system.p_in_ans)  # 原本为list，转为ndarray
            if len(p_in_ans) < 2:
                error = 1
            else:
                error = np.sum(np.abs(p_in_ans[-1] - p_in_ans[-2])) / np.sum(p_in_ans)
            print(i, ',', error)
            if error < 1e-4:
                break
        system.plot_p_3d()
    else:
        print('错误！请传入系统对象')
