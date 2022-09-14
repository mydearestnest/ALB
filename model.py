import boundary
from system import System
import numpy as np
from scipy import sparse as sp


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
def hydrostatic_film_bearing_with_orifice(system, iter_number=200, error_set=1e-8):
    if isinstance(system, System):
        system.creat_cal_dom()
        system.cal_h()
        system.cal_kf()
        system.setting_orifce_position(system.orifiec_x, system.orifiec_y)  # 设置油源位置
        for i in range(iter_number):
            system.boundary_setting_with_lagrange()  # 组装拉格朗日乘子阵
            # system.add_fq()
            system.add_fqs(method='direct', nq=[1], node_nos=[5000])
            system.cal_p_direct()
            # 判断收敛条件
            # 至少需要算两次迭代
            if len(system.p_in_ans) >= 2:
                # 针对静压孔压力值计算是否收敛
                error = system.cal_error()
                print(i, ',', error)
                if error < error_set:
                    break
        system.plot_p_3d()
    else:
        print('错误！请将系统对象传入')


# 使用牛顿迭代并引入雷诺边界条件
def hydrostatic_with_newton_and_reynold(system, iter_number=200, error_set=1e-8):
    if isinstance(system, System):
        system.setting_orifce_position(system.orifiec_x, system.orifiec_y)  # 设置油源位置
        system.creat_cal_dom()
        system.cal_h()
        tanks = np.loadtxt('config/tank.txt', comments='#', encoding='UTF-8', delimiter=',')
        system.add_h_tanks(tanks)
        system.cal_kf()
        # system.boundary_setting_with_lagrange()
        # system.cal_p_direct()
        for i in range(iter_number):
            system.boundarys_setting()  # 组装拉格朗日乘子阵
            system.add_fqs()
            system.cal_p_iter_newton()
            boundary.reynold_boundary(system)
            if i < 1:
                error = 1
            else:
                error = system.cal_error()
            print(i, ',', error)
            if error < error_set:
                break
        system.plot_p_3d()
        system.p_result = system.p_result[0:system.freedoms_norank]
        system.cal_p_finshed = True
    else:
        print('错误！请传入系统对象')


# 动态特性计算模块
def dynamic_char_with_orifice(system, iter_number=200, error_set=1e-8):
    # 必须完成压力计算才可执行动态特性计算
    if system.cal_p_finshed is True:
        # 建立adxc稀疏空矩阵
        system.adxc = sp.dok_matrix((system.freedoms_norank,system.freedoms_norank), dtype=np.float32)

        pass
    else:
        print('请至少一次计算压力场')
