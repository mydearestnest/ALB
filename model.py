import boundary
from film.filmsystem import FilmSystem
import numpy as np
from dynamic.dysystem import DySystem

# 纯动压轴承计算
def hydrodynamic_film_bearing(system):
    if isinstance(system, FilmSystem):
        system.creat_cal_dom()
        system.cal_h()
        system.cal_kf()
        system.boundary_setting_with_lagrange()
        system.cal_p_direct()
        system.plot_p_3d()
    else:
        print('错误！请将系统对象传入')



# 使用牛顿迭代并引入雷诺边界条件
def hydrostatic_with_newton_and_reynold(system, iter_number=200, error_set=1e-8):
    if isinstance(system, FilmSystem):
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
            system.ip_init()
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
        system.updata_node()
        system.plot_p_3d()
        system.p_result = system.p_result[0:system.freedoms_norank]
        system.cal_p_finshed = True
    else:
        print('错误！请传入系统对象')


# 动态特性计算模块
def dynamic_char(system):
    # 必须完成压力计算才可执行动态特性计算
    system.setting_orifce_position(system.orifiec_x, system.orifiec_y)  # 设置油源位置
    system.creat_cal_dom()
    system.cal_h()
    tanks = np.loadtxt('config/tank.txt', comments='#', encoding='UTF-8', delimiter=',')
    system.add_h_tanks(tanks)
    system.add_qds()
    for i in range(4):
        system.cal_f(case=i)
        system.boundarys_setting(p_set=0)
        system.cal_p_direct()
        system.updata_node()
        system.cal_KC(case=i)

