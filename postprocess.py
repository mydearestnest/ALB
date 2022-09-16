from film.filmsystem import FilmSystem
import numpy as np


# 计算轴承静特性，包括承载力及其相位角
def static_performance_cal(system):
    if isinstance(system, FilmSystem):
        return integral_surface(system)
    else:
        print('错误！请将系统传入该函数内')
        return 0


# 对系统的p的x，y分量进行积分
# 输入为系统，输出为x，y方向的承载力与相位角
def integral_surface(system):
    load_x = 0
    load_y = 0
    lx = system.lx
    lz = system.lz
    for i in range(len(system.elems)):
        elem = system.elems[i]
        p = np.array([elem.nodes[j].p for j in range(elem.rank ** 2)])
        x = np.array([elem.nodes[j].coords[0] for j in range(elem.rank ** 2)])
        x0 = np.sum(x) / 4
        avg_p = np.sum(p) / 4
        load_x += avg_p * lx * lz * np.sin(x0)
        load_y += -avg_p * lx * lz * np.cos(x0)
    force_angel = np.arctan2(load_x, load_y)
    return load_x, load_y, force_angel

