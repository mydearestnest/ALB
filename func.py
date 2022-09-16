import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt


# @jit(nopython=True)
# 直接计算非齐次项

# @jit(nopython=True)
def cal_h(nx, nz, e):
    x = np.linspace(0, 2 * np.pi, nx + 1)
    h = 1 + e * np.cos(x + np.pi)
    h = h.repeat(nz + 1)
    h = h.reshape(nx + 1, nz + 1).T
    return h


#  残差计算
def cal_error(freedoms, **kwargs):
    result = kwargs['result']
    old = kwargs['old']
    dp = result - old
    error = np.sum(np.abs(dp[0:freedoms])) / np.sum(np.abs(result[0:freedoms]))

    return error


# 计算无量纲流量因数
def load_input(filename='input.txt'):
    return np.loadtxt(filename, comments='#', encoding='UTF-8', delimiter=',')
    # comment为注释符号，默认为#
    # 这里注意使用该读取命令时指定编码encoding = ‘UTF-8’


# 读取txt文件输入参数后建立变量与变量名的字典
def init_input(filename=None):
    if filename is None:
        filename = ['config/input.txt', 'config/orifice_position.txt']
    input_data = load_input(filename[0])
    args_name = ['e', 'w', 'lx', 'lz', 'nx', 'nz', 'u', 'c', 'pr', 'rho', 'r', 'l', 'cd', 'a0', 'ps', 'g', 'k']
    init_args = {}
    for i, arg_name in enumerate(args_name):
        init_args[arg_name] = input_data[i]
    input_data = load_input(filename[1])
    args_name = ['orifice_x', 'orifice_y']
    for i, arg_name in enumerate(args_name):
        if len(input_data) != 0:
            init_args[arg_name] = input_data[i]
        else:
            init_args[arg_name] = None
    return init_args


def load_capacity_cal_sp(system):
    load_x = 0
    load_y = 0
    lx = system.lx
    lz = system.lz
    for i in range(len(system.elems)):
        elem = system.elems[i]
        p = [elem.nodes[j].p for j in range(elem.rank ** 2)]
        x0 = elem.nodes[0].coords[0]
        load_x += integrate.nquad(p_x_div, [[0, lx], [0, lz]], args=(lx, lz, p, x0))[0]
        load_y += integrate.nquad(p_y_div, [[0, lx], [0, lz]], args=(lx, lz, p, x0))[0]
    return load_x, load_y


# 定义单元的插值函数，只要是节点上的变量都可以插值
# 该插值使用矩形一阶单元
# x,z是在单元的相对位置，范围为（0，lx），（0，lz）
def element_interpolation(x, z, lx, lz, p):
    f0 = (1 - z / lz) * (1 - x / lx)
    f1 = (1 - z / lz) * x / lx
    f2 = (1 - x / lx) * z / lz
    f3 = x * z / lz / lx
    p = f0 * p[0] + f1 * p[1] + f2 * p[2] + f3 * p[3]
    return p


# 定义全局x方向压力分量计算函数
def p_x_div(x, z, lx, lz, p, x0):
    return -element_interpolation(x, z, lx, lz, p) * np.sin(2 * (x + x0))


# 定义全局y方向压力计算函数
def p_y_div(x, z, lx, lz, p, x0):
    return element_interpolation(x, z, lx, lz, p) * np.cos(2 * (x + x0))


# 表面画图
#  表面画图
def plot_3d(mesh, data):
    ax = plt.axes(projection="3d")
    X = mesh[0]
    Z = mesh[1]
    ax.plot_surface(X, Z, data,
                    rstride=1,  # rstride（row）指定行的跨度
                    cstride=1,  # cstride(column)指定列的跨度,
                    cmap="rainbow")
    plt.show()

# 节流孔位置输入
# 设置单元x积分边界
# @jit(nopython=True)
# def x_boundary(lx):
#     return [0, lx]


# 设置单元z积分边界
# @jit(nopython=True)
# def z_boundary(lz):
#     return [0, lz]


# 计算局部刚度,暂不使用
# @jit(nopython=True)
# def func_k(x, z, n, m, x0, z0, lr, lx, lz, e):
#     dxf1 = dzf1 = dxf2 = dzf2 = 0
#     if n == 0:
#         dxf1 = - (1 - z / lz) / lx
#         dzf1 = - (1 - x / lx) / lz
#     elif n == 1:
#         dxf1 = (1 - z / lz) / lx
#         dzf1 = -x / lz / lx
#     elif n == 2:
#         dxf1 = -z / lz / lx
#         dzf1 = (1 - x / lx) / lz
#     elif n == 3:
#         dxf1 = z / lz / lx
#         dzf1 = x / lz / lx
#
#     if m == 0:
#         dxf2 = - (1 - z / lz) / lx
#         dzf2 = - (1 - x / lx) / lz
#     elif m == 1:
#         dxf2 = (1 - z / lz) / lx
#         dzf2 = -x / lz / lx
#     elif m == 2:
#         dxf2 = -z / lz / lx
#         dzf2 = (1 - x / lx) / lz
#     elif m == 3:
#         dxf2 = z / lz / lx
#         dzf2 = x / lz / lx
#
#     h = 1 + e * np.cos(x + x0)
#     p1 = h ** 3 * (
#             lr ** 2 * dxf1 * dxf2 + dzf1 * dzf2)
#     return p1


# 设置非齐次项函数
# @jit(nopython=True)
# def func_f(x, z, n, x0, z0, lx, lz, e, vx):
#     f_shape = 0
#     if n == 0:
#         f_shape = (1 - z / lz) * (1 - x / lx)
#     elif n == 1:
#         f_shape = (1 - z / lz) * x / lx
#     elif n == 2:
#         f_shape = (1 - x / lx) * z / lz
#     elif n == 3:
#         f_shape = x * z / lz / lx
#     p1 = f_shape * e * np.sin(x + x0) * vx
#     return p1


# def nard_f(n, x0, z0, lx, lz, e, vx):
#     return intergrate.nquad(func_f, [x_boundary(lx), z_boundary(lz)],
#                             args=(n, x0, z0, lx, lz, e, vx))  # 将膜厚带入，进行局部非齐次项计算


# def nard_k(n, m, x0, z0, lr, lx, lz, e):
#     return intergrate.nquad(func_k, [x_boundary(lx), z_boundary(lz)],
#                             args=(n, m, x0, z0, lr, lx, lz, e))
