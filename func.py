import numpy as np
import scipy.integrate as intergrate
from numba import jit, vectorize, float64


# 设置单元x积分边界
@jit(nopython=True)
def x_boundary(lx):
    return [0, lx]


# 设置单元z积分边界
@jit(nopython=True)
def z_boundary(lz):
    return [0, lz]


# 计算局部刚度
@jit(nopython=True)
def func_k(x, z, n, m, x0, z0, lr, lx, lz, e):
    dxf1 = dzf1 = dxf2 = dzf2 = 0
    if n == 0:
        dxf1 = - (1 - z / lz) / lx
        dzf1 = - (1 - x / lx) / lz
    elif n == 1:
        dxf1 = (1 - z / lz) / lx
        dzf1 = -x / lz / lx
    elif n == 2:
        dxf1 = -z / lz / lx
        dzf1 = (1 - x / lx) / lz
    elif n == 3:
        dxf1 = z / lz / lx
        dzf1 = x / lz / lx

    if m == 0:
        dxf2 = - (1 - z / lz) / lx
        dzf2 = - (1 - x / lx) / lz
    elif m == 1:
        dxf2 = (1 - z / lz) / lx
        dzf2 = -x / lz / lx
    elif m == 2:
        dxf2 = -z / lz / lx
        dzf2 = (1 - x / lx) / lz
    elif m == 3:
        dxf2 = z / lz / lx
        dzf2 = x / lz / lx

    h = 1 + e * np.cos(x + x0)
    p1 = h ** 3 * (
            lr ** 2 * dxf1 * dxf2 + dzf1 * dzf2)
    return p1


# 设置非齐次项函数
@jit(nopython=True)
def func_f(x, z, n, x0, z0, lx, lz, e, vx):
    f_shape = 0
    if n == 0:
        f_shape = (1 - z / lz) * (1 - x / lx)
    elif n == 1:
        f_shape = (1 - z / lz) * x / lx
    elif n == 2:
        f_shape = (1 - x / lx) * z / lz
    elif n == 3:
        f_shape = x * z / lz / lx
    p1 = f_shape * e * np.sin(x + x0) * vx
    return p1


def nard_f(n, x0, z0, lx, lz, e, vx):
    return intergrate.nquad(func_f, [x_boundary(lx), z_boundary(lz)],
                            args=(n, x0, z0, lx, lz, e, vx))  # 将膜厚带入，进行局部非齐次项计算


# def nard_k(n, m, x0, z0, lr, lx, lz, e):
#     return intergrate.nquad(func_k, [x_boundary(lx), z_boundary(lz)],
#                             args=(n, m, x0, z0, lr, lx, lz, e))


def direct_fe(lx, lz, vx, h):
    h0 = h[0]
    h1 = h[1]
    h2 = h[2]
    h3 = h[3]
    fe = np.array([(h2 * lx * vx * (h0 + 2 * h1 + 3 * h2 + 2 * h3)) / 72 - (h2 ** 2 * lx ** 2 * vx) / (72 * lz) - (
                lz * vx * (
                    3 * h0 * h1 - 3 * h0 * h2 + h0 * h3 + 3 * h1 * h2 + 2 * h1 * h3 + 3 * h2 * h3 - 6 * h0 ** 2 + 3 * h1 ** 2 + h3 ** 2)) / 72,
                   (h2 * lx * vx * (4 * h1 - h0 + 3 * h2 + 4 * h3)) / 72 - (lz * vx * (
                               3 * h1 * h2 - 3 * h0 * h2 - h0 * h3 - 3 * h0 * h1 + 4 * h1 * h3 + 3 * h2 * h3 - 3 * h0 ** 2 + 6 * h1 ** 2 + 2 * h3 ** 2)) / 72 - (
                               h2 ** 2 * lx ** 2 * vx) / (36 * lz), -(vx * (
                    3 * h1 ** 2 * lz ** 3 - 3 * h0 ** 2 * lz ** 3 - 6 * h2 ** 2 * lx ** 3 + 9 * h3 ** 2 * lz ** 3 + h0 ** 2 * lx * lz ** 2 - 2 * h1 ** 2 * lx * lz ** 2 - 18 * h2 ** 2 * lx * lz ** 2 + 18 * h2 ** 2 * lx ** 2 * lz - 6 * h3 ** 2 * lx * lz ** 2 - 6 * h0 * h2 * lz ** 3 + 6 * h1 * h2 * lz ** 3 + 6 * h1 * h3 * lz ** 3 + 18 * h2 * h3 * lz ** 3 + h0 * h1 * lx * lz ** 2 + 3 * h0 * h2 * lx * lz ** 2 - h0 * h2 * lx ** 2 * lz + h0 * h3 * lx * lz ** 2 - 9 * h1 * h2 * lx * lz ** 2 + 4 * h1 * h2 * lx ** 2 * lz - 4 * h1 * h3 * lx * lz ** 2 - 27 * h2 * h3 * lx * lz ** 2 + 12 * h2 * h3 * lx ** 2 * lz)) / (
                               72 * lz ** 2), (h2 * lx * vx * (4 * h1 - h0 + 9 * h2 + 12 * h3)) / 72 - (lz * vx * (
                    3 * h1 * h2 - 3 * h0 * h2 - h0 * h3 - h0 * h1 + 4 * h1 * h3 + 9 * h2 * h3 - h0 ** 2 + 2 * h1 ** 2 + 6 * h3 ** 2)) / 72 - (
                               h2 ** 2 * lx ** 2 * vx) / (12 * lz), ])
    return fe


def direct_ke(lx, lz, lr, h):
    h0 = h[0]
    h1 = h[1]
    h2 = h[2]
    h3 = h[3]
    ke = np.array()
    return ke


@jit(nopython=True)
def cal_h(nx, nz, e):
    x = np.linspace(0, 2 * np.pi, nx + 1)
    h = 1 + e * np.cos(x)
    h = h.repeat(nz + 1)
    h = h.reshape(nx + 1, nz + 1).T
    return h
