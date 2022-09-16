from film.filmsystem import FilmSystem
from dynamic.dyelement import DyElement
import numpy as np
from scipy import sparse as sp
from postprocess import integral_surface


class DySystem(FilmSystem):
    def __init__(self, dy_init_args):
        self.qds = None
        init_args = dy_init_args['init_args']
        super(DySystem, self).__init__(init_args)
        self.p_result = dy_init_args['p_result']
        self.h_result = dy_init_args['h_result']
        self.k_init = dy_init_args['k_init']
        # 定义空全局adxc矩阵
        self.adxc = sp.dok_matrix(((self.nx + 1) * (self.nz + 1), (self.nx + 1) * (self.nz + 1)), dtype=np.float32)
        # 定义空全局bdxc矩阵
        self.bdxc = np.zeros([(self.nx + 1) * (self.nz + 1)])

    def creat_rect_elems(self):
        for i in range(self.nz):
            for j in range(self.nx):
                nodes = []
                for m in range(2):
                    for n in range(2):
                        nodes.append(self.nodes[(i + m) * (self.nx + 1) + j + n])
                elem = DyElement(nodes, lx=self.lx, lz=self.lz, lr=self.lr, vx=self.vx)
                self.add_elem(elem)

    # 计算流量项对xc，yc，xct，yct的偏导数，输入为无量纲压差，输出为流量项偏导数计算结果
    def _cal_qd(self, dp):
        cq = 12 * self.u * self.lr * self.cd * self.a0 / self.c ** 3 * np.sqrt(2 / self.rho / self.ps)
        self.cq = cq
        qd = - cq / 2 / np.sqrt(dp)
        return qd

    # 添加流量对轴心位移与速度的偏导项
    def add_qds(self):
        for i, node_no in enumerate(self.orifiec_node_no):
            pk = self.p_result[node_no]
            dp = 1 - pk
            qd = self._cal_qd(dp)
            # 将流量项加入非齐次项
            self.k_add_boundary = sp.csc_matrix(self.k_add_boundary)
            self.k_add_boundary[node_no, node_no] -= qd  # 这里非齐次项为正，由推导获得
        return True

    # 计算刚度矩阵与源项并组装
    def cal_f(self):
        self._cal_emat()
        for el in self.elems:
            elem = self.elems[el]
            for n in range(4):
                tol_n = elem.nodes[n].number
                self.f[tol_n] += elem.bdxc[n]  # 组装进总体非齐次项向量
                for m in range(4):
                    tol_m = elem.nodes[m].number
                    self.adxc[tol_n, tol_m] += elem.adxc[n, m]  # 组装总刚度矩阵，找到

        temp = self.adxc.dot(self.p_result)
        self.f = self.f - temp

    # 计算所有单元的ad，bd
    def _cal_emat(self):
        for elem in self.elems.values():
            elem.cal_el_adxc()
            elem.cal_el_bdxc()

    def cal_Kxx_Kyx(self):
        res = integral_surface(self)
        Kxx = res[0]
        Kyx = res[1]
        self.Kxx = Kxx
        self.Kyx = Kyx
