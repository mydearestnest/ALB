from film.filmsystem import FilmSystem
from dynamic.dyelement import DyElement
import numpy as np
from scipy import sparse as sp
from postprocess import integral_surface
from scipy.sparse import linalg as sl


class DySystem(FilmSystem):
    def __init__(self, dy_init_args):
        self.NKC = []
        self.KC = []
        self.qds = None
        init_args = dy_init_args['init_args']
        super(DySystem, self).__init__(init_args)
        self.p_ans = dy_init_args['p_result']
        self.h_result = dy_init_args['h_result']
        self.k_init = dy_init_args['k_init']
        # 定义空全局adxc矩阵
        self.ad = np.zeros(((self.nx + 1) * (self.nz + 1), (self.nx + 1) * (self.nz + 1)), dtype=np.float32)
        # 定义空全局bdxc矩阵
        self.bd = np.zeros([(self.nx + 1) * (self.nz + 1)])

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
        qd = cq / 2 / np.sqrt(dp)
        return qd

    # 添加流量对轴心位移与速度的偏导项
    def add_qds(self):
        for i, node_no in enumerate(self.orifiec_node_no):
            pk = self.p_ans[node_no]
            dp = 1 - pk
            qd = self._cal_qd(dp)
            # 将流量项加入非齐次项
            # self.k_init = sp.csc_matrix(self.k_add_boundary)
            self.k_init[node_no, node_no] += qd  # 这里非齐次项为正，由推导获得
        return True

    # 计算刚度矩阵与源项并组装
    def cal_f(self, case=0):
        self._cal_emat(case)
        self.f = np.zeros([(self.nx + 1) * (self.nz + 1)])  # 计算前重置非齐次项
        self.ad = np.zeros([self.freedoms_norank, self.freedoms_norank])
        for el in self.elems:
            elem = self.elems[el]
            for n in range(4):
                tol_n = elem.nodes[n].number
                self.f[tol_n] += elem.bd[n]  # 组装进总体非齐次项向量
                # 当计算刚度时启动以下两项
                if case == 0 or case == 1:
                    for m in range(4):
                        tol_m = elem.nodes[m].number
                        self.ad[tol_n, tol_m] += elem.ad[n, m]  # 组装总刚度矩阵，找到
        # 当计算刚度时启用以下两项
        if case == 0 or case == 1:
            temp = self.ad.dot(self.p_ans)
            self.ad = sp.csc_matrix(self.ad)
            self.f = self.f - temp

    # 计算所有单元的ad，bd
    def _cal_emat(self, case):
        if case == 0:
            for elem in self.elems.values():
                elem.cal_el_adxc()
                elem.cal_el_bdxc()
        elif case == 1:
            for elem in self.elems.values():
                elem.cal_el_adyc()
                elem.cal_el_bdyc()
        elif case == 2:
            for elem in self.elems.values():
                elem.cal_el_bdxct()
        elif case == 3:
            for elem in self.elems.values():
                elem.cal_el_bdyct()

    def cal_KC(self, case=0):
        res = integral_surface(self)
        self.NKC.append(-res[0])
        self.NKC.append(-res[1])
        ans1 = -res[0] * self.l * self.r / 2 * self.ps / self.c
        ans2 = -res[1] * self.l * self.r / 2 * self.ps / self.c
        if case == 2 or case == 3:
            t0 = 3 * self.u * self.l**2 / self.ps / self.c**2
            ans1 = ans1 * t0
            ans2 = ans2 * t0
        self.KC.append(ans1)
        self.KC.append(ans2)
