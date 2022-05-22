import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse as sp
from scipy.sparse import linalg as sl
from func import direct_fe, direct_ke

'''定义全局节点,全局节点属性包括全局节点编号与全局节点坐标'''


class Node:
    def __init__(self, coords):
        self._number = None  # 节点编号
        self._coords = coords  # 节点坐标
        self._dim = 2  # 节点维度
        self._h = 0  # 节点厚度
        self._p = 1  # 节点压力

    # 调用节点编号
    @property
    def number(self):
        return self._number

    # 修改节点编号
    @number.setter
    def number(self, val):
        if isinstance(val, int):
            self._number = val
        else:
            pass

    # 调用节点坐标
    @property
    def coords(self):
        return self._coords

    # 修改节点坐标
    @coords.setter
    def coords(self, val):
        if isinstance(val, np.ndarray):
            self._coords = val
        else:
            print('error node coords set')
            pass

    # 调用节点维度
    @property
    def dim(self):
        return self._dim

    # 修改节点维度
    @dim.setter
    def dim(self, val):
        pass

    # 调用节点压力
    @property
    def p(self):
        return self._p

    # 修改节点压力
    @p.setter
    def p(self, val):
        if isinstance(val, float):
            self._p = val
        else:
            pass

    # 调用节点厚度
    @property
    def h(self):
        return self._h

    # 修改节点厚度
    @h.setter
    def h(self, val):
        if isinstance(val, float):
            self._h = val
        else:
            pass


'''定义单元，其中包括内部节点与单元编号'''


class Elem:
    def __init__(self, nodes):
        self.avg_p = None
        self._nodes = nodes  # 定义单元内节点
        self._number = None  # 定义单元编号
        self._rank = 2  # 定义单元形函数阶数
        self._ke = np.zeros([self.rank ** 2, self.rank ** 2])  # 定义单元局部矩阵
        self._fe = np.zeros([self.rank ** 2])

    # 调用单元内节点
    @property
    def nodes(self):
        return self._nodes

    # 调用单元内行函数阶数
    @property
    def rank(self):
        return self._rank

    # 调用单元编号
    @property
    def number(self):
        return self._number

    # 修改单元编号
    @number.setter
    def number(self, val):
        if isinstance(val, int):
            self._number = val
        else:
            print('error elem number set')
            pass

    # 修改行函数阶数
    @rank.setter
    def rank(self, val):
        pass

    # 调用单元刚度矩阵
    @property
    def ke(self):
        return self._ke

        # 修改单元刚度矩阵

    @ke.setter
    def ke(self, val):
        if isinstance(val, np.ndarray):
            self._ke = val
        else:
            print('error elem ke set')
            pass

        # 调用单元刚度矩阵

    @property
    def fe(self):
        return self._fe

        # 修改单元刚度矩阵

    @fe.setter
    def fe(self, val):
        if isinstance(val, np.ndarray):
            self._fe = val
        else:
            print('error elem fe set')
            pass

    def avg_p(self):
        avg_p = 0
        for n in range(4):
            avg_p += self.nodes[n].p
        self.avg_p = avg_p / 4
        return self.avg_p


# 定义求解系统

class System:
    def __init__(self, init_args):
        self.nodes = {}  # 定义网格上所有节点
        self.elems = {}  # 定义网格上所有单元
        self._init_input_args(init_args)  # 初始化输入参数
        self._init_system_args()  # 初始化系统参数
        self._mesh_init()  # 网格坐标初始化

    def _init_input_args(self, init_args):
        self.nx = np.int_(init_args['nx'])  # 定义单元数量（Nx，Ny）
        self.nz = np.int_(init_args['nz'])
        self.lx = init_args['lx'] / self.nx  # 定义单元长度
        self.lz = init_args['lz'] / self.nz
        self.vx = init_args['vx']
        self.lr = init_args['lr']
        self.e = init_args['e']
        self.u = init_args['u']
        self.l = init_args['l']
        self.r = init_args['r']
        self.rho = init_args['rho']
        self.c = init_args['c']
        self.pr = init_args['pr']
        self.a0 = init_args['a0']
        self.cd = init_args['cd']
        self.ps = init_args['ps']
        self.orifiec_x = init_args['orifice_x']
        self.orifiec_y = init_args['orifice_y']

    def _init_system_args(self):
        # self.k = np.zeros([(nx + 1) * (nz + 1), (nx + 1) * (nz + 1)])  # 定义刚度
        self.k = sp.dok_matrix(((self.nx + 1) * (self.nz + 1), (self.nx + 1) * (self.nz + 1)), dtype=np.float32)
        # 使用dok稀疏矩阵内存占用更少，但计算速度降低，主要时间用于组装
        self.f = np.zeros([(self.nx + 1) * (self.nz + 1)])  # 定义非齐次项
        self.nq = []  # 无量纲流量因数
        self.q = []  # 节流孔流量
        self.orifiec_node_no = []  # 节流孔节点位置
        # 输入初始化结束
        # 压力场初始化
        self.p = np.ones_like(self.f)
        # 液膜厚度初始化
        self.h = None
        # 用于储存补充拉格朗日乘子矩阵
        self.kc = None
        self.fc = None
        # 用于储存节流孔入口压力计算结果
        self.p_in_ans = []
        # 用于储存节流孔入口压力初值
        self.p_in_init = []
        # 当压差小于零0时所需要的压力值
        self.pf = self.pr

    # 初始化网格，用于画图
    def _mesh_init(self):
        x = np.linspace(0, self.lx * self.nx, self.nx + 1)
        z = np.linspace(0, self.lz * self.nz, self.nz + 1)
        self.X, self.Z = np.meshgrid(x, z)

    # 增加单个节点
    def add_node(self, node):
        if isinstance(node, Node):
            if node.number is None:
                node.number = self.non
            self.nodes[node.number] = node
        else:
            return print('请添加node类')

    # 增加节点集
    def add_nodes(self, *nodes):
        if isinstance(nodes, list) or isinstance(nodes, tuple) or isinstance(nodes, np.ndarray):
            for node in nodes:
                self.add_node(node)
        else:
            self.add_node(nodes)

    # 增加单个单元
    def add_elem(self, elem):
        if isinstance(elem, Elem):
            if elem.number is None:
                elem.number = self.noe
            self.elems[elem.number] = elem
        else:
            return print('请添加elem类')

    # 增加单元集
    def add_elems(self, *elems):
        if isinstance(elems, list) or isinstance(elems, tuple) or isinstance(elems, np.ndarray):
            for elem in elems:
                self.add_elem(elem)
        else:
            self.add_elem(elems)

    # 返回当前系统储存节点数，作为下一个节点编号
    @property
    def non(self):
        return len(self.nodes)

    # 返回当前系统储存单元数，作为下一个单元编号
    @property
    def noe(self):
        return len(self.elems)

    # 小孔节流静压轴承计算
    def hydrostatic_film_bearing_with_orifice(self):
        self.creat_cal_dom()
        self.cal_h()
        self.cal_kf()
        self.setting_orifce_position(self.orifiec_x, self.orifiec_y)  # 设置油源位置
        for i in range(100):
            self.set_bondary()        # 组装拉格朗日乘子阵
            # self.add_fq()
            self.add_fqs()
            self.cal_p_direct()
            if len(self.p_in_ans) >= 2:
                if abs((sum(self.p_in_ans[-1]) - sum(self.p_in_ans[-2])) / sum(self.p_in_ans[-1])) < 1e-8:
                    break
        self.plot_p_3d()

    # 纯动压轴承计算
    def hydrodynamic_film_bearing(self):
        self.creat_cal_dom()
        self.cal_h()
        self.cal_kf()
        self.set_bondary()
        self.cal_p_direct()

    # 创建计算域
    def creat_cal_dom(self):
        self.creat_rect_nodes()
        self.crear_rect_elems()

    # 创建矩形单元节点
    def creat_rect_nodes(self):
        for i in range(self.nz + 1):
            for j in range(self.nx + 1):
                node = Node(np.array([j * self.lx, i * self.lz]))
                self.add_node(node)

    # 创建矩形单元
    def crear_rect_elems(self):
        for i in range(self.nz):
            for j in range(self.nx):
                nodes = []
                for m in range(2):
                    for n in range(2):
                        nodes.append(self.nodes[(i + m) * (self.nx + 1) + j + n])
                elem = Elem(nodes)
                self.add_elem(elem)

    # 计算刚度阵与无量纲轴承数项并组装
    def cal_kf(self):
        for el in self.elems:
            elem = self.elems[el]
            h0 = elem.nodes[0].h
            h1 = elem.nodes[1].h
            h2 = elem.nodes[2].h
            h3 = elem.nodes[3].h
            elem.ke = direct_ke(self.lx, self.lz, self.lr, h0, h1, h2, h3)
            elem.fe = direct_fe(self.lx, self.lz, self.vx, h0, h1, h2, h3)
            for n in range(4):
                tol_n = elem.nodes[n].number
                self.f[tol_n] += elem.fe[n]  # 组装进总体非齐次项向量
                for m in range(4):
                    tol_m = elem.nodes[m].number
                    self.k[tol_n, tol_m] += elem.ke[n, m]  # 组装总刚度矩阵，找到

    def set_bondary(self):
        nx = self.nx
        nz = self.nz
        #  边界1：p（0：NX，0）= 1
        q_nodes = (nx + 1) * (nz + 1)
        l_nodes = nx + 1
        r_nodes = nz - 1
        kp1 = sp.dok_matrix((l_nodes, q_nodes), dtype=np.float32)
        fp1 = np.zeros(l_nodes)
        for i in range(l_nodes):
            kp1[i, i] = 1
            fp1[i] = 1
        #  边界2：p（0：NX，NY）= 1
        kp2 = sp.dok_matrix((l_nodes, q_nodes), dtype=np.float32)
        fp2 = np.zeros(l_nodes)
        for i in range(l_nodes):
            kp2[i, (nx + 1) * nz + i] = 1
            fp2[i] = 1
        #  边界3：周期边界条件：p（0，0：NY）= p(NX,0:NY)
        kp3 = sp.dok_matrix((r_nodes, q_nodes), dtype=np.float32)
        fp3 = np.zeros(r_nodes)
        for i in range(r_nodes):
            kp3[i, (i + 1) * (nx + 1)] = 1  # 附加矩阵内，左侧边界置1
            kp3[i, (i + 1) * (nx + 1) + nx] = -1  # 附加矩阵内，右侧边界置-1

        kp = sp.vstack((kp1, kp2, kp3))  # 边界条件刚度阵组合
        fp = np.hstack((fp1, fp2, fp3))  # 附加矩阵非齐次项组合
        self.kc = sp.vstack((self.k, kp))  # 补充拉格朗日乘子刚度矩阵——下方

        zeros_mat = sp.coo_matrix(((2 * l_nodes + r_nodes), (2 * l_nodes + r_nodes)))  # 补充刚度零阵
        kp_t = sp.vstack((kp.T, zeros_mat))
        self.kc = sp.hstack((self.kc, kp_t))  # 补充拉格朗日乘子刚度矩阵——右方
        self.fc = np.hstack((self.f, fp))  # 补充附加矩阵非齐次项

    # 计算厚度场
    def cal_h(self):
        nx = self.nx
        nz = self.nz
        hz = np.empty([self.nx + 1, self.nz + 1])
        for i in range(nx + 1):
            for j in range(nz + 1):
                hz[i][j] = self.nodes[i + j * (nx + 1)].h = 1 + self.e * np.cos(i * self.lx)

    # 计算压力场
    def cal_p_direct(self):
        # self.p = np.linalg.solve(self.k, self.f)
        # self.p = self.p[0:(self.nx + 1) * (self.nz + 1)].reshape(self.nz + 1, self.nx + 1)
        sp_p = sp.csc_matrix(self.kc)
        self.p = sl.spsolve(sp_p, self.fc)
        # 将计算后压力重新赋给各节点
        for i in range(len(self.nodes)):
            self.nodes[i].p = self.p[i]
        print('calculation of pressure is over')
        p_in_ans = []
        for node_no in self.orifiec_node_no:
            p_in_ans.append(self.p[node_no])  # 将节流孔的计算压力值记录储存
        self.p_in_ans.append(p_in_ans)

    def cal_p_iter_newton(self):
        # 将流量项对入口压力的Jacobi矩阵与刚度阵相加，不过由于Jacobi矩阵项较少，因此各项单独加入刚度矩阵
        # 构成dp前的迭代矩阵
        kp = sp.csc_matrix(self.kc)  # 迭代矩阵初始化sp_p = sp.csc_matrix(self.kc)
        for i, node_no in enumerate(self.orifiec_node_no):
            q_dp = 12 * self.u * self.l ** 2 * self.cd * self.a0 / (self.r * self.l * self.pr * self.c ** 3) * np.sqrt(
                0.5 / self.rho / (self.ps - self.p[node_no]))
            kp[node_no, node_no] += q_dp
        self.add_fqs()
        fp = self.fc - self.kc @ self.p
        dp = sl.spsolve(kp, fp)
        self.p = self.p + dp
        return dp

    def setting_orifce_position(self, orifice_x, orifice_z):
        if len(orifice_x) != len(orifice_z):
            print('error!节流孔x方向坐标数量必须等于y方向坐标')
        else:
            for i in range(len(orifice_x)):
                orifice_No = int(orifice_z[i] * self.nz) * (self.nx + 1) + int(orifice_x[i] * self.nx)
                self.orifiec_node_no.append(orifice_No)

    # 添加节流孔
    # def add_orifices(self , position):
    #     for
    # 计算无量纲流量因子,添加静压源项
    def add_fq(self):
        middle_No = int(0.5 * self.nz) * (self.nx + 1) + int(0.5 * self.nx)  # 获得节流孔结点位置（中心）
        node = self.nodes[middle_No]  # 找到节流孔对应结点
        # 第一次计算没有两个压力值，无法使用比例分割法迭代
        if len(self.p_in_init) == 0:
            p_init = node.p
        # 使用比例分割法，使下一次迭代的节流孔压力初始值为如下所示公式
        else:
            p_init = 1 / 10000 * (self.p_in_ans[-1] - self.p_in_init[-1]) + self.p_in_init[-1]
        dp = self.ps - p_init * self.pr  # 计算压力差
        # if dp < 0:
        #     p_init = 1 / 200 * (self.p_in_ans[-1] - self.p_in_init[-1]) + self.p_in_init[-1]
        #     dp = self.ps - p_init * self.pr  # 计算压力差
        self.p_in_init.append(p_init)  # 添加节流孔压力初始值
        a0 = self.a0  # 节流孔面积
        self.q = q = self.cd * a0 * np.sqrt(2 * dp / self.rho)  # 节流孔流量
        print(q)
        self.nq = fq = 12 * self.u * self.l ** 2 * q / (self.pr * self.c ** 3 * self.r * self.l)  # 无量纲流量因数计算
        # 添加流量的非齐次项
        self.fc[middle_No] += fq  # 这里非齐次项为负，由推导获得

    #  计算无量纲流量因子,添加静压源项
    def add_fqs(self):
        qs = []
        nqs = []
        p_in_init = []
        for i, node_no in enumerate(self.orifiec_node_no):
            node = self.nodes[node_no]  # 找到节流孔对应结点
            # 第一次计算没有两个压力值，无法使用比例分割法迭代
            if len(self.p_in_init) == 0:
                p_init = node.p
            # 使用比例分割法，使下一次迭代的节流孔压力初始值为如下所示公式
            else:
                p_init = 1 / 200 * (self.p_in_ans[-1][i] - self.p_in_init[-1][i]) + self.p_in_init[-1][i]
            dp = self.ps - p_init * self.pr  # 计算压力差
            # if dp < 0:
            #     p_init = 1 / 200 * (self.p_in_ans[-1] - self.p_in_init[-1]) + self.p_in_init[-1]
            #     dp = self.ps - p_init * self.pr  # 计算压力差
            p_in_init.append(p_init)  # 添加节流孔压力初始值
            a0 = self.a0  # 节流孔面积
            q = self.cd * a0 * np.sqrt(2 * dp / self.rho)  # 节流孔流量
            qs.append(q)
            fq = 12 * self.u * self.l ** 2 * q / (self.pr * self.c ** 3 * self.r * self.l)  # 无量纲流量因数计算
            nqs.append(fq)
            # 添加流量的非齐次项
            self.fc[node_no] += fq  # 这里非齐次项为正，由推导获得
        self.p_in_init.append(p_in_init)
        self.q.append(qs)
        self.nq.append(nqs)
        return nqs



    #  三维表面画图
    def plot_p_3d(self):
        ax = plt.axes(projection="3d")
        p = self.p[0:(self.nx + 1) * (self.nz + 1)].reshape(self.nz + 1, self.nx + 1)
        ax.plot_surface(self.X, self.Z, p,
                        rstride=1,  # rstride（row）指定行的跨度
                        cstride=1,  # cstride(column)指定列的跨度,
                        cmap="rainbow")
        plt.show()

    # 存储压力数据
    def save_output_p(self, filename='output_p.dat'):
        out_put_p = self.p[0:(self.nx + 1) * (self.nz + 1)].reshape(self.nz + 1, self.nx + 1)  # 将压力数据存为dat
        np.savetxt(filename, out_put_p)
# test_node = Node(np.array([1, 2]))
# test_system = System(20, 20, 1, 1)
# test_system.add_node(test_node)
# test_element = Elem(test_node)
# test_node1 = Node(np.array([1, 2]))
# test_system.add_nodes(test_node, test_node1)
# find_node = test_system.nodes[1]
# test_element1 = Elem(test_node1)
# test_system.add_elem(test_element)
# test_system.add_elems(test_element, test_element1)
# test_node3 = Node(np.array([1, 2]))
# print(test_node3.coords)
# test_node3.coords = np.array([3.4])
# def set_bondary(self):
#     nx = self.nx
#     nz = self.nz
#     #  对称边界条件设置
#
#     #  边界1：p（0：NX，0）= 1
#     q_nodes = (nx + 1) * (nz + 1)
#     l_nodes = nx + 1
#     r_nodes = nz - 1
#     kp1 = np.zeros((l_nodes, q_nodes))
#     fp1 = np.zeros(l_nodes)
#     for i in range(l_nodes):
#         kp1[i][i] = 1
#         fp1[i] = 1
#     #  边界2：p（0：NX，NY）= 1
#     kp2 = np.zeros((l_nodes, q_nodes))
#     fp2 = np.zeros(l_nodes)
#     for i in range(l_nodes):
#         kp2[i][(nx + 1) * nz + i] = 1
#         fp2[i] = 1
#     #  边界3：周期边界条件：p（0，0：NY）=p(NX,0:NY)
#     kp3 = np.zeros((r_nodes, q_nodes))
#     fp3 = np.zeros(r_nodes)
#     for i in range(r_nodes):
#         kp3[i][(i + 1) * (nx + 1)] = 1  # 附加矩阵内，左侧边界置1
#         kp3[i][(i + 1) * (nx + 1) + nx] = -1  # 附加矩阵内，右侧边界置-1
#
#     kp = np.row_stack((kp1, kp2, kp3))  # 边界条件刚度阵组合
#
#     fp = np.hstack((fp1, fp2, fp3))  # 附加矩阵非齐次项组合
#     self.k = np.row_stack((self.k, kp))  # 补充拉格朗日乘子刚度矩阵——下方
#
#     zeros_mat = np.zeros(((2 * l_nodes + r_nodes), (2 * l_nodes + r_nodes)))  # 补充刚度零阵
#     kp_t = np.row_stack((kp.T, zeros_mat))
#     self.k = np.column_stack((self.k, kp_t))  # 补充拉格朗日乘子刚度矩阵——右方
#     self.f = np.hstack((self.f, fp))  # 补充附加矩阵非齐次项
