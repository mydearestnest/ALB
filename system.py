import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse as sp
from scipy.sparse import linalg as sl
import time
from func import direct_fe, direct_ke
from boundary import set_pressure_boundary, set_continuity_boundary
from base import BaseSystem
from node import Node
from elem import Elem


# 定义求解系统

class System(BaseSystem):
    def __init__(self, init_args):
        super().__init__()
        self._init_input_args(init_args)  # 初始化输入参数
        self._init_system_args()  # 初始化系统参数
        self._init_info_args()  # 初始化信息参数，包括程序开始时间与结束时间
        self._mesh_init()  # 网格坐标初始化

    def _init_input_args(self, init_args):
        self.nx = np.int_(init_args['nx'])  # 定义单元数量（Nx，Ny）
        self.nz = np.int_(init_args['nz'])
        self.lx = init_args['lx'] / self.nx  # 定义单元长度
        self.lz = init_args['lz'] / self.nz
        self.w = init_args['w']
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
        self.g = init_args['g'] * np.ones_like(self.orifiec_x)
        self.k = init_args['k']
        self.vx = 3 / 2 * self.u * (self.w * 2 * np.pi / 60) * self.l ** 2 / self.ps / self.c ** 2
        self.lr = self.l / (2 * self.r)
        self.freedoms_norank = (self.nx + 1) * (self.nz + 1)

    def _init_system_args(self):
        # self.k = np.zeros([(nx + 1) * (nz + 1), (nx + 1) * (nz + 1)])  # 定义刚度
        self.k_init = sp.dok_matrix(((self.nx + 1) * (self.nz + 1), (self.nx + 1) * (self.nz + 1)), dtype=np.float32)
        # 使用dok稀疏矩阵内存占用更少，但计算速度降低，主要时间用于组装
        self.f = np.zeros([(self.nx + 1) * (self.nz + 1)])  # 定义非齐次项
        self.nq = []  # 无量纲流量因数
        self.q = []  # 节流孔流量
        self.orifiec_node_no = []  # 节流孔节点位置
        # 输入初始化结束
        # 压力场初始化
        self.p_result = np.zeros_like(self.f)
        # 用于储存上一步的压力值
        self.p_old = np.zeros_like(self.f)
        # 液膜厚度初始化
        self.h_result = None
        # 用于储存补充拉格朗日乘子矩阵
        self.k_add_boundary = None
        self.f_add_boundary = None
        # 用于储存节流孔入口压力计算结果
        self.p_in_ans = []
        # 用于储存节流孔入口压力初值
        self.p_in_init = []

    def _init_info_args(self):
        self.start_time = time.time()
        self.end_time = None

    # 初始化网格，用于画图
    def _mesh_init(self):
        x = np.linspace(0, self.lx * self.nx, self.nx + 1)
        z = np.linspace(0, self.lz * self.nz, self.nz + 1)
        self.X, self.Z = np.meshgrid(x, z)

    # 创建计算域
    def creat_cal_dom(self):
        self.creat_rect_nodes()
        self.crear_rect_elems()

    # 创建矩形单元节点
    def creat_rect_nodes(self, p_set=0.5):
        for i in range(self.nz + 1):
            for j in range(self.nx + 1):
                node = Node(np.array([j * self.lx, i * self.lz]))
                node.p = p_set
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
            h = [elem.nodes[i].h for i in range(elem.rank ** 2)]
            elem.ke = direct_ke(self.lx, self.lz, self.lr, h)
            elem.fe = direct_fe(self.lz, self.vx, h)
            for n in range(4):
                tol_n = elem.nodes[n].number
                self.f[tol_n] += elem.fe[n]  # 组装进总体非齐次项向量
                for m in range(4):
                    tol_m = elem.nodes[m].number
                    self.k_init[tol_n, tol_m] += elem.ke[n, m]  # 组装总刚度矩阵，找到

    # 将拉格朗日乘子阵组装进整体矩阵
    def couple_boundary_matrix(self, kps, fps):
        kp = sp.vstack(kps)  # 边界条件刚度阵组合
        fp = np.hstack(fps)  # 附加矩阵非齐次项组合
        self.k_add_boundary = sp.vstack((self.k_init, kp))  # 补充拉格朗日乘子刚度矩阵——下方
        zeros_mat = sp.coo_matrix((len(fp), len(fp)))  # 补充刚度零阵
        kp_t = sp.vstack((kp.T, zeros_mat))
        self.k_add_boundary = sp.hstack((self.k_add_boundary, kp_t))  # 补充拉格朗日乘子刚度矩阵——右方
        self.f_add_boundary = np.hstack((self.f, fp))  # 补充附加矩阵非齐次项
        #  由于在初始化时没有考虑边界条件，导致额外的拉格朗日乘子没有被添加
        if len(self.p_result) < len(self.f_add_boundary):
            add_length = abs(len(self.f_add_boundary) - len(self.p_result))
            self.p_result = np.hstack((self.p_result, np.zeros(add_length)))
            print('已补充压力拉格朗日乘子')

    def boundarys_setting(self):
        nx = self.nx
        nz = self.nz
        #  边界1：p（0：NX，0）= 1
        q_nodes = self.freedoms_norank
        l_nodes = nx + 1
        r_nodes = nz - 1
        #  获取轴承两端节点编号
        nodes1 = [i for i in range(l_nodes)]
        nodes2 = [(nx + 1) * nz + i for i in range(l_nodes)]
        #  根据节点两端编号，将两侧节点设置为所设定的无量纲压力值，若不输入压力默认为0.5
        kp1, fp1 = set_pressure_boundary(q_nodes, nodes1, p_set=self.pr / self.ps)
        kp2, fp2 = set_pressure_boundary(q_nodes, nodes2, p_set=self.pr / self.ps)
        #  获取连续边界上的节点编号
        nodes3 = [(i + 1) * (nx + 1) for i in range(r_nodes)]
        nodes4 = [(i + 1) * (nx + 1) + nx for i in range(r_nodes)]
        #  设置连续边界
        kp3, fp3 = set_continuity_boundary(q_nodes, nodes3, nodes4)
        #  将边界条件组装进入矩阵中
        self.couple_boundary_matrix([kp1, kp2, kp3], [fp1, fp2, fp3])

    def boundary_setting_with_lagrange(self):
        nx = self.nx
        nz = self.nz
        #  边界1：p（0：NX，0）= 1
        q_nodes = self.freedoms_norank
        l_nodes = nx + 1
        r_nodes = nz - 1
        kp1 = sp.dok_matrix((l_nodes, q_nodes), dtype=np.float32)
        fp1 = np.zeros(l_nodes)
        for i in range(l_nodes):
            kp1[i, i] = 1
            fp1[i] = self.pr / self.ps
        #  边界2：p（0：NX，NY）= 1
        kp2 = sp.dok_matrix((l_nodes, q_nodes), dtype=np.float32)
        fp2 = np.zeros(l_nodes)
        for i in range(l_nodes):
            kp2[i, (nx + 1) * nz + i] = 1
            fp2[i] = self.pr / self.ps
        #  边界3：周期边界条件：p（0，0：NY）= p(NX,0:NY)
        kp3 = sp.dok_matrix((r_nodes, q_nodes), dtype=np.float32)
        fp3 = np.zeros(r_nodes)
        for i in range(r_nodes):
            kp3[i, (i + 1) * (nx + 1)] = 1  # 附加矩阵内，左侧边界置1
            kp3[i, (i + 1) * (nx + 1) + nx] = -1  # 附加矩阵内，右侧边界置-1

        kp = sp.vstack((kp1, kp2, kp3))  # 边界条件刚度阵组合
        fp = np.hstack((fp1, fp2, fp3))  # 附加矩阵非齐次项组合
        self.k_add_boundary = sp.vstack((self.k_init, kp))  # 补充拉格朗日乘子刚度矩阵——下方

        zeros_mat = sp.coo_matrix(((2 * l_nodes + r_nodes), (2 * l_nodes + r_nodes)))  # 补充刚度零阵
        kp_t = sp.vstack((kp.T, zeros_mat))
        self.k_add_boundary = sp.hstack((self.k_add_boundary, kp_t))  # 补充拉格朗日乘子刚度矩阵——右方
        self.f_add_boundary = np.hstack((self.f, fp))  # 补充附加矩阵非齐次项

    # 计算厚度场
    def cal_h(self, angel=0):
        nx = self.nx
        nz = self.nz
        hz = np.zeros([self.nx + 1, self.nz + 1])
        for i in range(nx + 1):
            for j in range(nz + 1):
                hz[i][j] = self.nodes[i + j * (nx + 1)].h = 1 + self.e * np.cos(i * self.lx + angel)
        self.h_result = hz

    # 添加均压槽处的油膜厚度
    # 输入x方向与z方向的油槽无量纲始末位置，并输入相应无量纲厚度
    # x方向与z方向要求输入为长度为2的ndarray类型数据，且在[0,1]之间
    def add_h_tank(self, h_tanks):
        if h_tanks.shape[0] == 5 and isinstance(h_tanks, np.ndarray):
            x_nodes = [i * self.nx for i in h_tanks[0:2]]
            z_nodes = [i * self.nz for i in h_tanks[2:4]]
            h_tank = h_tanks[4]
            x_nodes = list(map(int, x_nodes))
            z_nodes = list(map(int, z_nodes))
            for i in range(x_nodes[0], x_nodes[1] + 1):
                for j in range(z_nodes[0], z_nodes[1] + 1):
                    self.nodes[i + j * (self.nx + 1)].h += h_tank
                    self.h_result[i, j] += h_tank
        else:
            print('错误！请传入ndarray类型数据，且长度为2')

    # 多个均压槽添加
    # 要求输入为凹槽位置信息数组，单行为一个凹槽
    # 多次调用add_h_tank以到达目的
    def add_h_tanks(self, args):
        if len(args.shape) > 1:
            for i in range(len(args)):
                arg = args[i]
                self.add_h_tank(arg)
        else:
            self.add_h_tank(args)

    # 计算压力场
    def cal_p_direct(self):
        # 将用于迭代的压力进行存储
        self.p_old = self.p_result
        sp_p = sp.csc_matrix(self.k_add_boundary)
        self.p_result = sl.spsolve(sp_p, self.f_add_boundary)
        # 将计算后压力重新赋给各节点
        for i in range(len(self.nodes)):
            self.nodes[i].p = self.p_result[i]
        print('calculation of pressure is over')
        p_in_ans = []
        for node_no in self.orifiec_node_no:
            p_in_ans.append(self.p_result[node_no])  # 将节流孔的计算压力值记录储存
        self.p_in_ans.append(p_in_ans)

    def cal_p_iter_newton(self):
        # 将用于迭代的压力进行存储
        self.p_old = self.p_result
        # 将流量项对入口压力的Jacobi矩阵与刚度阵相加，不过由于Jacobi矩阵项较少，因此各项单独加入刚度矩阵
        # 构成dp前的迭代矩阵
        kp = sp.csc_matrix(self.k_add_boundary)  # 迭代矩阵初始化sp_p = sp.csc_matrix(self.kc)
        # # 由于f组装拉格朗日乘子阵后形成的fc的shape发生变化，因此p与fc的维度可能不一致，需要补充增加p的维度
        for i, node_no in enumerate(self.orifiec_node_no):
            dp = self.ps - self.ps * self.p_result[node_no]
            q_dp = 12 * self.u * self.lr * self.cd * self.a0 / (
                    self.ps * self.c ** 3) * np.sqrt(
                0.5 / self.rho / dp)
            kp[node_no, node_no] += q_dp
        fp = self.f_add_boundary - self.k_add_boundary @ self.p_result
        dp = sl.spsolve(kp, fp)
        # 更新压力
        self.dp = dp
        self.p_result = self.p_result + dp
        p_in_ans = []
        # 记录节流孔计算结果
        for node_no in self.orifiec_node_no:
            p_in_ans.append(self.p_result[node_no])  # 将节流孔的计算压力值记录储存
        self.p_in_ans.append(p_in_ans)
        # 更新各节点压力值
        for i in range(len(self.nodes)):
            self.nodes[i].p = self.p_result[self.nodes[i].number]
        return dp

    def setting_orifce_position(self, orifice_x, orifice_z):
        if orifice_x is None and orifice_z is None:
            pass
        elif isinstance(orifice_x, np.float) and isinstance(orifice_z, np.float):
            orifice_No = int(orifice_z * self.nz) * (self.nx + 1) + int(orifice_x * self.nx)
            self.orifiec_node_no.append(orifice_No)
        elif len(orifice_x) != len(orifice_z):
            print('error!节流孔x方向坐标数量必须等于y方向坐标')
        else:
            for i in range(len(orifice_x)):
                orifice_No = int(orifice_z[i] * self.nz) * (self.nx + 1) + int(orifice_x[i] * self.nx)
                self.orifiec_node_no.append(orifice_No)

    #  通过松弛因子/比例系数调整迭代初始值
    #  只有在使用add_fqs输入method！=direct才使用，仅针对供油孔处流场压力变化迭代初始值
    def ip_init(self):
        for i, node_no in enumerate(self.orifiec_node_no):
            if len(self.p_in_init) == 0:
                p_init = []
                p_init.append()


    #  计算无量纲流量因子,添加静压源项
    #  输入为method方法名，以及相关参数
    #  method为orifice时，将会调用config/orifice_position.txt内的供油孔位置参数添加油源项
    #  method为direct时，需要在**keys输入无量纲流量nqs，以及对应节点位置node_nos，最好多个供油孔，单个可能报错
    def add_fqs(self, method='orifice', **keys):
        # 当输入为orifce时，调用小孔节流模型
        if method == 'orifice':
            nqs = []  # 本次无量纲流量暂存列表
            for i, node_no in enumerate(self.orifiec_node_no):
                p_init = self.p_in_init[-1][i]
                dp = self.ps - p_init * self.ps
                fq = self.cal_fq(dp)
                nqs.append(fq)
                # 将流量项加入非齐次项
                self.f_add_boundary[node_no] += fq  # 这里非齐次项为正，由推导获得
            self.nq.append(nqs)  # 将各次迭代前的无量纲流量记录
            return nqs
        elif method == 'direct':
            nqs = keys['nq']
            node_nos = keys['node_nos']
            if (isinstance(nqs, list) or isinstance(nqs, np.ndarray)) and len(nqs) == len(node_nos):
                for i in range(len(node_nos)):
                    self.f_add_boundary[node_nos[i]] += nqs[i]
                self.nq.append(nqs)
                return True
            else:
                print('直接添加流量项错误！需要保证传入的流量列表与对应节点列表数量一致，且数据类型为列表或者ndarray')
                return False

    #  计算无量纲流量
    def cal_fq(self, dp):
        if dp > 0:
            cq = 12 * self.u * self.lr * self.cd * self.a0 / self.c ** 3 * np.sqrt(2 / self.ps / self.rho)
            test_q = cq * np.sqrt(dp / self.ps)
            print(test_q)
            q = self.cd * self.a0 * np.sqrt(2 * dp / self.rho)  # 小孔节流孔流量
            print('小孔流量:', q)
            fq = 12 * self.u * self.lr * q / (self.ps * self.c ** 3)  # 流量项计算
            return fq
        else:
            print('计算无量纲流量cal_fq错误！请传入正压差')
            return False

    #  计算无量纲流量导数
    def cal_q_dp(self, dp):
        if dp <= 0:
            q_dp = 12 * self.u * self.lr * self.cd * self.a0 / (
                    self.ps * self.c ** 3) * np.sqrt(
                0.5 / self.rho / dp)
            return q_dp
        else:
            print('计算无量纲流量导数错误！请传入正压差')
            return False

    #  残差计算
    def cal_error(self):
        dp = self.p_old[0:self.freedoms_norank] - self.p_result[0:self.freedoms_norank]
        error = np.abs((np.sum(dp)) / np.sum(self.p_result[0:self.freedoms_norank]))
        return error

    #  三维表面画图
    def plot_p_3d(self):
        ax = plt.axes(projection="3d")
        p = self.p_result[0:(self.nx + 1) * (self.nz + 1)].reshape(self.nz + 1, self.nx + 1)
        ax.plot_surface(self.X, self.Z, p,
                        rstride=1,  # rstride（row）指定行的跨度
                        cstride=1,  # cstride(column)指定列的跨度,
                        cmap="rainbow")
        plt.show()

    # 存储压力数据
    def save_output_p(self, filename='output_p.dat'):
        out_put_p = self.p_result[0:(self.nx + 1) * (self.nz + 1)].reshape(self.nz + 1, self.nx + 1)  # 将压力数据存为dat
        np.savetxt(filename, out_put_p)
