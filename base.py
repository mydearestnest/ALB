import numpy as np
from matplotlib import pyplot as plt
from func import direct_fe, direct_ke, cal_h, nard_f

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


# 定义求解系统

class System:
    def __init__(self, nx, nz, lx, lz, vx, lr, e):
        self.nodes = {}  # 定义网格上所有节点
        self.elems = {}  # 定义网格上所有单元
        self.nx = nx  # 定义单元数量（Nx，Ny）
        self.nz = nz
        self.lx = lx  # 定义单元长度
        self.lz = lz
        self.k = np.zeros([(nx + 1) * (nz + 1), (nx + 1) * (nz + 1)])  # 定义刚度
        self.f = np.zeros([(nx + 1) * (nz + 1)])  # 定义非齐次项
        self.vx = vx
        self.lr = lr
        self.e = e
        self.p = None
        self.X, self.Z = self.mesh_init()
        self.h = None

    # 初始化网格，用于画图
    def mesh_init(self):
        x = np.linspace(0, self.lx * self.nx, self.nx + 1)
        z = np.linspace(0, self.lz * self.nz, self.nz + 1)
        x, z = np.meshgrid(x, z)
        return x, z

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

    # 计算刚度并组装
    def cal_k(self):
        for el in self.elems:
            elem = self.elems[el]
            x0 = elem.nodes[0].coords[0]
            z0 = elem.nodes[0].coords[1]
            elem.h = [elem.nodes[0].h,elem.nodes[1].h,elem.nodes[2].h,elem.nodes[3].h]
            elem.ke = direct_ke(self.lx,self.lz,self.lr,elem.h)
            elem.fe = direct_fe(self.lx,self.lr,self.vx,elem.h)
            for n in range(4):
                # ans = nard_f(n, x0, z0, self.lx, self.lz, self.e, self.vx)
                # elem.fe[n] = ans[0]
                self.f[elem.nodes[n].number] += elem.fe[n]  # 组装进总体非齐次项向量
                for m in range(4):
                    tol_n = elem.nodes[n].number
                    tol_m = elem.nodes[m].number
                    # ans = nard_k(n, m, x0, z0, self.lr, self.lx, self.lz, self.e)
                    # elem.ke
                    self.k[tol_n][tol_m] += elem.ke[n, m]  # 组装总刚度矩阵，找到

    def set_bondary(self):
        nx = self.nx
        nz = self.nz
        #  对称边界条件设置

        #  边界1：p（0：NX，0）= 1
        q_nodes = (nx + 1) * (nz + 1)
        l_nodes = nx + 1
        r_nodes = nz - 1
        kp1 = np.zeros((l_nodes, q_nodes))
        fp1 = np.zeros(l_nodes)
        for i in range(l_nodes):
            kp1[i][i] = 1
            fp1[i] = 1
        #  边界2：p（0：NX，NY）= 1
        kp2 = np.zeros((l_nodes, q_nodes))
        fp2 = np.zeros(l_nodes)
        for i in range(l_nodes):
            kp2[i][(nx + 1) * nz + i] = 1
            fp2[i] = 1
        #  边界3：周期边界条件：p（0，0：NY）=p(NX,0:NY)
        kp3 = np.zeros((r_nodes, q_nodes))
        fp3 = np.zeros(r_nodes)
        for i in range(r_nodes):
            kp3[i][(i + 1) * (nx + 1)] = 1  # 附加矩阵内，左侧边界置1
            kp3[i][(i + 1) * (nx + 1) + nx] = -1  # 附加矩阵内，右侧边界置-1

        kp = np.row_stack((kp1, kp2, kp3))  # 边界条件刚度阵组合

        fp = np.hstack((fp1, fp2, fp3))  # 附加矩阵非齐次项组合
        self.k = np.row_stack((self.k, kp))  # 补充拉格朗日乘子刚度矩阵——下方

        zeros_mat = np.zeros(((2 * l_nodes + r_nodes), (2 * l_nodes + r_nodes)))  # 补充刚度零阵
        kp_t = np.row_stack((kp.T, zeros_mat))
        self.k = np.column_stack((self.k, kp_t))  # 补充拉格朗日乘子刚度矩阵——右方
        self.f = np.hstack((self.f, fp))  # 补充附加矩阵非齐次项

    # 计算厚度场
    def cal_h(self):
        self.h = cal_h(self.nx, self.nz, self.e)

    # 计算压力场
    def cal_p(self):
        self.p = np.linalg.solve(self.k, self.f)
        self.p = self.p[0:(self.nx + 1) * (self.nz + 1)].reshape(self.nz + 1, self.nx + 1)

    def plot_p_3d(self):
        ax = plt.axes(projection="3d")
        ax.plot_surface(self.X, self.Z, self.p, cmap="rainbow")
        plt.show()
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
