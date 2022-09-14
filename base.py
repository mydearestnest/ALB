import numpy as np

'''定义全局节点,全局节点属性包括全局节点编号与全局节点坐标'''


class BaseNode:
    def __init__(self, coords, dim=2):
        self._number = None  # 节点编号
        self._coords = coords  # 节点坐标
        self._dim = dim  # 节点维度

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


'''定义单元，其中包括内部节点与单元编号'''


class BaseElem:
    def __init__(self, nodes, rank=2):
        self._nodes = nodes  # 定义单元内节点
        self._number = None  # 定义单元编号
        self._rank = rank  # 定义单元形函数阶数
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


class BaseSystem:
    def __init__(self):
        self.nodes = {}  # 定义网格上所有节点
        self.elems = {}  # 定义网格上所有单元

    # 增加单个节点
    def add_node(self, node):
        assert issubclass(type(node), BaseNode), '请添加节点'
        if node.number is None:
            node.number = self.non
        self.nodes[node.number] = node

    # 增加节点集
    def add_nodes(self, *nodes):
        if isinstance(nodes, list) or isinstance(nodes, tuple) or isinstance(nodes, np.ndarray):
            for node in nodes:
                self.add_node(node)
        else:
            self.add_node(nodes)

    # 增加单个单元
    def add_elem(self, elem):
        assert issubclass(type(elem), BaseElem)
        if elem.number is None:
            elem.number = self.noe
        self.elems[elem.number] = elem

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
