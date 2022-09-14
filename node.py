from base import BaseNode

'''定义全局节点,全局节点属性包括全局节点编号与全局节点坐标'''


class Node(BaseNode):
    def __init__(self, coords, dim=2):
        super().__init__(coords, dim)
        self._h = 0  # 节点厚度
        self._p = 0  # 节点压力

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
