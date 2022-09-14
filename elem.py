
from base import BaseElem


class Elem(BaseElem):
    def __init__(self, nodes):
        super().__init__(nodes)
        self.avg_p = None

    def avg_p(self):
        avg_p = 0
        for n in range(4):
            avg_p += self.nodes[n].p
        self.avg_p = avg_p / 4
        return self.avg_p
