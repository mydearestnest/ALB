import numpy as np
from base import BaseElem


class FilmElem(BaseElem):
    def __init__(self, nodes, **args):
        super().__init__(nodes)
        self.avg_p = None
        self.lr = args['lr']
        self.lx = args['lx']
        self.lz = args['lz']
        self.vx = args['vx']

    def avg_p(self):
        avg_p = 0
        for n in range(4):
            avg_p += self.nodes[n].p
        self.avg_p = avg_p / 4
        return self.avg_p

    def cal_ke(self):
        h = [node.h for node in self.nodes]
        h0 = h[0]
        h1 = h[1]
        h2 = h[2]
        h3 = h[3]
        lr = self.lr
        lx = self.lx
        lz = self.lz
        self.ke = np.array([[((h0 ** 3 * lr ** 2 * lz ** 2) / 8 + (h1 ** 3 * lr ** 2 * lz ** 2) / 8 + (
                h2 ** 3 * lr ** 2 * lz ** 2) / 24 + (h3 ** 3 * lr ** 2 * lz ** 2) / 24) / (lx * lz) + (
                                     lx * (h0 ** 3 / 8 + h1 ** 3 / 24 + h2 ** 3 / 8 + h3 ** 3 / 24)) / lz,
                             (lx * (h0 ** 3 / 24 + h1 ** 3 / 24 + h2 ** 3 / 24 + h3 ** 3 / 24)) / lz - (
                                     (h0 ** 3 * lr ** 2 * lz ** 2) / 8 + (h1 ** 3 * lr ** 2 * lz ** 2) / 8 + (
                                     h2 ** 3 * lr ** 2 * lz ** 2) / 24 + (h3 ** 3 * lr ** 2 * lz ** 2) / 24) / (
                                         lx * lz), (
                                     (h0 ** 3 * lr ** 2 * lz ** 2) / 24 + (h1 ** 3 * lr ** 2 * lz ** 2) / 24 + (
                                     h2 ** 3 * lr ** 2 * lz ** 2) / 24 + (h3 ** 3 * lr ** 2 * lz ** 2) / 24) / (
                                         lx * lz) - (
                                     lx * (h0 ** 3 / 8 + h1 ** 3 / 24 + h2 ** 3 / 8 + h3 ** 3 / 24)) / lz,
                             -((lx ** 2 + lr ** 2 * lz ** 2) * (h0 ** 3 + h1 ** 3 + h2 ** 3 + h3 ** 3)) / (
                                         24 * lx * lz), ],
                            [
                                (lx * (h0 ** 3 / 24 + h1 ** 3 / 24 + h2 ** 3 / 24 + h3 ** 3 / 24)) / lz - (
                                        (h0 ** 3 * lr ** 2 * lz ** 2) / 8 + (h1 ** 3 * lr ** 2 * lz ** 2) / 8 + (
                                        h2 ** 3 * lr ** 2 * lz ** 2) / 24 + (h3 ** 3 * lr ** 2 * lz ** 2) / 24) / (
                                            lx * lz),
                                (
                                        (h0 ** 3 * lr ** 2 * lz ** 2) / 8 + (h1 ** 3 * lr ** 2 * lz ** 2) / 8 + (
                                        h2 ** 3 * lr ** 2 * lz ** 2) / 24 + (h3 ** 3 * lr ** 2 * lz ** 2) / 24) / (
                                        lx * lz) + (
                                        lx * (h0 ** 3 / 24 + h1 ** 3 / 8 + h2 ** 3 / 24 + h3 ** 3 / 8)) / lz,
                                -((lx ** 2 + lr ** 2 * lz ** 2) * (h0 ** 3 + h1 ** 3 + h2 ** 3 + h3 ** 3)) / (
                                            24 * lx * lz),
                                (
                                        (h0 ** 3 * lr ** 2 * lz ** 2) / 24 + (h1 ** 3 * lr ** 2 * lz ** 2) / 24 + (
                                        h2 ** 3 * lr ** 2 * lz ** 2) / 24 + (h3 ** 3 * lr ** 2 * lz ** 2) / 24) / (
                                        lx * lz) - (
                                        lx * (h0 ** 3 / 24 + h1 ** 3 / 8 + h2 ** 3 / 24 + h3 ** 3 / 8)) / lz, ], [((
                                                                                                                           h0 ** 3 * lr ** 2 * lz ** 2) / 24 + (
                                                                                                                           h1 ** 3 * lr ** 2 * lz ** 2) / 24 + (
                                                                                                                           h2 ** 3 * lr ** 2 * lz ** 2) / 24 + (
                                                                                                                           h3 ** 3 * lr ** 2 * lz ** 2) / 24) / (
                                                                                                                          lx * lz) - (
                                                                                                                          lx * (
                                                                                                                          h0 ** 3 / 8 + h1 ** 3 / 24 + h2 ** 3 / 8 + h3 ** 3 / 24)) / lz,
                                                                                                                  -((
                                                                                                                            lx ** 2 + lr ** 2 * lz ** 2) * (
                                                                                                                            h0 ** 3 + h1 ** 3 + h2 ** 3 + h3 ** 3)) / (
                                                                                                                          24 * lx * lz),
                                                                                                                  ((
                                                                                                                           h0 ** 3 * lx ** 2) / 8 + (
                                                                                                                           h1 ** 3 * lx ** 2) / 24 + (
                                                                                                                           h2 ** 3 * lx ** 2) / 8 + (
                                                                                                                           h3 ** 3 * lx ** 2) / 24) / (
                                                                                                                          lx * lz) + (
                                                                                                                          lz * (
                                                                                                                          (
                                                                                                                                  h0 ** 3 * lr ** 2) / 24 + (
                                                                                                                                  h1 ** 3 * lr ** 2) / 24 + (
                                                                                                                                  h2 ** 3 * lr ** 2) / 8 + (
                                                                                                                                  h3 ** 3 * lr ** 2) / 8)) / lx,
                                                                                                                  ((
                                                                                                                           h0 ** 3 * lx ** 2) / 24 + (
                                                                                                                           h1 ** 3 * lx ** 2) / 24 + (
                                                                                                                           h2 ** 3 * lx ** 2) / 24 + (
                                                                                                                           h3 ** 3 * lx ** 2) / 24) / (
                                                                                                                          lx * lz) - (
                                                                                                                          lz * (
                                                                                                                          (
                                                                                                                                  h0 ** 3 * lr ** 2) / 24 + (
                                                                                                                                  h1 ** 3 * lr ** 2) / 24 + (
                                                                                                                                  h2 ** 3 * lr ** 2) / 8 + (
                                                                                                                                  h3 ** 3 * lr ** 2) / 8)) / lx, ],
                            [-((lx ** 2 + lr ** 2 * lz ** 2) * (h0 ** 3 + h1 ** 3 + h2 ** 3 + h3 ** 3)) / (
                                        24 * lx * lz), (
                                     (h0 ** 3 * lr ** 2 * lz ** 2) / 24 + (h1 ** 3 * lr ** 2 * lz ** 2) / 24 + (
                                     h2 ** 3 * lr ** 2 * lz ** 2) / 24 + (h3 ** 3 * lr ** 2 * lz ** 2) / 24) / (
                                         lx * lz) - (
                                     lx * (h0 ** 3 / 24 + h1 ** 3 / 8 + h2 ** 3 / 24 + h3 ** 3 / 8)) / lz, (
                                     (h0 ** 3 * lx ** 2) / 24 + (h1 ** 3 * lx ** 2) / 24 + (h2 ** 3 * lx ** 2) / 24 + (
                                     h3 ** 3 * lx ** 2) / 24) / (lx * lz) - (lz * (
                                    (h0 ** 3 * lr ** 2) / 24 + (h1 ** 3 * lr ** 2) / 24 + (h2 ** 3 * lr ** 2) / 8 + (
                                    h3 ** 3 * lr ** 2) / 8)) / lx, (
                                     (h0 ** 3 * lr ** 2 * lz ** 2) / 24 + (h1 ** 3 * lr ** 2 * lz ** 2) / 24 + (
                                     h2 ** 3 * lr ** 2 * lz ** 2) / 8 + (h3 ** 3 * lr ** 2 * lz ** 2) / 8) / (
                                         lx * lz) + (
                                     lx * (h0 ** 3 / 24 + h1 ** 3 / 8 + h2 ** 3 / 24 + h3 ** 3 / 8)) / lz, ], ])
        return True

    def cal_fe(self):
        h = [node.h for node in self.nodes]
        h0 = h[0]
        h1 = h[1]
        h2 = h[2]
        h3 = h[3]
        lz = self.lz
        vx = self.vx
        self.fe = np.array([-(lz * vx * (2 * h0 + 2 * h1 + h2 + h3)) / 12, (lz * vx * (2 * h0 + 2 * h1 + h2 + h3)) / 12,
                            -(lz * vx * (h0 + h1 + 2 * h2 + 2 * h3)) / 12,
                            (lz * vx * (h0 + h1 + 2 * h2 + 2 * h3)) / 12, ])
        return True
