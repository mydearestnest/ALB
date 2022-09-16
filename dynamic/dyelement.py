from base import BaseElem
import numpy as np


class DyElement(BaseElem):
    def __init__(self, nodes, rank=2, **keyargs):
        self.lx = keyargs['lx']
        self.lz = keyargs['lz']
        self.lr = keyargs['lr']
        self.vx = keyargs['vx']
        super(DyElement, self).__init__(nodes, rank)
        self.h = None
        self.adxc = None
        self.adyc = None
        self.adxct = None
        self.adyct = None
        self.fexc = None
        self.feyc = None
        self.fexct = None
        self.feyct = None
        self.bdxc = None

    def cal_el_adxc(self):
        h = [node.h for node in self.nodes]
        x0 = self.nodes[0].coords[0]
        lx = self.lx
        lz = self.lz
        lr = self.lr
        h0 = h[0]
        h1 = h[1]
        h2 = h[2]
        h3 = h[3]
        k1 = 36 * h1 ** 2 * np.sin(x0) - 36 * h0 ** 2 * np.sin(x0) - 36 * h2 ** 2 * np.sin(
            x0) + 36 * h3 ** 2 * np.sin(x0) + 36 * h0 ** 2 * np.sin(lx + x0) - 36 * h1 ** 2 * np.sin(
            lx + x0) + 36 * h2 ** 2 * np.sin(lx + x0) - 36 * h3 ** 2 * np.sin(lx + x0)
        k0 = k1 + 12 * h1 ** 2 * lx * np.cos(
            lx + x0) + 12 * h3 ** 2 * lx * np.cos(lx + x0) - 36 * h0 ** 2 * lx * np.cos(
            x0) + 24 * h1 ** 2 * lx * np.cos(
            x0) - 36 * h2 ** 2 * lx * np.cos(x0) + 24 * h3 ** 2 * lx * np.cos(x0)
        adxc = [((lz * (k0 - 3 * h0 ** 2 * lr ** 2 * np.sin(
            lx + x0) + 3 * h1 ** 2 * lr ** 2 * np.sin(lx + x0) - h2 ** 2 * lr ** 2 * np.sin(
            lx + x0) + h3 ** 2 * lr ** 2 * np.sin(lx + x0) + 6 * h0 ** 2 * lx ** 3 * np.cos(
            x0) + 6 * h2 ** 2 * lx ** 3 * np.cos(x0) + 3 * h0 ** 2 * lr ** 2 * np.sin(
            x0) - 3 * h1 ** 2 * lr ** 2 * np.sin(
            x0) + h2 ** 2 * lr ** 2 * np.sin(x0) - h3 ** 2 * lr ** 2 * np.sin(x0) + 18 * h0 ** 2 * lx ** 2 * np.sin(
            x0) - 6 * h1 ** 2 * lx ** 2 * np.sin(x0) + 18 * h2 ** 2 * lx ** 2 * np.sin(
            x0) - 6 * h3 ** 2 * lx ** 2 * np.sin(
            x0) - 3 * h1 ** 2 * lr ** 2 * lx * np.cos(lx + x0) - h3 ** 2 * lr ** 2 * lx * np.cos(
            lx + x0) + 3 * h0 ** 2 * lr ** 2 * lx * np.cos(x0) + h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3),
                 -(
                         lz * ( k1 - 12 * h0 ** 2 * lx * np.cos(
                     lx + x0) + 24 * h1 ** 2 * lx * np.cos(lx + x0) - 12 * h2 ** 2 * lx * np.cos(
                     lx + x0) + 24 * h3 ** 2 * lx * np.cos(lx + x0) - 24 * h0 ** 2 * lx * np.cos(
                     x0) + 12 * h1 ** 2 * lx * np.cos(x0) - 24 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(
                     x0) - 3 * h0 ** 2 * lr ** 2 * np.sin(lx + x0) + 3 * h1 ** 2 * lr ** 2 * np.sin(
                     lx + x0) - h2 ** 2 * lr ** 2 * np.sin(lx + x0) + h3 ** 2 * lr ** 2 * np.sin(
                     lx + x0) + 6 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 6 * h3 ** 2 * lx ** 2 * np.sin(
                     lx + x0) + 3 * h0 ** 2 * lr ** 2 * np.sin(x0) - 3 * h1 ** 2 * lr ** 2 * np.sin(
                     x0) + h2 ** 2 * lr ** 2 * np.sin(x0) - h3 ** 2 * lr ** 2 * np.sin(
                     x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(
                     x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(x0) - 3 * h1 ** 2 * lr ** 2 * lx * np.cos(
                     lx + x0) - h3 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + 3 * h0 ** 2 * lr ** 2 * lx * np.cos(
                     x0) + h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), -(lz * (
                k0 + h0 ** 2 * lr ** 2 * np.sin(lx + x0) - h1 ** 2 * lr ** 2 * np.sin(
            lx + x0) + h2 ** 2 * lr ** 2 * np.sin(lx + x0) - h3 ** 2 * lr ** 2 * np.sin(
            lx + x0) + 6 * h0 ** 2 * lx ** 3 * np.cos(x0) + 6 * h2 ** 2 * lx ** 3 * np.cos(
            x0) - h0 ** 2 * lr ** 2 * np.sin(x0) + h1 ** 2 * lr ** 2 * np.sin(x0) - h2 ** 2 * lr ** 2 * np.sin(
            x0) + h3 ** 2 * lr ** 2 * np.sin(x0) + 18 * h0 ** 2 * lx ** 2 * np.sin(x0) - 6 * h1 ** 2 * lx ** 2 * np.sin(
            x0) + 18 * h2 ** 2 * lx ** 2 * np.sin(x0) - 6 * h3 ** 2 * lx ** 2 * np.sin(
            x0) + h1 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + h3 ** 2 * lr ** 2 * lx * np.cos(
            lx + x0) - h0 ** 2 * lr ** 2 * lx * np.cos(x0) - h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), (
                         lz * (k1 - 12 * h0 ** 2 * lx * np.cos(lx + x0) + 24 * h1 ** 2 * lx * np.cos(
                     lx + x0) - 12 * h2 ** 2 * lx * np.cos(lx + x0) + 24 * h3 ** 2 * lx * np.cos(
                     lx + x0) - 24 * h0 ** 2 * lx * np.cos(x0) + 12 * h1 ** 2 * lx * np.cos(
                     x0) - 24 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(x0) + h0 ** 2 * lr ** 2 * np.sin(
                     lx + x0) - h1 ** 2 * lr ** 2 * np.sin(lx + x0) + h2 ** 2 * lr ** 2 * np.sin(
                     lx + x0) - h3 ** 2 * lr ** 2 * np.sin(lx + x0) + 6 * h1 ** 2 * lx ** 2 * np.sin(
                     lx + x0) + 6 * h3 ** 2 * lx ** 2 * np.sin(lx + x0) - h0 ** 2 * lr ** 2 * np.sin(
                     x0) + h1 ** 2 * lr ** 2 * np.sin(x0) - h2 ** 2 * lr ** 2 * np.sin(x0) + h3 ** 2 * lr ** 2 * np.sin(
                     x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(
                     x0) + h1 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + h3 ** 2 * lr ** 2 * lx * np.cos(
                     lx + x0) - h0 ** 2 * lr ** 2 * lx * np.cos(x0) - h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (
                         4 * lx ** 3),), [-(lz * (
                k1 - 12 * h0 ** 2 * lx * np.cos(
            lx + x0) + 24 * h1 ** 2 * lx * np.cos(lx + x0) - 12 * h2 ** 2 * lx * np.cos(
            lx + x0) + 24 * h3 ** 2 * lx * np.cos(lx + x0) - 24 * h0 ** 2 * lx * np.cos(
            x0) + 12 * h1 ** 2 * lx * np.cos(x0) - 24 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(
            x0) - 3 * h0 ** 2 * lr ** 2 * np.sin(lx + x0) + 3 * h1 ** 2 * lr ** 2 * np.sin(
            lx + x0) - h2 ** 2 * lr ** 2 * np.sin(lx + x0) + h3 ** 2 * lr ** 2 * np.sin(
            lx + x0) + 6 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 6 * h3 ** 2 * lx ** 2 * np.sin(
            lx + x0) + 3 * h0 ** 2 * lr ** 2 * np.sin(x0) - 3 * h1 ** 2 * lr ** 2 * np.sin(
            x0) + h2 ** 2 * lr ** 2 * np.sin(x0) - h3 ** 2 * lr ** 2 * np.sin(x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(
            x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(x0) - 3 * h1 ** 2 * lr ** 2 * lx * np.cos(
            lx + x0) - h3 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + 3 * h0 ** 2 * lr ** 2 * lx * np.cos(
            x0) + h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), -(lz * (
                36 * h0 ** 2 * np.sin(x0) - 36 * h1 ** 2 * np.sin(x0) + 36 * h2 ** 2 * np.sin(
            x0) - 36 * h3 ** 2 * np.sin(x0) - 36 * h0 ** 2 * np.sin(lx + x0) + 36 * h1 ** 2 * np.sin(
            lx + x0) - 36 * h2 ** 2 * np.sin(lx + x0) + 36 * h3 ** 2 * np.sin(lx + x0) + 24 * h0 ** 2 * lx * np.cos(
            lx + x0) - 36 * h1 ** 2 * lx * np.cos(lx + x0) + 24 * h2 ** 2 * lx * np.cos(
            lx + x0) - 36 * h3 ** 2 * lx * np.cos(lx + x0) + 12 * h0 ** 2 * lx * np.cos(
            x0) + 12 * h2 ** 2 * lx * np.cos(x0) + 6 * h1 ** 2 * lx ** 3 * np.cos(
            lx + x0) + 6 * h3 ** 2 * lx ** 3 * np.cos(lx + x0) + 3 * h0 ** 2 * lr ** 2 * np.sin(
            lx + x0) - 3 * h1 ** 2 * lr ** 2 * np.sin(lx + x0) + h2 ** 2 * lr ** 2 * np.sin(
            lx + x0) - h3 ** 2 * lr ** 2 * np.sin(lx + x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(
            lx + x0) - 18 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(
            lx + x0) - 18 * h3 ** 2 * lx ** 2 * np.sin(lx + x0) - 3 * h0 ** 2 * lr ** 2 * np.sin(
            x0) + 3 * h1 ** 2 * lr ** 2 * np.sin(x0) - h2 ** 2 * lr ** 2 * np.sin(x0) + h3 ** 2 * lr ** 2 * np.sin(
            x0) + 3 * h1 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + h3 ** 2 * lr ** 2 * lx * np.cos(
            lx + x0) - 3 * h0 ** 2 * lr ** 2 * lx * np.cos(x0) - h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3),
                                          (lz * ( k1 - 12 * h0 ** 2 * lx * np.cos(
                                              lx + x0) + 24 * h1 ** 2 * lx * np.cos(
                                              lx + x0) - 12 * h2 ** 2 * lx * np.cos(
                                              lx + x0) + 24 * h3 ** 2 * lx * np.cos(
                                              lx + x0) - 24 * h0 ** 2 * lx * np.cos(x0) + 12 * h1 ** 2 * lx * np.cos(
                                              x0) - 24 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(
                                              x0) + h0 ** 2 * lr ** 2 * np.sin(lx + x0) - h1 ** 2 * lr ** 2 * np.sin(
                                              lx + x0) + h2 ** 2 * lr ** 2 * np.sin(
                                              lx + x0) - h3 ** 2 * lr ** 2 * np.sin(
                                              lx + x0) + 6 * h1 ** 2 * lx ** 2 * np.sin(
                                              lx + x0) + 6 * h3 ** 2 * lx ** 2 * np.sin(
                                              lx + x0) - h0 ** 2 * lr ** 2 * np.sin(x0) + h1 ** 2 * lr ** 2 * np.sin(
                                              x0) - h2 ** 2 * lr ** 2 * np.sin(x0) + h3 ** 2 * lr ** 2 * np.sin(
                                              x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(
                                              x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(
                                              x0) + h1 ** 2 * lr ** 2 * lx * np.cos(
                                              lx + x0) + h3 ** 2 * lr ** 2 * lx * np.cos(
                                              lx + x0) - h0 ** 2 * lr ** 2 * lx * np.cos(
                                              x0) - h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), (lz * (
                    36 * h0 ** 2 * np.sin(x0) - 36 * h1 ** 2 * np.sin(x0) + 36 * h2 ** 2 * np.sin(
                x0) - 36 * h3 ** 2 * np.sin(x0) - 36 * h0 ** 2 * np.sin(lx + x0) + 36 * h1 ** 2 * np.sin(
                lx + x0) - 36 * h2 ** 2 * np.sin(lx + x0) + 36 * h3 ** 2 * np.sin(lx + x0) + 24 * h0 ** 2 * lx * np.cos(
                lx + x0) - 36 * h1 ** 2 * lx * np.cos(lx + x0) + 24 * h2 ** 2 * lx * np.cos(
                lx + x0) - 36 * h3 ** 2 * lx * np.cos(lx + x0) + 12 * h0 ** 2 * lx * np.cos(
                x0) + 12 * h2 ** 2 * lx * np.cos(x0) + 6 * h1 ** 2 * lx ** 3 * np.cos(
                lx + x0) + 6 * h3 ** 2 * lx ** 3 * np.cos(lx + x0) - h0 ** 2 * lr ** 2 * np.sin(
                lx + x0) + h1 ** 2 * lr ** 2 * np.sin(lx + x0) - h2 ** 2 * lr ** 2 * np.sin(
                lx + x0) + h3 ** 2 * lr ** 2 * np.sin(lx + x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(
                lx + x0) - 18 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(
                lx + x0) - 18 * h3 ** 2 * lx ** 2 * np.sin(lx + x0) + h0 ** 2 * lr ** 2 * np.sin(
                x0) - h1 ** 2 * lr ** 2 * np.sin(x0) + h2 ** 2 * lr ** 2 * np.sin(x0) - h3 ** 2 * lr ** 2 * np.sin(
                x0) - h1 ** 2 * lr ** 2 * lx * np.cos(lx + x0) - h3 ** 2 * lr ** 2 * lx * np.cos(
                lx + x0) + h0 ** 2 * lr ** 2 * lx * np.cos(x0) + h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (
                                                  4 * lx ** 3), ], [-(lz * (
                k0 + h0 ** 2 * lr ** 2 * np.sin(lx + x0) - h1 ** 2 * lr ** 2 * np.sin(
            lx + x0) + h2 ** 2 * lr ** 2 * np.sin(lx + x0) - h3 ** 2 * lr ** 2 * np.sin(
            lx + x0) + 6 * h0 ** 2 * lx ** 3 * np.cos(x0) + 6 * h2 ** 2 * lx ** 3 * np.cos(
            x0) - h0 ** 2 * lr ** 2 * np.sin(x0) + h1 ** 2 * lr ** 2 * np.sin(x0) - h2 ** 2 * lr ** 2 * np.sin(
            x0) + h3 ** 2 * lr ** 2 * np.sin(x0) + 18 * h0 ** 2 * lx ** 2 * np.sin(x0) - 6 * h1 ** 2 * lx ** 2 * np.sin(
            x0) + 18 * h2 ** 2 * lx ** 2 * np.sin(x0) - 6 * h3 ** 2 * lx ** 2 * np.sin(
            x0) + h1 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + h3 ** 2 * lr ** 2 * lx * np.cos(
            lx + x0) - h0 ** 2 * lr ** 2 * lx * np.cos(x0) - h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), (
                                                                            lz * ( k1 - 12 * h0 ** 2 * lx * np.cos(
                                                                        lx + x0) + 24 * h1 ** 2 * lx * np.cos(
                                                                        lx + x0) - 12 * h2 ** 2 * lx * np.cos(
                                                                        lx + x0) + 24 * h3 ** 2 * lx * np.cos(
                                                                        lx + x0) - 24 * h0 ** 2 * lx * np.cos(
                                                                        x0) + 12 * h1 ** 2 * lx * np.cos(
                                                                        x0) - 24 * h2 ** 2 * lx * np.cos(
                                                                        x0) + 12 * h3 ** 2 * lx * np.cos(
                                                                        x0) + h0 ** 2 * lr ** 2 * np.sin(
                                                                        lx + x0) - h1 ** 2 * lr ** 2 * np.sin(
                                                                        lx + x0) + h2 ** 2 * lr ** 2 * np.sin(
                                                                        lx + x0) - h3 ** 2 * lr ** 2 * np.sin(
                                                                        lx + x0) + 6 * h1 ** 2 * lx ** 2 * np.sin(
                                                                        lx + x0) + 6 * h3 ** 2 * lx ** 2 * np.sin(
                                                                        lx + x0) - h0 ** 2 * lr ** 2 * np.sin(
                                                                        x0) + h1 ** 2 * lr ** 2 * np.sin(
                                                                        x0) - h2 ** 2 * lr ** 2 * np.sin(
                                                                        x0) + h3 ** 2 * lr ** 2 * np.sin(
                                                                        x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(
                                                                        x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(
                                                                        x0) + h1 ** 2 * lr ** 2 * lx * np.cos(
                                                                        lx + x0) + h3 ** 2 * lr ** 2 * lx * np.cos(
                                                                        lx + x0) - h0 ** 2 * lr ** 2 * lx * np.cos(
                                                                        x0) - h2 ** 2 * lr ** 2 * lx * np.cos(
                                                                        x0))) / (4 * lx ** 3), (lz * (
                k0 - h0 ** 2 * lr ** 2 * np.sin(lx + x0) + h1 ** 2 * lr ** 2 * np.sin(
            lx + x0) - 3 * h2 ** 2 * lr ** 2 * np.sin(lx + x0) + 3 * h3 ** 2 * lr ** 2 * np.sin(
            lx + x0) + 6 * h0 ** 2 * lx ** 3 * np.cos(x0) + 6 * h2 ** 2 * lx ** 3 * np.cos(
            x0) + h0 ** 2 * lr ** 2 * np.sin(x0) - h1 ** 2 * lr ** 2 * np.sin(x0) + 3 * h2 ** 2 * lr ** 2 * np.sin(
            x0) - 3 * h3 ** 2 * lr ** 2 * np.sin(x0) + 18 * h0 ** 2 * lx ** 2 * np.sin(
            x0) - 6 * h1 ** 2 * lx ** 2 * np.sin(x0) + 18 * h2 ** 2 * lx ** 2 * np.sin(
            x0) - 6 * h3 ** 2 * lx ** 2 * np.sin(x0) - h1 ** 2 * lr ** 2 * lx * np.cos(
            lx + x0) - 3 * h3 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + h0 ** 2 * lr ** 2 * lx * np.cos(
            x0) + 3 * h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), -(lz * (
                k1 - 12 * h0 ** 2 * lx * np.cos(
            lx + x0) + 24 * h1 ** 2 * lx * np.cos(lx + x0) - 12 * h2 ** 2 * lx * np.cos(
            lx + x0) + 24 * h3 ** 2 * lx * np.cos(lx + x0) - 24 * h0 ** 2 * lx * np.cos(
            x0) + 12 * h1 ** 2 * lx * np.cos(x0) - 24 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(
            x0) - h0 ** 2 * lr ** 2 * np.sin(lx + x0) + h1 ** 2 * lr ** 2 * np.sin(
            lx + x0) - 3 * h2 ** 2 * lr ** 2 * np.sin(lx + x0) + 3 * h3 ** 2 * lr ** 2 * np.sin(
            lx + x0) + 6 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 6 * h3 ** 2 * lx ** 2 * np.sin(
            lx + x0) + h0 ** 2 * lr ** 2 * np.sin(x0) - h1 ** 2 * lr ** 2 * np.sin(x0) + 3 * h2 ** 2 * lr ** 2 * np.sin(
            x0) - 3 * h3 ** 2 * lr ** 2 * np.sin(x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(
            x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(x0) - h1 ** 2 * lr ** 2 * lx * np.cos(
            lx + x0) - 3 * h3 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + h0 ** 2 * lr ** 2 * lx * np.cos(
            x0) + 3 * h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), ], [(lz * (
                k1 - 12 * h0 ** 2 * lx * np.cos(
            lx + x0) + 24 * h1 ** 2 * lx * np.cos(lx + x0) - 12 * h2 ** 2 * lx * np.cos(
            lx + x0) + 24 * h3 ** 2 * lx * np.cos(lx + x0) - 24 * h0 ** 2 * lx * np.cos(
            x0) + 12 * h1 ** 2 * lx * np.cos(x0) - 24 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(
            x0) + h0 ** 2 * lr ** 2 * np.sin(lx + x0) - h1 ** 2 * lr ** 2 * np.sin(
            lx + x0) + h2 ** 2 * lr ** 2 * np.sin(lx + x0) - h3 ** 2 * lr ** 2 * np.sin(
            lx + x0) + 6 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 6 * h3 ** 2 * lx ** 2 * np.sin(
            lx + x0) - h0 ** 2 * lr ** 2 * np.sin(x0) + h1 ** 2 * lr ** 2 * np.sin(x0) - h2 ** 2 * lr ** 2 * np.sin(
            x0) + h3 ** 2 * lr ** 2 * np.sin(x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(
            x0) + h1 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + h3 ** 2 * lr ** 2 * lx * np.cos(
            lx + x0) - h0 ** 2 * lr ** 2 * lx * np.cos(x0) - h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), (
                                                                                         lz * (
                                                                                         36 * h0 ** 2 * np.sin(
                                                                                     x0) - 36 * h1 ** 2 * np.sin(
                                                                                     x0) + 36 * h2 ** 2 * np.sin(
                                                                                     x0) - 36 * h3 ** 2 * np.sin(
                                                                                     x0) - 36 * h0 ** 2 * np.sin(
                                                                                     lx + x0) + 36 * h1 ** 2 * np.sin(
                                                                                     lx + x0) - 36 * h2 ** 2 * np.sin(
                                                                                     lx + x0) + 36 * h3 ** 2 * np.sin(
                                                                                     lx + x0) + 24 * h0 ** 2 * lx * np.cos(
                                                                                     lx + x0) - 36 * h1 ** 2 * lx * np.cos(
                                                                                     lx + x0) + 24 * h2 ** 2 * lx * np.cos(
                                                                                     lx + x0) - 36 * h3 ** 2 * lx * np.cos(
                                                                                     lx + x0) + 12 * h0 ** 2 * lx * np.cos(
                                                                                     x0) + 12 * h2 ** 2 * lx * np.cos(
                                                                                     x0) + 6 * h1 ** 2 * lx ** 3 * np.cos(
                                                                                     lx + x0) + 6 * h3 ** 2 * lx ** 3 * np.cos(
                                                                                     lx + x0) - h0 ** 2 * lr ** 2 * np.sin(
                                                                                     lx + x0) + h1 ** 2 * lr ** 2 * np.sin(
                                                                                     lx + x0) - h2 ** 2 * lr ** 2 * np.sin(
                                                                                     lx + x0) + h3 ** 2 * lr ** 2 * np.sin(
                                                                                     lx + x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(
                                                                                     lx + x0) - 18 * h1 ** 2 * lx ** 2 * np.sin(
                                                                                     lx + x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(
                                                                                     lx + x0) - 18 * h3 ** 2 * lx ** 2 * np.sin(
                                                                                     lx + x0) + h0 ** 2 * lr ** 2 * np.sin(
                                                                                     x0) - h1 ** 2 * lr ** 2 * np.sin(
                                                                                     x0) + h2 ** 2 * lr ** 2 * np.sin(
                                                                                     x0) - h3 ** 2 * lr ** 2 * np.sin(
                                                                                     x0) - h1 ** 2 * lr ** 2 * lx * np.cos(
                                                                                     lx + x0) - h3 ** 2 * lr ** 2 * lx * np.cos(
                                                                                     lx + x0) + h0 ** 2 * lr ** 2 * lx * np.cos(
                                                                                     x0) + h2 ** 2 * lr ** 2 * lx * np.cos(
                                                                                     x0))) / (4 * lx ** 3), -(
                lz * (k1 - 12 * h0 ** 2 * lx * np.cos(
            lx + x0) + 24 * h1 ** 2 * lx * np.cos(lx + x0) - 12 * h2 ** 2 * lx * np.cos(
            lx + x0) + 24 * h3 ** 2 * lx * np.cos(lx + x0) - 24 * h0 ** 2 * lx * np.cos(
            x0) + 12 * h1 ** 2 * lx * np.cos(x0) - 24 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(
            x0) - h0 ** 2 * lr ** 2 * np.sin(lx + x0) + h1 ** 2 * lr ** 2 * np.sin(
            lx + x0) - 3 * h2 ** 2 * lr ** 2 * np.sin(lx + x0) + 3 * h3 ** 2 * lr ** 2 * np.sin(
            lx + x0) + 6 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 6 * h3 ** 2 * lx ** 2 * np.sin(
            lx + x0) + h0 ** 2 * lr ** 2 * np.sin(x0) - h1 ** 2 * lr ** 2 * np.sin(x0) + 3 * h2 ** 2 * lr ** 2 * np.sin(
            x0) - 3 * h3 ** 2 * lr ** 2 * np.sin(x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(
            x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(x0) - h1 ** 2 * lr ** 2 * lx * np.cos(
            lx + x0) - 3 * h3 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + h0 ** 2 * lr ** 2 * lx * np.cos(
            x0) + 3 * h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), -(lz * (
                36 * h0 ** 2 * np.sin(x0) - 36 * h1 ** 2 * np.sin(x0) + 36 * h2 ** 2 * np.sin(
            x0) - 36 * h3 ** 2 * np.sin(x0) - 36 * h0 ** 2 * np.sin(lx + x0) + 36 * h1 ** 2 * np.sin(
            lx + x0) - 36 * h2 ** 2 * np.sin(lx + x0) + 36 * h3 ** 2 * np.sin(lx + x0) + 24 * h0 ** 2 * lx * np.cos(
            lx + x0) - 36 * h1 ** 2 * lx * np.cos(lx + x0) + 24 * h2 ** 2 * lx * np.cos(
            lx + x0) - 36 * h3 ** 2 * lx * np.cos(lx + x0) + 12 * h0 ** 2 * lx * np.cos(
            x0) + 12 * h2 ** 2 * lx * np.cos(x0) + 6 * h1 ** 2 * lx ** 3 * np.cos(
            lx + x0) + 6 * h3 ** 2 * lx ** 3 * np.cos(lx + x0) + h0 ** 2 * lr ** 2 * np.sin(
            lx + x0) - h1 ** 2 * lr ** 2 * np.sin(lx + x0) + 3 * h2 ** 2 * lr ** 2 * np.sin(
            lx + x0) - 3 * h3 ** 2 * lr ** 2 * np.sin(lx + x0) + 6 * h0 ** 2 * lx ** 2 * np.sin(
            lx + x0) - 18 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 6 * h2 ** 2 * lx ** 2 * np.sin(
            lx + x0) - 18 * h3 ** 2 * lx ** 2 * np.sin(lx + x0) - h0 ** 2 * lr ** 2 * np.sin(
            x0) + h1 ** 2 * lr ** 2 * np.sin(x0) - 3 * h2 ** 2 * lr ** 2 * np.sin(x0) + 3 * h3 ** 2 * lr ** 2 * np.sin(
            x0) + h1 ** 2 * lr ** 2 * lx * np.cos(lx + x0) + 3 * h3 ** 2 * lr ** 2 * lx * np.cos(
            lx + x0) - h0 ** 2 * lr ** 2 * lx * np.cos(x0) - 3 * h2 ** 2 * lr ** 2 * lx * np.cos(x0))) / (
                                                                                         4 * lx ** 3), ], ]
        adxc = np.array(adxc)
        self.adxc = adxc


    def cal_el_bdxc(self):
        x0 = self.nodes[0].coords[0]
        lx = self.lx
        lz = self.lz
        vx = self.vx
        bdxc = [(lz * vx * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx), -(lz * vx * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx),
         (lz * vx * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx), -(lz * vx * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx), ]
        self.bdxc = np.array(bdxc)

    def cal_keyc(self):
        pass

    def cal_feyc(self):
        pass

    def cal_kexct(self):
        pass

    def cal_fexct(self):
        pass

    def cal_keyct(self):
        pass

    def cal_feyct(self):
        pass
