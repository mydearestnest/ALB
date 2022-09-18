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
        self.ad = None
        self.bd = None

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
        adxc = [[(18 * h1 ** 2 * np.sin(x0) - 18 * h0 ** 2 * np.sin(x0) - 18 * h2 ** 2 * np.sin(
            x0) + 18 * h3 ** 2 * np.sin(x0) + 18 * h0 ** 2 * np.sin(lx + x0) - 18 * h1 ** 2 * np.sin(
            lx + x0) + 18 * h2 ** 2 * np.sin(lx + x0) - 18 * h3 ** 2 * np.sin(lx + x0) + 6 * h1 ** 2 * lx * np.cos(
            lx + x0) + 6 * h3 ** 2 * lx * np.cos(lx + x0) - 18 * h0 ** 2 * lx * np.cos(x0) + 12 * h1 ** 2 * lx * np.cos(
            x0) - 18 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(x0) + 3 * h0 ** 2 * lx ** 3 * np.cos(
            x0) + 3 * h2 ** 2 * lx ** 3 * np.cos(x0) + 9 * h0 ** 2 * lx ** 2 * np.sin(
            x0) - 3 * h1 ** 2 * lx ** 2 * np.sin(x0) + 9 * h2 ** 2 * lx ** 2 * np.sin(
            x0) - 3 * h3 ** 2 * lx ** 2 * np.sin(x0)) / (2 * lx ** 3 * lz) + (lr ** 2 * lz * (
                3 * h0 ** 2 * np.sin(x0) - 3 * h1 ** 2 * np.sin(x0) + h2 ** 2 * np.sin(x0) - h3 ** 2 * np.sin(
            x0) - 3 * h0 ** 2 * np.sin(lx + x0) + 3 * h1 ** 2 * np.sin(lx + x0) - h2 ** 2 * np.sin(
            lx + x0) + h3 ** 2 * np.sin(lx + x0) - 3 * h1 ** 2 * lx * np.cos(lx + x0) - h3 ** 2 * lx * np.cos(
            lx + x0) + 3 * h0 ** 2 * lx * np.cos(x0) + h2 ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), - (
                18 * h1 ** 2 * np.sin(x0) - 18 * h0 ** 2 * np.sin(x0) - 18 * h2 ** 2 * np.sin(
            x0) + 18 * h3 ** 2 * np.sin(x0) + 18 * h0 ** 2 * np.sin(lx + x0) - 18 * h1 ** 2 * np.sin(
            lx + x0) + 18 * h2 ** 2 * np.sin(lx + x0) - 18 * h3 ** 2 * np.sin(lx + x0) - 6 * h0 ** 2 * lx * np.cos(
            lx + x0) + 12 * h1 ** 2 * lx * np.cos(lx + x0) - 6 * h2 ** 2 * lx * np.cos(
            lx + x0) + 12 * h3 ** 2 * lx * np.cos(lx + x0) - 12 * h0 ** 2 * lx * np.cos(
            x0) + 6 * h1 ** 2 * lx * np.cos(x0) - 12 * h2 ** 2 * lx * np.cos(x0) + 6 * h3 ** 2 * lx * np.cos(
            x0) + 3 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.sin(
            lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(x0)) / (
                         2 * lx ** 3 * lz) - (lr ** 2 * lz * (
                3 * h0 ** 2 * np.sin(x0) - 3 * h1 ** 2 * np.sin(x0) + h2 ** 2 * np.sin(x0) - h3 ** 2 * np.sin(
            x0) - 3 * h0 ** 2 * np.sin(lx + x0) + 3 * h1 ** 2 * np.sin(lx + x0) - h2 ** 2 * np.sin(
            lx + x0) + h3 ** 2 * np.sin(lx + x0) - 3 * h1 ** 2 * lx * np.cos(lx + x0) - h3 ** 2 * lx * np.cos(
            lx + x0) + 3 * h0 ** 2 * lx * np.cos(x0) + h2 ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), (
                         lr ** 2 * lz * (
                         h0 ** 2 * np.sin(x0) - h1 ** 2 * np.sin(x0) + h2 ** 2 * np.sin(x0) - h3 ** 2 * np.sin(
                     x0) - h0 ** 2 * np.sin(lx + x0) + h1 ** 2 * np.sin(lx + x0) - h2 ** 2 * np.sin(
                     lx + x0) + h3 ** 2 * np.sin(lx + x0) - h1 ** 2 * lx * np.cos(
                     lx + x0) - h3 ** 2 * lx * np.cos(lx + x0) + h0 ** 2 * lx * np.cos(
                     x0) + h2 ** 2 * lx * np.cos(x0))) / (4 * lx ** 3) - (
                         18 * h1 ** 2 * np.sin(x0) - 18 * h0 ** 2 * np.sin(x0) - 18 * h2 ** 2 * np.sin(
                     x0) + 18 * h3 ** 2 * np.sin(x0) + 18 * h0 ** 2 * np.sin(lx + x0) - 18 * h1 ** 2 * np.sin(
                     lx + x0) + 18 * h2 ** 2 * np.sin(lx + x0) - 18 * h3 ** 2 * np.sin(
                     lx + x0) + 6 * h1 ** 2 * lx * np.cos(lx + x0) + 6 * h3 ** 2 * lx * np.cos(
                     lx + x0) - 18 * h0 ** 2 * lx * np.cos(x0) + 12 * h1 ** 2 * lx * np.cos(
                     x0) - 18 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(
                     x0) + 3 * h0 ** 2 * lx ** 3 * np.cos(x0) + 3 * h2 ** 2 * lx ** 3 * np.cos(
                     x0) + 9 * h0 ** 2 * lx ** 2 * np.sin(x0) - 3 * h1 ** 2 * lx ** 2 * np.sin(
                     x0) + 9 * h2 ** 2 * lx ** 2 * np.sin(x0) - 3 * h3 ** 2 * lx ** 2 * np.sin(x0)) / (
                         2 * lx ** 3 * lz), (
                         18 * h1 ** 2 * np.sin(x0) - 18 * h0 ** 2 * np.sin(x0) - 18 * h2 ** 2 * np.sin(
                     x0) + 18 * h3 ** 2 * np.sin(x0) + 18 * h0 ** 2 * np.sin(lx + x0) - 18 * h1 ** 2 * np.sin(
                     lx + x0) + 18 * h2 ** 2 * np.sin(lx + x0) - 18 * h3 ** 2 * np.sin(
                     lx + x0) - 6 * h0 ** 2 * lx * np.cos(lx + x0) + 12 * h1 ** 2 * lx * np.cos(
                     lx + x0) - 6 * h2 ** 2 * lx * np.cos(lx + x0) + 12 * h3 ** 2 * lx * np.cos(
                     lx + x0) - 12 * h0 ** 2 * lx * np.cos(x0) + 6 * h1 ** 2 * lx * np.cos(
                     x0) - 12 * h2 ** 2 * lx * np.cos(x0) + 6 * h3 ** 2 * lx * np.cos(
                     x0) + 3 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.sin(
                     lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(x0)) / (
                         2 * lx ** 3 * lz) - (lr ** 2 * lz * (
                h0 ** 2 * np.sin(x0) - h1 ** 2 * np.sin(x0) + h2 ** 2 * np.sin(x0) - h3 ** 2 * np.sin(
            x0) - h0 ** 2 * np.sin(lx + x0) + h1 ** 2 * np.sin(lx + x0) - h2 ** 2 * np.sin(
            lx + x0) + h3 ** 2 * np.sin(lx + x0) - h1 ** 2 * lx * np.cos(lx + x0) - h3 ** 2 * lx * np.cos(
            lx + x0) + h0 ** 2 * lx * np.cos(x0) + h2 ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), ], [- (
                18 * h1 ** 2 * np.sin(x0) - 18 * h0 ** 2 * np.sin(x0) - 18 * h2 ** 2 * np.sin(
            x0) + 18 * h3 ** 2 * np.sin(x0) + 18 * h0 ** 2 * np.sin(lx + x0) - 18 * h1 ** 2 * np.sin(
            lx + x0) + 18 * h2 ** 2 * np.sin(lx + x0) - 18 * h3 ** 2 * np.sin(lx + x0) - 6 * h0 ** 2 * lx * np.cos(
            lx + x0) + 12 * h1 ** 2 * lx * np.cos(lx + x0) - 6 * h2 ** 2 * lx * np.cos(
            lx + x0) + 12 * h3 ** 2 * lx * np.cos(lx + x0) - 12 * h0 ** 2 * lx * np.cos(
            x0) + 6 * h1 ** 2 * lx * np.cos(x0) - 12 * h2 ** 2 * lx * np.cos(x0) + 6 * h3 ** 2 * lx * np.cos(
            x0) + 3 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.sin(
            lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(x0)) / (
                                                                                                            2 * lx ** 3 * lz) - (
                                                                                                            lr ** 2 * lz * (
                                                                                                            3 * h0 ** 2 * np.sin(
                                                                                                        x0) - 3 * h1 ** 2 * np.sin(
                                                                                                        x0) + h2 ** 2 * np.sin(
                                                                                                        x0) - h3 ** 2 * np.sin(
                                                                                                        x0) - 3 * h0 ** 2 * np.sin(
                                                                                                        lx + x0) + 3 * h1 ** 2 * np.sin(
                                                                                                        lx + x0) - h2 ** 2 * np.sin(
                                                                                                        lx + x0) + h3 ** 2 * np.sin(
                                                                                                        lx + x0) - 3 * h1 ** 2 * lx * np.cos(
                                                                                                        lx + x0) - h3 ** 2 * lx * np.cos(
                                                                                                        lx + x0) + 3 * h0 ** 2 * lx * np.cos(
                                                                                                        x0) + h2 ** 2 * lx * np.cos(
                                                                                                        x0))) / (
                                                                                                            4 * lx ** 3),
                                                                                                    (
                                                                                                            lr ** 2 * lz * (
                                                                                                            3 * h0 ** 2 * np.sin(
                                                                                                        x0) - 3 * h1 ** 2 * np.sin(
                                                                                                        x0) + h2 ** 2 * np.sin(
                                                                                                        x0) - h3 ** 2 * np.sin(
                                                                                                        x0) - 3 * h0 ** 2 * np.sin(
                                                                                                        lx + x0) + 3 * h1 ** 2 * np.sin(
                                                                                                        lx + x0) - h2 ** 2 * np.sin(
                                                                                                        lx + x0) + h3 ** 2 * np.sin(
                                                                                                        lx + x0) - 3 * h1 ** 2 * lx * np.cos(
                                                                                                        lx + x0) - h3 ** 2 * lx * np.cos(
                                                                                                        lx + x0) + 3 * h0 ** 2 * lx * np.cos(
                                                                                                        x0) + h2 ** 2 * lx * np.cos(
                                                                                                        x0))) / (
                                                                                                            4 * lx ** 3) - (
                                                                                                            18 * h0 ** 2 * np.sin(
                                                                                                        x0) - 18 * h1 ** 2 * np.sin(
                                                                                                        x0) + 18 * h2 ** 2 * np.sin(
                                                                                                        x0) - 18 * h3 ** 2 * np.sin(
                                                                                                        x0) - 18 * h0 ** 2 * np.sin(
                                                                                                        lx + x0) + 18 * h1 ** 2 * np.sin(
                                                                                                        lx + x0) - 18 * h2 ** 2 * np.sin(
                                                                                                        lx + x0) + 18 * h3 ** 2 * np.sin(
                                                                                                        lx + x0) + 12 * h0 ** 2 * lx * np.cos(
                                                                                                        lx + x0) - 18 * h1 ** 2 * lx * np.cos(
                                                                                                        lx + x0) + 12 * h2 ** 2 * lx * np.cos(
                                                                                                        lx + x0) - 18 * h3 ** 2 * lx * np.cos(
                                                                                                        lx + x0) + 6 * h0 ** 2 * lx * np.cos(
                                                                                                        x0) + 6 * h2 ** 2 * lx * np.cos(
                                                                                                        x0) + 3 * h1 ** 2 * lx ** 3 * np.cos(
                                                                                                        lx + x0) + 3 * h3 ** 2 * lx ** 3 * np.cos(
                                                                                                        lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(
                                                                                                        lx + x0) - 9 * h1 ** 2 * lx ** 2 * np.sin(
                                                                                                        lx + x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(
                                                                                                        lx + x0) - 9 * h3 ** 2 * lx ** 2 * np.sin(
                                                                                                        lx + x0)) / (
                                                                                                            2 * lx ** 3 * lz),
                                                                                                    (
                                                                                                            18 * h1 ** 2 * np.sin(
                                                                                                        x0) - 18 * h0 ** 2 * np.sin(
                                                                                                        x0) - 18 * h2 ** 2 * np.sin(
                                                                                                        x0) + 18 * h3 ** 2 * np.sin(
                                                                                                        x0) + 18 * h0 ** 2 * np.sin(
                                                                                                        lx + x0) - 18 * h1 ** 2 * np.sin(
                                                                                                        lx + x0) + 18 * h2 ** 2 * np.sin(
                                                                                                        lx + x0) - 18 * h3 ** 2 * np.sin(
                                                                                                        lx + x0) - 6 * h0 ** 2 * lx * np.cos(
                                                                                                        lx + x0) + 12 * h1 ** 2 * lx * np.cos(
                                                                                                        lx + x0) - 6 * h2 ** 2 * lx * np.cos(
                                                                                                        lx + x0) + 12 * h3 ** 2 * lx * np.cos(
                                                                                                        lx + x0) - 12 * h0 ** 2 * lx * np.cos(
                                                                                                        x0) + 6 * h1 ** 2 * lx * np.cos(
                                                                                                        x0) - 12 * h2 ** 2 * lx * np.cos(
                                                                                                        x0) + 6 * h3 ** 2 * lx * np.cos(
                                                                                                        x0) + 3 * h1 ** 2 * lx ** 2 * np.sin(
                                                                                                        lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.sin(
                                                                                                        lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(
                                                                                                        x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(
                                                                                                        x0)) / (
                                                                                                            2 * lx ** 3 * lz) - (
                                                                                                            lr ** 2 * lz * (
                                                                                                            h0 ** 2 * np.sin(
                                                                                                        x0) - h1 ** 2 * np.sin(
                                                                                                        x0) + h2 ** 2 * np.sin(
                                                                                                        x0) - h3 ** 2 * np.sin(
                                                                                                        x0) - h0 ** 2 * np.sin(
                                                                                                        lx + x0) + h1 ** 2 * np.sin(
                                                                                                        lx + x0) - h2 ** 2 * np.sin(
                                                                                                        lx + x0) + h3 ** 2 * np.sin(
                                                                                                        lx + x0) - h1 ** 2 * lx * np.cos(
                                                                                                        lx + x0) - h3 ** 2 * lx * np.cos(
                                                                                                        lx + x0) + h0 ** 2 * lx * np.cos(
                                                                                                        x0) + h2 ** 2 * lx * np.cos(
                                                                                                        x0))) / (
                                                                                                            4 * lx ** 3),
                                                                                                    (
                                                                                                            18 * h0 ** 2 * np.sin(
                                                                                                        x0) - 18 * h1 ** 2 * np.sin(
                                                                                                        x0) + 18 * h2 ** 2 * np.sin(
                                                                                                        x0) - 18 * h3 ** 2 * np.sin(
                                                                                                        x0) - 18 * h0 ** 2 * np.sin(
                                                                                                        lx + x0) + 18 * h1 ** 2 * np.sin(
                                                                                                        lx + x0) - 18 * h2 ** 2 * np.sin(
                                                                                                        lx + x0) + 18 * h3 ** 2 * np.sin(
                                                                                                        lx + x0) + 12 * h0 ** 2 * lx * np.cos(
                                                                                                        lx + x0) - 18 * h1 ** 2 * lx * np.cos(
                                                                                                        lx + x0) + 12 * h2 ** 2 * lx * np.cos(
                                                                                                        lx + x0) - 18 * h3 ** 2 * lx * np.cos(
                                                                                                        lx + x0) + 6 * h0 ** 2 * lx * np.cos(
                                                                                                        x0) + 6 * h2 ** 2 * lx * np.cos(
                                                                                                        x0) + 3 * h1 ** 2 * lx ** 3 * np.cos(
                                                                                                        lx + x0) + 3 * h3 ** 2 * lx ** 3 * np.cos(
                                                                                                        lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(
                                                                                                        lx + x0) - 9 * h1 ** 2 * lx ** 2 * np.sin(
                                                                                                        lx + x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(
                                                                                                        lx + x0) - 9 * h3 ** 2 * lx ** 2 * np.sin(
                                                                                                        lx + x0)) / (
                                                                                                            2 * lx ** 3 * lz) + (
                                                                                                            lr ** 2 * lz * (
                                                                                                            h0 ** 2 * np.sin(
                                                                                                        x0) - h1 ** 2 * np.sin(
                                                                                                        x0) + h2 ** 2 * np.sin(
                                                                                                        x0) - h3 ** 2 * np.sin(
                                                                                                        x0) - h0 ** 2 * np.sin(
                                                                                                        lx + x0) + h1 ** 2 * np.sin(
                                                                                                        lx + x0) - h2 ** 2 * np.sin(
                                                                                                        lx + x0) + h3 ** 2 * np.sin(
                                                                                                        lx + x0) - h1 ** 2 * lx * np.cos(
                                                                                                        lx + x0) - h3 ** 2 * lx * np.cos(
                                                                                                        lx + x0) + h0 ** 2 * lx * np.cos(
                                                                                                        x0) + h2 ** 2 * lx * np.cos(
                                                                                                        x0))) / (
                                                                                                            4 * lx ** 3), ],
                [(lr ** 2 * lz * (h0 ** 2 * np.sin(x0) - h1 ** 2 * np.sin(x0) + h2 ** 2 * np.sin(x0) - h3 ** 2 * np.sin(
                    x0) - h0 ** 2 * np.sin(lx + x0) + h1 ** 2 * np.sin(lx + x0) - h2 ** 2 * np.sin(
                    lx + x0) + h3 ** 2 * np.sin(lx + x0) - h1 ** 2 * lx * np.cos(lx + x0) - h3 ** 2 * lx * np.cos(
                    lx + x0) + h0 ** 2 * lx * np.cos(x0) + h2 ** 2 * lx * np.cos(x0))) / (4 * lx ** 3) - (
                         18 * h1 ** 2 * np.sin(x0) - 18 * h0 ** 2 * np.sin(x0) - 18 * h2 ** 2 * np.sin(
                     x0) + 18 * h3 ** 2 * np.sin(x0) + 18 * h0 ** 2 * np.sin(lx + x0) - 18 * h1 ** 2 * np.sin(
                     lx + x0) + 18 * h2 ** 2 * np.sin(lx + x0) - 18 * h3 ** 2 * np.sin(
                     lx + x0) + 6 * h1 ** 2 * lx * np.cos(lx + x0) + 6 * h3 ** 2 * lx * np.cos(
                     lx + x0) - 18 * h0 ** 2 * lx * np.cos(x0) + 12 * h1 ** 2 * lx * np.cos(
                     x0) - 18 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(
                     x0) + 3 * h0 ** 2 * lx ** 3 * np.cos(x0) + 3 * h2 ** 2 * lx ** 3 * np.cos(
                     x0) + 9 * h0 ** 2 * lx ** 2 * np.sin(x0) - 3 * h1 ** 2 * lx ** 2 * np.sin(
                     x0) + 9 * h2 ** 2 * lx ** 2 * np.sin(x0) - 3 * h3 ** 2 * lx ** 2 * np.sin(x0)) / (
                         2 * lx ** 3 * lz), (
                         18 * h1 ** 2 * np.sin(x0) - 18 * h0 ** 2 * np.sin(x0) - 18 * h2 ** 2 * np.sin(
                     x0) + 18 * h3 ** 2 * np.sin(x0) + 18 * h0 ** 2 * np.sin(lx + x0) - 18 * h1 ** 2 * np.sin(
                     lx + x0) + 18 * h2 ** 2 * np.sin(lx + x0) - 18 * h3 ** 2 * np.sin(
                     lx + x0) - 6 * h0 ** 2 * lx * np.cos(lx + x0) + 12 * h1 ** 2 * lx * np.cos(
                     lx + x0) - 6 * h2 ** 2 * lx * np.cos(lx + x0) + 12 * h3 ** 2 * lx * np.cos(
                     lx + x0) - 12 * h0 ** 2 * lx * np.cos(x0) + 6 * h1 ** 2 * lx * np.cos(
                     x0) - 12 * h2 ** 2 * lx * np.cos(x0) + 6 * h3 ** 2 * lx * np.cos(
                     x0) + 3 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.sin(
                     lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(x0)) / (
                         2 * lx ** 3 * lz) - (lr ** 2 * lz * (
                        h0 ** 2 * np.sin(x0) - h1 ** 2 * np.sin(x0) + h2 ** 2 * np.sin(x0) - h3 ** 2 * np.sin(
                    x0) - h0 ** 2 * np.sin(lx + x0) + h1 ** 2 * np.sin(lx + x0) - h2 ** 2 * np.sin(
                    lx + x0) + h3 ** 2 * np.sin(lx + x0) - h1 ** 2 * lx * np.cos(lx + x0) - h3 ** 2 * lx * np.cos(
                    lx + x0) + h0 ** 2 * lx * np.cos(x0) + h2 ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), (
                         18 * h1 ** 2 * np.sin(x0) - 18 * h0 ** 2 * np.sin(x0) - 18 * h2 ** 2 * np.sin(
                     x0) + 18 * h3 ** 2 * np.sin(x0) + 18 * h0 ** 2 * np.sin(lx + x0) - 18 * h1 ** 2 * np.sin(
                     lx + x0) + 18 * h2 ** 2 * np.sin(lx + x0) - 18 * h3 ** 2 * np.sin(
                     lx + x0) + 6 * h1 ** 2 * lx * np.cos(lx + x0) + 6 * h3 ** 2 * lx * np.cos(
                     lx + x0) - 18 * h0 ** 2 * lx * np.cos(x0) + 12 * h1 ** 2 * lx * np.cos(
                     x0) - 18 * h2 ** 2 * lx * np.cos(x0) + 12 * h3 ** 2 * lx * np.cos(
                     x0) + 3 * h0 ** 2 * lx ** 3 * np.cos(x0) + 3 * h2 ** 2 * lx ** 3 * np.cos(
                     x0) + 9 * h0 ** 2 * lx ** 2 * np.sin(x0) - 3 * h1 ** 2 * lx ** 2 * np.sin(
                     x0) + 9 * h2 ** 2 * lx ** 2 * np.sin(x0) - 3 * h3 ** 2 * lx ** 2 * np.sin(x0)) / (
                         2 * lx ** 3 * lz) + (lr ** 2 * lz * (
                        h0 ** 2 * np.sin(x0) - h1 ** 2 * np.sin(x0) + 3 * h2 ** 2 * np.sin(
                    x0) - 3 * h3 ** 2 * np.sin(x0) - h0 ** 2 * np.sin(lx + x0) + h1 ** 2 * np.sin(
                    lx + x0) - 3 * h2 ** 2 * np.sin(lx + x0) + 3 * h3 ** 2 * np.sin(
                    lx + x0) - h1 ** 2 * lx * np.cos(lx + x0) - 3 * h3 ** 2 * lx * np.cos(
                    lx + x0) + h0 ** 2 * lx * np.cos(x0) + 3 * h2 ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), - (
                        18 * h1 ** 2 * np.sin(x0) - 18 * h0 ** 2 * np.sin(x0) - 18 * h2 ** 2 * np.sin(
                    x0) + 18 * h3 ** 2 * np.sin(x0) + 18 * h0 ** 2 * np.sin(lx + x0) - 18 * h1 ** 2 * np.sin(
                    lx + x0) + 18 * h2 ** 2 * np.sin(lx + x0) - 18 * h3 ** 2 * np.sin(
                    lx + x0) - 6 * h0 ** 2 * lx * np.cos(lx + x0) + 12 * h1 ** 2 * lx * np.cos(
                    lx + x0) - 6 * h2 ** 2 * lx * np.cos(lx + x0) + 12 * h3 ** 2 * lx * np.cos(
                    lx + x0) - 12 * h0 ** 2 * lx * np.cos(x0) + 6 * h1 ** 2 * lx * np.cos(
                    x0) - 12 * h2 ** 2 * lx * np.cos(x0) + 6 * h3 ** 2 * lx * np.cos(
                    x0) + 3 * h1 ** 2 * lx ** 2 * np.sin(lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.sin(
                    lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(x0)) / (
                         2 * lx ** 3 * lz) - (lr ** 2 * lz * (
                        h0 ** 2 * np.sin(x0) - h1 ** 2 * np.sin(x0) + 3 * h2 ** 2 * np.sin(
                    x0) - 3 * h3 ** 2 * np.sin(x0) - h0 ** 2 * np.sin(lx + x0) + h1 ** 2 * np.sin(
                    lx + x0) - 3 * h2 ** 2 * np.sin(lx + x0) + 3 * h3 ** 2 * np.sin(
                    lx + x0) - h1 ** 2 * lx * np.cos(lx + x0) - 3 * h3 ** 2 * lx * np.cos(
                    lx + x0) + h0 ** 2 * lx * np.cos(x0) + 3 * h2 ** 2 * lx * np.cos(x0))) / (4 * lx ** 3), ], [(
                                                                                                                        18 * h1 ** 2 * np.sin(
                                                                                                                    x0) - 18 * h0 ** 2 * np.sin(
                                                                                                                    x0) - 18 * h2 ** 2 * np.sin(
                                                                                                                    x0) + 18 * h3 ** 2 * np.sin(
                                                                                                                    x0) + 18 * h0 ** 2 * np.sin(
                                                                                                                    lx + x0) - 18 * h1 ** 2 * np.sin(
                                                                                                                    lx + x0) + 18 * h2 ** 2 * np.sin(
                                                                                                                    lx + x0) - 18 * h3 ** 2 * np.sin(
                                                                                                                    lx + x0) - 6 * h0 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + 12 * h1 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - 6 * h2 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + 12 * h3 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - 12 * h0 ** 2 * lx * np.cos(
                                                                                                                    x0) + 6 * h1 ** 2 * lx * np.cos(
                                                                                                                    x0) - 12 * h2 ** 2 * lx * np.cos(
                                                                                                                    x0) + 6 * h3 ** 2 * lx * np.cos(
                                                                                                                    x0) + 3 * h1 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(
                                                                                                                    x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(
                                                                                                                    x0)) / (
                                                                                                                        2 * lx ** 3 * lz) - (
                                                                                                                        lr ** 2 * lz * (
                                                                                                                        h0 ** 2 * np.sin(
                                                                                                                    x0) - h1 ** 2 * np.sin(
                                                                                                                    x0) + h2 ** 2 * np.sin(
                                                                                                                    x0) - h3 ** 2 * np.sin(
                                                                                                                    x0) - h0 ** 2 * np.sin(
                                                                                                                    lx + x0) + h1 ** 2 * np.sin(
                                                                                                                    lx + x0) - h2 ** 2 * np.sin(
                                                                                                                    lx + x0) + h3 ** 2 * np.sin(
                                                                                                                    lx + x0) - h1 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - h3 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + h0 ** 2 * lx * np.cos(
                                                                                                                    x0) + h2 ** 2 * lx * np.cos(
                                                                                                                    x0))) / (
                                                                                                                        4 * lx ** 3),
                                                                                                                (
                                                                                                                        18 * h0 ** 2 * np.sin(
                                                                                                                    x0) - 18 * h1 ** 2 * np.sin(
                                                                                                                    x0) + 18 * h2 ** 2 * np.sin(
                                                                                                                    x0) - 18 * h3 ** 2 * np.sin(
                                                                                                                    x0) - 18 * h0 ** 2 * np.sin(
                                                                                                                    lx + x0) + 18 * h1 ** 2 * np.sin(
                                                                                                                    lx + x0) - 18 * h2 ** 2 * np.sin(
                                                                                                                    lx + x0) + 18 * h3 ** 2 * np.sin(
                                                                                                                    lx + x0) + 12 * h0 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - 18 * h1 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + 12 * h2 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - 18 * h3 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + 6 * h0 ** 2 * lx * np.cos(
                                                                                                                    x0) + 6 * h2 ** 2 * lx * np.cos(
                                                                                                                    x0) + 3 * h1 ** 2 * lx ** 3 * np.cos(
                                                                                                                    lx + x0) + 3 * h3 ** 2 * lx ** 3 * np.cos(
                                                                                                                    lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0) - 9 * h1 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0) - 9 * h3 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0)) / (
                                                                                                                        2 * lx ** 3 * lz) + (
                                                                                                                        lr ** 2 * lz * (
                                                                                                                        h0 ** 2 * np.sin(
                                                                                                                    x0) - h1 ** 2 * np.sin(
                                                                                                                    x0) + h2 ** 2 * np.sin(
                                                                                                                    x0) - h3 ** 2 * np.sin(
                                                                                                                    x0) - h0 ** 2 * np.sin(
                                                                                                                    lx + x0) + h1 ** 2 * np.sin(
                                                                                                                    lx + x0) - h2 ** 2 * np.sin(
                                                                                                                    lx + x0) + h3 ** 2 * np.sin(
                                                                                                                    lx + x0) - h1 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - h3 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + h0 ** 2 * lx * np.cos(
                                                                                                                    x0) + h2 ** 2 * lx * np.cos(
                                                                                                                    x0))) / (
                                                                                                                        4 * lx ** 3),
                                                                                                                - (
                                                                                                                        18 * h1 ** 2 * np.sin(
                                                                                                                    x0) - 18 * h0 ** 2 * np.sin(
                                                                                                                    x0) - 18 * h2 ** 2 * np.sin(
                                                                                                                    x0) + 18 * h3 ** 2 * np.sin(
                                                                                                                    x0) + 18 * h0 ** 2 * np.sin(
                                                                                                                    lx + x0) - 18 * h1 ** 2 * np.sin(
                                                                                                                    lx + x0) + 18 * h2 ** 2 * np.sin(
                                                                                                                    lx + x0) - 18 * h3 ** 2 * np.sin(
                                                                                                                    lx + x0) - 6 * h0 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + 12 * h1 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - 6 * h2 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + 12 * h3 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - 12 * h0 ** 2 * lx * np.cos(
                                                                                                                    x0) + 6 * h1 ** 2 * lx * np.cos(
                                                                                                                    x0) - 12 * h2 ** 2 * lx * np.cos(
                                                                                                                    x0) + 6 * h3 ** 2 * lx * np.cos(
                                                                                                                    x0) + 3 * h1 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(
                                                                                                                    x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(
                                                                                                                    x0)) / (
                                                                                                                        2 * lx ** 3 * lz) - (
                                                                                                                        lr ** 2 * lz * (
                                                                                                                        h0 ** 2 * np.sin(
                                                                                                                    x0) - h1 ** 2 * np.sin(
                                                                                                                    x0) + 3 * h2 ** 2 * np.sin(
                                                                                                                    x0) - 3 * h3 ** 2 * np.sin(
                                                                                                                    x0) - h0 ** 2 * np.sin(
                                                                                                                    lx + x0) + h1 ** 2 * np.sin(
                                                                                                                    lx + x0) - 3 * h2 ** 2 * np.sin(
                                                                                                                    lx + x0) + 3 * h3 ** 2 * np.sin(
                                                                                                                    lx + x0) - h1 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - 3 * h3 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + h0 ** 2 * lx * np.cos(
                                                                                                                    x0) + 3 * h2 ** 2 * lx * np.cos(
                                                                                                                    x0))) / (
                                                                                                                        4 * lx ** 3),
                                                                                                                (
                                                                                                                        lr ** 2 * lz * (
                                                                                                                        h0 ** 2 * np.sin(
                                                                                                                    x0) - h1 ** 2 * np.sin(
                                                                                                                    x0) + 3 * h2 ** 2 * np.sin(
                                                                                                                    x0) - 3 * h3 ** 2 * np.sin(
                                                                                                                    x0) - h0 ** 2 * np.sin(
                                                                                                                    lx + x0) + h1 ** 2 * np.sin(
                                                                                                                    lx + x0) - 3 * h2 ** 2 * np.sin(
                                                                                                                    lx + x0) + 3 * h3 ** 2 * np.sin(
                                                                                                                    lx + x0) - h1 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - 3 * h3 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + h0 ** 2 * lx * np.cos(
                                                                                                                    x0) + 3 * h2 ** 2 * lx * np.cos(
                                                                                                                    x0))) / (
                                                                                                                        4 * lx ** 3) - (
                                                                                                                        18 * h0 ** 2 * np.sin(
                                                                                                                    x0) - 18 * h1 ** 2 * np.sin(
                                                                                                                    x0) + 18 * h2 ** 2 * np.sin(
                                                                                                                    x0) - 18 * h3 ** 2 * np.sin(
                                                                                                                    x0) - 18 * h0 ** 2 * np.sin(
                                                                                                                    lx + x0) + 18 * h1 ** 2 * np.sin(
                                                                                                                    lx + x0) - 18 * h2 ** 2 * np.sin(
                                                                                                                    lx + x0) + 18 * h3 ** 2 * np.sin(
                                                                                                                    lx + x0) + 12 * h0 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - 18 * h1 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + 12 * h2 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) - 18 * h3 ** 2 * lx * np.cos(
                                                                                                                    lx + x0) + 6 * h0 ** 2 * lx * np.cos(
                                                                                                                    x0) + 6 * h2 ** 2 * lx * np.cos(
                                                                                                                    x0) + 3 * h1 ** 2 * lx ** 3 * np.cos(
                                                                                                                    lx + x0) + 3 * h3 ** 2 * lx ** 3 * np.cos(
                                                                                                                    lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0) - 9 * h1 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0) + 3 * h2 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0) - 9 * h3 ** 2 * lx ** 2 * np.sin(
                                                                                                                    lx + x0)) / (
                                                                                                                        2 * lx ** 3 * lz), ], ]
        adxc = np.array(adxc)
        self.ad = adxc

    def cal_el_bdxc(self):
        x0 = self.nodes[0].coords[0]
        lx = self.lx
        lz = self.lz
        vx = self.vx
        bdxc = [(lz * vx * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx),
                -(lz * vx * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx),
                (lz * vx * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx),
                -(lz * vx * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx), ]
        self.bd = np.array(bdxc)

    def cal_el_adyc(self):
        h = [node.h for node in self.nodes]
        x0 = self.nodes[0].coords[0]
        lx = self.lx
        lz = self.lz
        lr = self.lr
        h0 = h[0]
        h1 = h[1]
        h2 = h[2]
        h3 = h[3]
        adyc = [[(18 * h0 ** 2 * np.cos(x0) - 18 * h1 ** 2 * np.cos(x0) + 18 * h2 ** 2 * np.cos(
            x0) - 18 * h3 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(lx + x0) + 18 * h1 ** 2 * np.cos(
            lx + x0) - 18 * h2 ** 2 * np.cos(lx + x0) + 18 * h3 ** 2 * np.cos(lx + x0) + 6 * h1 ** 2 * lx * np.sin(
            lx + x0) + 6 * h3 ** 2 * lx * np.sin(lx + x0) - 18 * h0 ** 2 * lx * np.sin(x0) + 12 * h1 ** 2 * lx * np.sin(
            x0) - 18 * h2 ** 2 * lx * np.sin(x0) + 12 * h3 ** 2 * lx * np.sin(x0) - 9 * h0 ** 2 * lx ** 2 * np.cos(
            x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(x0) - 9 * h2 ** 2 * lx ** 2 * np.cos(
            x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(x0) + 3 * h0 ** 2 * lx ** 3 * np.sin(
            x0) + 3 * h2 ** 2 * lx ** 3 * np.sin(x0)) / (2 * lx ** 3 * lz) - (lr ** 2 * lz * (
                3 * h0 ** 2 * np.cos(x0) - 3 * h1 ** 2 * np.cos(x0) + h2 ** 2 * np.cos(x0) - h3 ** 2 * np.cos(
            x0) - 3 * h0 ** 2 * np.cos(lx + x0) + 3 * h1 ** 2 * np.cos(lx + x0) - h2 ** 2 * np.cos(
            lx + x0) + h3 ** 2 * np.cos(lx + x0) + 3 * h1 ** 2 * lx * np.sin(lx + x0) + h3 ** 2 * lx * np.sin(
            lx + x0) - 3 * h0 ** 2 * lx * np.sin(x0) - h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3), (
                         18 * h1 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(x0) - 18 * h2 ** 2 * np.cos(
                     x0) + 18 * h3 ** 2 * np.cos(x0) + 18 * h0 ** 2 * np.cos(lx + x0) - 18 * h1 ** 2 * np.cos(
                     lx + x0) + 18 * h2 ** 2 * np.cos(lx + x0) - 18 * h3 ** 2 * np.cos(
                     lx + x0) + 6 * h0 ** 2 * lx * np.sin(lx + x0) - 12 * h1 ** 2 * lx * np.sin(
                     lx + x0) + 6 * h2 ** 2 * lx * np.sin(lx + x0) - 12 * h3 ** 2 * lx * np.sin(
                     lx + x0) + 12 * h0 ** 2 * lx * np.sin(x0) - 6 * h1 ** 2 * lx * np.sin(
                     x0) + 12 * h2 ** 2 * lx * np.sin(x0) - 6 * h3 ** 2 * lx * np.sin(
                     x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(
                     lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.cos(x0) + 3 * h2 ** 2 * lx ** 2 * np.cos(x0)) / (
                         2 * lx ** 3 * lz) + (lr ** 2 * lz * (
                3 * h0 ** 2 * np.cos(x0) - 3 * h1 ** 2 * np.cos(x0) + h2 ** 2 * np.cos(x0) - h3 ** 2 * np.cos(
            x0) - 3 * h0 ** 2 * np.cos(lx + x0) + 3 * h1 ** 2 * np.cos(lx + x0) - h2 ** 2 * np.cos(
            lx + x0) + h3 ** 2 * np.cos(lx + x0) + 3 * h1 ** 2 * lx * np.sin(lx + x0) + h3 ** 2 * lx * np.sin(
            lx + x0) - 3 * h0 ** 2 * lx * np.sin(x0) - h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3), - (
                18 * h0 ** 2 * np.cos(x0) - 18 * h1 ** 2 * np.cos(x0) + 18 * h2 ** 2 * np.cos(
            x0) - 18 * h3 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(lx + x0) + 18 * h1 ** 2 * np.cos(
            lx + x0) - 18 * h2 ** 2 * np.cos(lx + x0) + 18 * h3 ** 2 * np.cos(lx + x0) + 6 * h1 ** 2 * lx * np.sin(
            lx + x0) + 6 * h3 ** 2 * lx * np.sin(lx + x0) - 18 * h0 ** 2 * lx * np.sin(
            x0) + 12 * h1 ** 2 * lx * np.sin(x0) - 18 * h2 ** 2 * lx * np.sin(x0) + 12 * h3 ** 2 * lx * np.sin(
            x0) - 9 * h0 ** 2 * lx ** 2 * np.cos(x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(
            x0) - 9 * h2 ** 2 * lx ** 2 * np.cos(x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(
            x0) + 3 * h0 ** 2 * lx ** 3 * np.sin(x0) + 3 * h2 ** 2 * lx ** 3 * np.sin(x0)) / (2 * lx ** 3 * lz) - (
                         lr ** 2 * lz * (
                         h0 ** 2 * np.cos(x0) - h1 ** 2 * np.cos(x0) + h2 ** 2 * np.cos(x0) - h3 ** 2 * np.cos(
                     x0) - h0 ** 2 * np.cos(lx + x0) + h1 ** 2 * np.cos(lx + x0) - h2 ** 2 * np.cos(
                     lx + x0) + h3 ** 2 * np.cos(lx + x0) + h1 ** 2 * lx * np.sin(
                     lx + x0) + h3 ** 2 * lx * np.sin(lx + x0) - h0 ** 2 * lx * np.sin(
                     x0) - h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3), (lr ** 2 * lz * (
                h0 ** 2 * np.cos(x0) - h1 ** 2 * np.cos(x0) + h2 ** 2 * np.cos(x0) - h3 ** 2 * np.cos(
            x0) - h0 ** 2 * np.cos(lx + x0) + h1 ** 2 * np.cos(lx + x0) - h2 ** 2 * np.cos(
            lx + x0) + h3 ** 2 * np.cos(lx + x0) + h1 ** 2 * lx * np.sin(lx + x0) + h3 ** 2 * lx * np.sin(
            lx + x0) - h0 ** 2 * lx * np.sin(x0) - h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3) - (
                         18 * h1 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(x0) - 18 * h2 ** 2 * np.cos(
                     x0) + 18 * h3 ** 2 * np.cos(x0) + 18 * h0 ** 2 * np.cos(lx + x0) - 18 * h1 ** 2 * np.cos(
                     lx + x0) + 18 * h2 ** 2 * np.cos(lx + x0) - 18 * h3 ** 2 * np.cos(
                     lx + x0) + 6 * h0 ** 2 * lx * np.sin(lx + x0) - 12 * h1 ** 2 * lx * np.sin(
                     lx + x0) + 6 * h2 ** 2 * lx * np.sin(lx + x0) - 12 * h3 ** 2 * lx * np.sin(
                     lx + x0) + 12 * h0 ** 2 * lx * np.sin(x0) - 6 * h1 ** 2 * lx * np.sin(
                     x0) + 12 * h2 ** 2 * lx * np.sin(x0) - 6 * h3 ** 2 * lx * np.sin(
                     x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(
                     lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.cos(x0) + 3 * h2 ** 2 * lx ** 2 * np.cos(x0)) / (
                         2 * lx ** 3 * lz), ], [(18 * h1 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(
            x0) - 18 * h2 ** 2 * np.cos(x0) + 18 * h3 ** 2 * np.cos(x0) + 18 * h0 ** 2 * np.cos(
            lx + x0) - 18 * h1 ** 2 * np.cos(lx + x0) + 18 * h2 ** 2 * np.cos(lx + x0) - 18 * h3 ** 2 * np.cos(
            lx + x0) + 6 * h0 ** 2 * lx * np.sin(lx + x0) - 12 * h1 ** 2 * lx * np.sin(
            lx + x0) + 6 * h2 ** 2 * lx * np.sin(lx + x0) - 12 * h3 ** 2 * lx * np.sin(
            lx + x0) + 12 * h0 ** 2 * lx * np.sin(x0) - 6 * h1 ** 2 * lx * np.sin(x0) + 12 * h2 ** 2 * lx * np.sin(
            x0) - 6 * h3 ** 2 * lx * np.sin(x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(
            lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.cos(
            x0) + 3 * h2 ** 2 * lx ** 2 * np.cos(x0)) / (2 * lx ** 3 * lz) + (lr ** 2 * lz * (
                3 * h0 ** 2 * np.cos(x0) - 3 * h1 ** 2 * np.cos(x0) + h2 ** 2 * np.cos(x0) - h3 ** 2 * np.cos(
            x0) - 3 * h0 ** 2 * np.cos(lx + x0) + 3 * h1 ** 2 * np.cos(lx + x0) - h2 ** 2 * np.cos(
            lx + x0) + h3 ** 2 * np.cos(lx + x0) + 3 * h1 ** 2 * lx * np.sin(lx + x0) + h3 ** 2 * lx * np.sin(
            lx + x0) - 3 * h0 ** 2 * lx * np.sin(x0) - h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3), - (
                18 * h1 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(x0) - 18 * h2 ** 2 * np.cos(
            x0) + 18 * h3 ** 2 * np.cos(x0) + 18 * h0 ** 2 * np.cos(lx + x0) - 18 * h1 ** 2 * np.cos(
            lx + x0) + 18 * h2 ** 2 * np.cos(lx + x0) - 18 * h3 ** 2 * np.cos(lx + x0) + 12 * h0 ** 2 * lx * np.sin(
            lx + x0) - 18 * h1 ** 2 * lx * np.sin(lx + x0) + 12 * h2 ** 2 * lx * np.sin(
            lx + x0) - 18 * h3 ** 2 * lx * np.sin(lx + x0) + 6 * h0 ** 2 * lx * np.sin(
            x0) + 6 * h2 ** 2 * lx * np.sin(x0) - 3 * h0 ** 2 * lx ** 2 * np.cos(
            lx + x0) + 9 * h1 ** 2 * lx ** 2 * np.cos(lx + x0) - 3 * h2 ** 2 * lx ** 2 * np.cos(
            lx + x0) + 9 * h3 ** 2 * lx ** 2 * np.cos(lx + x0) + 3 * h1 ** 2 * lx ** 3 * np.sin(
            lx + x0) + 3 * h3 ** 2 * lx ** 3 * np.sin(lx + x0)) / (2 * lx ** 3 * lz) - (lr ** 2 * lz * (
                3 * h0 ** 2 * np.cos(x0) - 3 * h1 ** 2 * np.cos(x0) + h2 ** 2 * np.cos(x0) - h3 ** 2 * np.cos(
            x0) - 3 * h0 ** 2 * np.cos(lx + x0) + 3 * h1 ** 2 * np.cos(lx + x0) - h2 ** 2 * np.cos(
            lx + x0) + h3 ** 2 * np.cos(lx + x0) + 3 * h1 ** 2 * lx * np.sin(lx + x0) + h3 ** 2 * lx * np.sin(
            lx + x0) - 3 * h0 ** 2 * lx * np.sin(x0) - h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3), (
                                                        lr ** 2 * lz * (h0 ** 2 * np.cos(x0) - h1 ** 2 * np.cos(
                                                    x0) + h2 ** 2 * np.cos(x0) - h3 ** 2 * np.cos(
                                                    x0) - h0 ** 2 * np.cos(lx + x0) + h1 ** 2 * np.cos(
                                                    lx + x0) - h2 ** 2 * np.cos(lx + x0) + h3 ** 2 * np.cos(
                                                    lx + x0) + h1 ** 2 * lx * np.sin(
                                                    lx + x0) + h3 ** 2 * lx * np.sin(
                                                    lx + x0) - h0 ** 2 * lx * np.sin(
                                                    x0) - h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3) - (
                                                        18 * h1 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(
                                                    x0) - 18 * h2 ** 2 * np.cos(x0) + 18 * h3 ** 2 * np.cos(
                                                    x0) + 18 * h0 ** 2 * np.cos(
                                                    lx + x0) - 18 * h1 ** 2 * np.cos(
                                                    lx + x0) + 18 * h2 ** 2 * np.cos(
                                                    lx + x0) - 18 * h3 ** 2 * np.cos(
                                                    lx + x0) + 6 * h0 ** 2 * lx * np.sin(
                                                    lx + x0) - 12 * h1 ** 2 * lx * np.sin(
                                                    lx + x0) + 6 * h2 ** 2 * lx * np.sin(
                                                    lx + x0) - 12 * h3 ** 2 * lx * np.sin(
                                                    lx + x0) + 12 * h0 ** 2 * lx * np.sin(
                                                    x0) - 6 * h1 ** 2 * lx * np.sin(
                                                    x0) + 12 * h2 ** 2 * lx * np.sin(
                                                    x0) - 6 * h3 ** 2 * lx * np.sin(
                                                    x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(
                                                    lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(
                                                    lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.cos(
                                                    x0) + 3 * h2 ** 2 * lx ** 2 * np.cos(x0)) / (
                                                        2 * lx ** 3 * lz), (
                                                        18 * h1 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(
                                                    x0) - 18 * h2 ** 2 * np.cos(x0) + 18 * h3 ** 2 * np.cos(
                                                    x0) + 18 * h0 ** 2 * np.cos(
                                                    lx + x0) - 18 * h1 ** 2 * np.cos(
                                                    lx + x0) + 18 * h2 ** 2 * np.cos(
                                                    lx + x0) - 18 * h3 ** 2 * np.cos(
                                                    lx + x0) + 12 * h0 ** 2 * lx * np.sin(
                                                    lx + x0) - 18 * h1 ** 2 * lx * np.sin(
                                                    lx + x0) + 12 * h2 ** 2 * lx * np.sin(
                                                    lx + x0) - 18 * h3 ** 2 * lx * np.sin(
                                                    lx + x0) + 6 * h0 ** 2 * lx * np.sin(
                                                    x0) + 6 * h2 ** 2 * lx * np.sin(
                                                    x0) - 3 * h0 ** 2 * lx ** 2 * np.cos(
                                                    lx + x0) + 9 * h1 ** 2 * lx ** 2 * np.cos(
                                                    lx + x0) - 3 * h2 ** 2 * lx ** 2 * np.cos(
                                                    lx + x0) + 9 * h3 ** 2 * lx ** 2 * np.cos(
                                                    lx + x0) + 3 * h1 ** 2 * lx ** 3 * np.sin(
                                                    lx + x0) + 3 * h3 ** 2 * lx ** 3 * np.sin(lx + x0)) / (
                                                        2 * lx ** 3 * lz) - (lr ** 2 * lz * (
                h0 ** 2 * np.cos(x0) - h1 ** 2 * np.cos(x0) + h2 ** 2 * np.cos(x0) - h3 ** 2 * np.cos(
            x0) - h0 ** 2 * np.cos(lx + x0) + h1 ** 2 * np.cos(lx + x0) - h2 ** 2 * np.cos(
            lx + x0) + h3 ** 2 * np.cos(lx + x0) + h1 ** 2 * lx * np.sin(lx + x0) + h3 ** 2 * lx * np.sin(
            lx + x0) - h0 ** 2 * lx * np.sin(x0) - h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3), ], [- (
                18 * h0 ** 2 * np.cos(x0) - 18 * h1 ** 2 * np.cos(x0) + 18 * h2 ** 2 * np.cos(
            x0) - 18 * h3 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(lx + x0) + 18 * h1 ** 2 * np.cos(
            lx + x0) - 18 * h2 ** 2 * np.cos(lx + x0) + 18 * h3 ** 2 * np.cos(lx + x0) + 6 * h1 ** 2 * lx * np.sin(
            lx + x0) + 6 * h3 ** 2 * lx * np.sin(lx + x0) - 18 * h0 ** 2 * lx * np.sin(
            x0) + 12 * h1 ** 2 * lx * np.sin(x0) - 18 * h2 ** 2 * lx * np.sin(x0) + 12 * h3 ** 2 * lx * np.sin(
            x0) - 9 * h0 ** 2 * lx ** 2 * np.cos(x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(
            x0) - 9 * h2 ** 2 * lx ** 2 * np.cos(x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(
            x0) + 3 * h0 ** 2 * lx ** 3 * np.sin(x0) + 3 * h2 ** 2 * lx ** 3 * np.sin(x0)) / (2 * lx ** 3 * lz) - (
                                                                                                            lr ** 2 * lz * (
                                                                                                            h0 ** 2 * np.cos(
                                                                                                        x0) - h1 ** 2 * np.cos(
                                                                                                        x0) + h2 ** 2 * np.cos(
                                                                                                        x0) - h3 ** 2 * np.cos(
                                                                                                        x0) - h0 ** 2 * np.cos(
                                                                                                        lx + x0) + h1 ** 2 * np.cos(
                                                                                                        lx + x0) - h2 ** 2 * np.cos(
                                                                                                        lx + x0) + h3 ** 2 * np.cos(
                                                                                                        lx + x0) + h1 ** 2 * lx * np.sin(
                                                                                                        lx + x0) + h3 ** 2 * lx * np.sin(
                                                                                                        lx + x0) - h0 ** 2 * lx * np.sin(
                                                                                                        x0) - h2 ** 2 * lx * np.sin(
                                                                                                        x0))) / (
                                                                                                            4 * lx ** 3),
                                                                                                    (
                                                                                                            lr ** 2 * lz * (
                                                                                                            h0 ** 2 * np.cos(
                                                                                                        x0) - h1 ** 2 * np.cos(
                                                                                                        x0) + h2 ** 2 * np.cos(
                                                                                                        x0) - h3 ** 2 * np.cos(
                                                                                                        x0) - h0 ** 2 * np.cos(
                                                                                                        lx + x0) + h1 ** 2 * np.cos(
                                                                                                        lx + x0) - h2 ** 2 * np.cos(
                                                                                                        lx + x0) + h3 ** 2 * np.cos(
                                                                                                        lx + x0) + h1 ** 2 * lx * np.sin(
                                                                                                        lx + x0) + h3 ** 2 * lx * np.sin(
                                                                                                        lx + x0) - h0 ** 2 * lx * np.sin(
                                                                                                        x0) - h2 ** 2 * lx * np.sin(
                                                                                                        x0))) / (
                                                                                                            4 * lx ** 3) - (
                                                                                                            18 * h1 ** 2 * np.cos(
                                                                                                        x0) - 18 * h0 ** 2 * np.cos(
                                                                                                        x0) - 18 * h2 ** 2 * np.cos(
                                                                                                        x0) + 18 * h3 ** 2 * np.cos(
                                                                                                        x0) + 18 * h0 ** 2 * np.cos(
                                                                                                        lx + x0) - 18 * h1 ** 2 * np.cos(
                                                                                                        lx + x0) + 18 * h2 ** 2 * np.cos(
                                                                                                        lx + x0) - 18 * h3 ** 2 * np.cos(
                                                                                                        lx + x0) + 6 * h0 ** 2 * lx * np.sin(
                                                                                                        lx + x0) - 12 * h1 ** 2 * lx * np.sin(
                                                                                                        lx + x0) + 6 * h2 ** 2 * lx * np.sin(
                                                                                                        lx + x0) - 12 * h3 ** 2 * lx * np.sin(
                                                                                                        lx + x0) + 12 * h0 ** 2 * lx * np.sin(
                                                                                                        x0) - 6 * h1 ** 2 * lx * np.sin(
                                                                                                        x0) + 12 * h2 ** 2 * lx * np.sin(
                                                                                                        x0) - 6 * h3 ** 2 * lx * np.sin(
                                                                                                        x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(
                                                                                                        lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(
                                                                                                        lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.cos(
                                                                                                        x0) + 3 * h2 ** 2 * lx ** 2 * np.cos(
                                                                                                        x0)) / (
                                                                                                            2 * lx ** 3 * lz),
                                                                                                    (
                                                                                                            18 * h0 ** 2 * np.cos(
                                                                                                        x0) - 18 * h1 ** 2 * np.cos(
                                                                                                        x0) + 18 * h2 ** 2 * np.cos(
                                                                                                        x0) - 18 * h3 ** 2 * np.cos(
                                                                                                        x0) - 18 * h0 ** 2 * np.cos(
                                                                                                        lx + x0) + 18 * h1 ** 2 * np.cos(
                                                                                                        lx + x0) - 18 * h2 ** 2 * np.cos(
                                                                                                        lx + x0) + 18 * h3 ** 2 * np.cos(
                                                                                                        lx + x0) + 6 * h1 ** 2 * lx * np.sin(
                                                                                                        lx + x0) + 6 * h3 ** 2 * lx * np.sin(
                                                                                                        lx + x0) - 18 * h0 ** 2 * lx * np.sin(
                                                                                                        x0) + 12 * h1 ** 2 * lx * np.sin(
                                                                                                        x0) - 18 * h2 ** 2 * lx * np.sin(
                                                                                                        x0) + 12 * h3 ** 2 * lx * np.sin(
                                                                                                        x0) - 9 * h0 ** 2 * lx ** 2 * np.cos(
                                                                                                        x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(
                                                                                                        x0) - 9 * h2 ** 2 * lx ** 2 * np.cos(
                                                                                                        x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(
                                                                                                        x0) + 3 * h0 ** 2 * lx ** 3 * np.sin(
                                                                                                        x0) + 3 * h2 ** 2 * lx ** 3 * np.sin(
                                                                                                        x0)) / (
                                                                                                            2 * lx ** 3 * lz) - (
                                                                                                            lr ** 2 * lz * (
                                                                                                            h0 ** 2 * np.cos(
                                                                                                        x0) - h1 ** 2 * np.cos(
                                                                                                        x0) + 3 * h2 ** 2 * np.cos(
                                                                                                        x0) - 3 * h3 ** 2 * np.cos(
                                                                                                        x0) - h0 ** 2 * np.cos(
                                                                                                        lx + x0) + h1 ** 2 * np.cos(
                                                                                                        lx + x0) - 3 * h2 ** 2 * np.cos(
                                                                                                        lx + x0) + 3 * h3 ** 2 * np.cos(
                                                                                                        lx + x0) + h1 ** 2 * lx * np.sin(
                                                                                                        lx + x0) + 3 * h3 ** 2 * lx * np.sin(
                                                                                                        lx + x0) - h0 ** 2 * lx * np.sin(
                                                                                                        x0) - 3 * h2 ** 2 * lx * np.sin(
                                                                                                        x0))) / (
                                                                                                            4 * lx ** 3),
                                                                                                    (
                                                                                                            18 * h1 ** 2 * np.cos(
                                                                                                        x0) - 18 * h0 ** 2 * np.cos(
                                                                                                        x0) - 18 * h2 ** 2 * np.cos(
                                                                                                        x0) + 18 * h3 ** 2 * np.cos(
                                                                                                        x0) + 18 * h0 ** 2 * np.cos(
                                                                                                        lx + x0) - 18 * h1 ** 2 * np.cos(
                                                                                                        lx + x0) + 18 * h2 ** 2 * np.cos(
                                                                                                        lx + x0) - 18 * h3 ** 2 * np.cos(
                                                                                                        lx + x0) + 6 * h0 ** 2 * lx * np.sin(
                                                                                                        lx + x0) - 12 * h1 ** 2 * lx * np.sin(
                                                                                                        lx + x0) + 6 * h2 ** 2 * lx * np.sin(
                                                                                                        lx + x0) - 12 * h3 ** 2 * lx * np.sin(
                                                                                                        lx + x0) + 12 * h0 ** 2 * lx * np.sin(
                                                                                                        x0) - 6 * h1 ** 2 * lx * np.sin(
                                                                                                        x0) + 12 * h2 ** 2 * lx * np.sin(
                                                                                                        x0) - 6 * h3 ** 2 * lx * np.sin(
                                                                                                        x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(
                                                                                                        lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(
                                                                                                        lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.cos(
                                                                                                        x0) + 3 * h2 ** 2 * lx ** 2 * np.cos(
                                                                                                        x0)) / (
                                                                                                            2 * lx ** 3 * lz) + (
                                                                                                            lr ** 2 * lz * (
                                                                                                            h0 ** 2 * np.cos(
                                                                                                        x0) - h1 ** 2 * np.cos(
                                                                                                        x0) + 3 * h2 ** 2 * np.cos(
                                                                                                        x0) - 3 * h3 ** 2 * np.cos(
                                                                                                        x0) - h0 ** 2 * np.cos(
                                                                                                        lx + x0) + h1 ** 2 * np.cos(
                                                                                                        lx + x0) - 3 * h2 ** 2 * np.cos(
                                                                                                        lx + x0) + 3 * h3 ** 2 * np.cos(
                                                                                                        lx + x0) + h1 ** 2 * lx * np.sin(
                                                                                                        lx + x0) + 3 * h3 ** 2 * lx * np.sin(
                                                                                                        lx + x0) - h0 ** 2 * lx * np.sin(
                                                                                                        x0) - 3 * h2 ** 2 * lx * np.sin(
                                                                                                        x0))) / (
                                                                                                            4 * lx ** 3), ],
                [(lr ** 2 * lz * (h0 ** 2 * np.cos(x0) - h1 ** 2 * np.cos(x0) + h2 ** 2 * np.cos(x0) - h3 ** 2 * np.cos(
                    x0) - h0 ** 2 * np.cos(lx + x0) + h1 ** 2 * np.cos(lx + x0) - h2 ** 2 * np.cos(
                    lx + x0) + h3 ** 2 * np.cos(lx + x0) + h1 ** 2 * lx * np.sin(lx + x0) + h3 ** 2 * lx * np.sin(
                    lx + x0) - h0 ** 2 * lx * np.sin(x0) - h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3) - (
                         18 * h1 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(x0) - 18 * h2 ** 2 * np.cos(
                     x0) + 18 * h3 ** 2 * np.cos(x0) + 18 * h0 ** 2 * np.cos(lx + x0) - 18 * h1 ** 2 * np.cos(
                     lx + x0) + 18 * h2 ** 2 * np.cos(lx + x0) - 18 * h3 ** 2 * np.cos(
                     lx + x0) + 6 * h0 ** 2 * lx * np.sin(lx + x0) - 12 * h1 ** 2 * lx * np.sin(
                     lx + x0) + 6 * h2 ** 2 * lx * np.sin(lx + x0) - 12 * h3 ** 2 * lx * np.sin(
                     lx + x0) + 12 * h0 ** 2 * lx * np.sin(x0) - 6 * h1 ** 2 * lx * np.sin(
                     x0) + 12 * h2 ** 2 * lx * np.sin(x0) - 6 * h3 ** 2 * lx * np.sin(
                     x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(
                     lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.cos(x0) + 3 * h2 ** 2 * lx ** 2 * np.cos(x0)) / (
                         2 * lx ** 3 * lz), (
                         18 * h1 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(x0) - 18 * h2 ** 2 * np.cos(
                     x0) + 18 * h3 ** 2 * np.cos(x0) + 18 * h0 ** 2 * np.cos(lx + x0) - 18 * h1 ** 2 * np.cos(
                     lx + x0) + 18 * h2 ** 2 * np.cos(lx + x0) - 18 * h3 ** 2 * np.cos(
                     lx + x0) + 12 * h0 ** 2 * lx * np.sin(lx + x0) - 18 * h1 ** 2 * lx * np.sin(
                     lx + x0) + 12 * h2 ** 2 * lx * np.sin(lx + x0) - 18 * h3 ** 2 * lx * np.sin(
                     lx + x0) + 6 * h0 ** 2 * lx * np.sin(x0) + 6 * h2 ** 2 * lx * np.sin(
                     x0) - 3 * h0 ** 2 * lx ** 2 * np.cos(lx + x0) + 9 * h1 ** 2 * lx ** 2 * np.cos(
                     lx + x0) - 3 * h2 ** 2 * lx ** 2 * np.cos(lx + x0) + 9 * h3 ** 2 * lx ** 2 * np.cos(
                     lx + x0) + 3 * h1 ** 2 * lx ** 3 * np.sin(lx + x0) + 3 * h3 ** 2 * lx ** 3 * np.sin(
                     lx + x0)) / (2 * lx ** 3 * lz) - (lr ** 2 * lz * (
                        h0 ** 2 * np.cos(x0) - h1 ** 2 * np.cos(x0) + h2 ** 2 * np.cos(x0) - h3 ** 2 * np.cos(
                    x0) - h0 ** 2 * np.cos(lx + x0) + h1 ** 2 * np.cos(lx + x0) - h2 ** 2 * np.cos(
                    lx + x0) + h3 ** 2 * np.cos(lx + x0) + h1 ** 2 * lx * np.sin(lx + x0) + h3 ** 2 * lx * np.sin(
                    lx + x0) - h0 ** 2 * lx * np.sin(x0) - h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3), (
                         18 * h1 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(x0) - 18 * h2 ** 2 * np.cos(
                     x0) + 18 * h3 ** 2 * np.cos(x0) + 18 * h0 ** 2 * np.cos(lx + x0) - 18 * h1 ** 2 * np.cos(
                     lx + x0) + 18 * h2 ** 2 * np.cos(lx + x0) - 18 * h3 ** 2 * np.cos(
                     lx + x0) + 6 * h0 ** 2 * lx * np.sin(lx + x0) - 12 * h1 ** 2 * lx * np.sin(
                     lx + x0) + 6 * h2 ** 2 * lx * np.sin(lx + x0) - 12 * h3 ** 2 * lx * np.sin(
                     lx + x0) + 12 * h0 ** 2 * lx * np.sin(x0) - 6 * h1 ** 2 * lx * np.sin(
                     x0) + 12 * h2 ** 2 * lx * np.sin(x0) - 6 * h3 ** 2 * lx * np.sin(
                     x0) + 3 * h1 ** 2 * lx ** 2 * np.cos(lx + x0) + 3 * h3 ** 2 * lx ** 2 * np.cos(
                     lx + x0) + 3 * h0 ** 2 * lx ** 2 * np.cos(x0) + 3 * h2 ** 2 * lx ** 2 * np.cos(x0)) / (
                         2 * lx ** 3 * lz) + (lr ** 2 * lz * (
                        h0 ** 2 * np.cos(x0) - h1 ** 2 * np.cos(x0) + 3 * h2 ** 2 * np.cos(
                    x0) - 3 * h3 ** 2 * np.cos(x0) - h0 ** 2 * np.cos(lx + x0) + h1 ** 2 * np.cos(
                    lx + x0) - 3 * h2 ** 2 * np.cos(lx + x0) + 3 * h3 ** 2 * np.cos(
                    lx + x0) + h1 ** 2 * lx * np.sin(lx + x0) + 3 * h3 ** 2 * lx * np.sin(
                    lx + x0) - h0 ** 2 * lx * np.sin(x0) - 3 * h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3), - (
                        18 * h1 ** 2 * np.cos(x0) - 18 * h0 ** 2 * np.cos(x0) - 18 * h2 ** 2 * np.cos(
                    x0) + 18 * h3 ** 2 * np.cos(x0) + 18 * h0 ** 2 * np.cos(lx + x0) - 18 * h1 ** 2 * np.cos(
                    lx + x0) + 18 * h2 ** 2 * np.cos(lx + x0) - 18 * h3 ** 2 * np.cos(
                    lx + x0) + 12 * h0 ** 2 * lx * np.sin(lx + x0) - 18 * h1 ** 2 * lx * np.sin(
                    lx + x0) + 12 * h2 ** 2 * lx * np.sin(lx + x0) - 18 * h3 ** 2 * lx * np.sin(
                    lx + x0) + 6 * h0 ** 2 * lx * np.sin(x0) + 6 * h2 ** 2 * lx * np.sin(
                    x0) - 3 * h0 ** 2 * lx ** 2 * np.cos(lx + x0) + 9 * h1 ** 2 * lx ** 2 * np.cos(
                    lx + x0) - 3 * h2 ** 2 * lx ** 2 * np.cos(lx + x0) + 9 * h3 ** 2 * lx ** 2 * np.cos(
                    lx + x0) + 3 * h1 ** 2 * lx ** 3 * np.sin(lx + x0) + 3 * h3 ** 2 * lx ** 3 * np.sin(
                    lx + x0)) / (2 * lx ** 3 * lz) - (lr ** 2 * lz * (
                        h0 ** 2 * np.cos(x0) - h1 ** 2 * np.cos(x0) + 3 * h2 ** 2 * np.cos(
                    x0) - 3 * h3 ** 2 * np.cos(x0) - h0 ** 2 * np.cos(lx + x0) + h1 ** 2 * np.cos(
                    lx + x0) - 3 * h2 ** 2 * np.cos(lx + x0) + 3 * h3 ** 2 * np.cos(
                    lx + x0) + h1 ** 2 * lx * np.sin(lx + x0) + 3 * h3 ** 2 * lx * np.sin(
                    lx + x0) - h0 ** 2 * lx * np.sin(x0) - 3 * h2 ** 2 * lx * np.sin(x0))) / (4 * lx ** 3), ], ]
        self.ad = np.array(adyc)

    def cal_el_bdyc(self):
        x0 = self.nodes[0].coords[0]
        lx = self.lx
        lz = self.lz
        vx = self.vx
        bdyc = [(lz * vx * (np.sin(lx + x0) - np.sin(x0))) / (2 * lx),
                -(lz * vx * (np.sin(lx + x0) - np.sin(x0))) / (2 * lx),
                (lz * vx * (np.sin(lx + x0) - np.sin(x0))) / (2 * lx),
                -(lz * vx * (np.sin(lx + x0) - np.sin(x0))) / (2 * lx), ]
        self.bd = np.array(bdyc)

    def cal_el_bdxct(self):
        x0 = self.nodes[0].coords[0]
        lx = self.lx
        lz = self.lz
        self.bd = [(lz * (np.sin(lx + x0) - np.sin(x0))) / (2 * lx) - (lz * np.cos(x0)) / 2,
                   (lz * np.cos(lx + x0)) / 2 - (lz * (np.sin(lx + x0) - np.sin(x0))) / (2 * lx),
                   (lz * (np.sin(lx + x0) - np.sin(x0))) / (2 * lx) - (lz * np.cos(x0)) / 2,
                   (lz * np.cos(lx + x0)) / 2 - (lz * (np.sin(lx + x0) - np.sin(x0))) / (2 * lx), ]
        self.bd = np.array(self.bd)

    def cal_el_bdyct(self):
        x0 = self.nodes[0].coords[0]
        lx = self.lx
        lz = self.lz
        self.bd = [- (lz * np.sin(x0)) / 2 - (lz * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx),
                   (lz * np.sin(lx + x0)) / 2 + (lz * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx),
                   - (lz * np.sin(x0)) / 2 - (lz * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx),
                   (lz * np.sin(lx + x0)) / 2 + (lz * (np.cos(lx + x0) - np.cos(x0))) / (2 * lx), ]
        self.bd = np.array(self.bd)

    def cal_keyct(self):
        pass

    def cal_feyct(self):
        pass
