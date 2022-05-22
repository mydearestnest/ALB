import numpy as np
from base import Node, System, Elem
import sympy as sp
from matplotlib import pyplot as plt
from func import load_input, init_input


def main(filename='input'):
    init_args = init_input(filename)
    LN_sys = System(init_args)
    LN_sys.creat_rect_nodes()
    LN_sys.crear_rect_elems()
    LN_sys.cal_h()
    LN_sys.cal_kf()
    LN_sys.set_bondary()
    LN_sys.cal_p_direct()
    LN_sys.plot_p_3d()
    save_output(LN_sys.p)


if __name__ == '__main__':
    main()
