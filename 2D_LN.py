import numpy as np
from base import Node,System,Elem
import sympy as sp
from matplotlib import pyplot as plt
from func import load_input,save_output

def main(filename='input.txt'):
    input_data = load_input(filename)
    NX = np.int_(input_data[0]);NZ = np.int_(input_data[1]);e = input_data[2]
    vx = input_data[3];lr = input_data[4];LX = input_data[5];LZ = input_data[6]
    LN_sys = System(NX,NZ,LX,LZ,vx,lr,e)
    LN_sys.creat_rect_nodes()
    LN_sys.crear_rect_elems()
    LN_sys.cal_h()
    LN_sys.cal_k()
    LN_sys.set_bondary()
    LN_sys.cal_p()
    LN_sys.plot_p_3d()
    save_output(LN_sys.p)
if __name__ == '__main__':
    main()