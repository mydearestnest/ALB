import numpy as np
from base import Node,System,Elem
import sympy as sp
from matplotlib import pyplot as plt
NX = np.int64(100);NZ = np.int64(10);e=0;vx = 1;lr = 1
LX = np.float64(2*np.pi)
LZ = np.float64(1)
lx = LX / NX
lz = LZ / NZ
LN_sys = System(NX,NZ,lx,lz,vx,lr,e)
LN_sys.creat_rect_nodes()
LN_sys.crear_rect_elems()
# LN_sys.cal_h()
#设置薄膜厚度
hz = np.empty([NX+1,NZ+1])
for i in range(NX+1):
    for j in range(NZ+1):
        hz[i][j] = LN_sys.nodes[i+j*(NX+1)].h = 1+e*np.cos(i*lx)
LN_sys.h = hz
LN_sys.cal_k()
LN_sys.set_bondary()
LN_sys.cal_p()
LN_sys.plot_p_3d()
#计算单元刚度
# ##通过符号运算获得关于h1,h2,h3,h4的刚度矩阵
# x,z=sp.symbols(['x','z'])
# h0,h1,h2,h3 = sp.symbols(['h1','h2','h3','h4'])
# f0 = (1-x/lx)*(1-z/lz)
# f1 = x/lx*(1-z/lz)
# f2 = z/lz*(1-x/lz)
# f3 = x/lx*z/lz
# h = h0*f0+h1*f1+h2*f2+h3*f3
# ke = []
# f = [f0,f1,f2,f3]
# for n in range(4):
#     ke1 = []
#     for m in range(4):
#         p1 = h ** 3 * (sp.diff(f[n], x) * sp.diff(f[m], x) + sp.diff(f[n], z) * sp.diff(f[m], z))
#         ke1.append(sp.integrate(sp.integrate(p1,(x,0,lx)),(z,0,lz)))
#     ke.append(ke1)
#
# ##计算每个单元的刚度矩阵
# for el in LN_sys.elems:
#     elem = LN_sys.elems[el]
#     for n in range(4):
#         for m in range(4):
#             elem.ke[n,m] = ke[n][m].evalf(subs={h0:elem.nodes[0].h,h1:elem.nodes[1].h,h2:elem.nodes[2].h,h3:elem.nodes[3].h})
#             LN_sys.k[elem.nodes[n].number][elem.nodes[m].number] += elem.ke[n,m]  ##计算总刚度矩阵
#
#
