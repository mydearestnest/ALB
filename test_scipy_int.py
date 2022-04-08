from scipy import integrate



def f(x, z,lx):
    def dzfk(x, z, lx, lz):
        dzf1 = (1 - x / lx) / lz
        return dzf1
    h1 = 1
    h2 = 1
    h3 = 1
    h0 = 1
    lr = 1
    lz = 1
    f0 = (1 - x / lx) * (1 - z / lz)  # 局部节点0形函数
    f1 = x / lx * (1 - z / lz)  # 局部节点1形函数
    f2 = z / lz * (1 - x / lx)  # 局部节点2形函数
    f3 = x / lx * z / lz  # 局部节点3形函数
    dxf0 = - (1 - z / lz) / lx
    dxf1 = (1 - z / lz) / lx
    dxf2 = -z / lz / lx
    dxf3 = z / lz / lx
    dzf0 = - (1 - x / lx) / lz
    dzf1 = dzfk(x,z,lx,lz)
    dzf2 = -x / lz / lx
    dzf3 = x / lz / lx
    h = h0 * f0 + h1 * f1 + h2 * f2 + h3 * f3  # 局部膜厚插值函数
    p1 = h ** 3 * (lr ** 2 * dxf1 * dxf1 + dzf1 * dzf1)
    return p1


def x_boundary():
    return [0, 1]


def y_boundary():
    return [0, 1]


ans = integrate.nquad(f, [x_boundary(), y_boundary()],args=([1]))
