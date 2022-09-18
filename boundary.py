import numpy as np
import scipy.sparse as sp


def reynold_boundary(system):
    p_set = system.pr / system.ps
    for i in range(len(system.p_result)):
        if system.p_result[i] < p_set:
            system.p_result[i] = p_set


# 设置连续/周期边界条件
# nodes1与nodes2需要传入对应节点编号，且需要一一对应
# 返回拉格朗日乘子阵及其非齐次项
def set_continuity_boundary(freedoms, nodes1, nodes2):
    if len(nodes1) == len(nodes2):
        q_nodes = freedoms
        r_nodes = len(nodes1)
        kp3 = np.zeros((r_nodes, q_nodes), dtype=np.float32)
        fp3 = np.zeros(r_nodes)
        for i in range(r_nodes):
            kp3[i, nodes1[i]] = 1  # 附加矩阵内，左侧边界置1
            kp3[i, nodes2[i]] = -1  # 附加矩阵内，右侧边界置-1
        return kp3, fp3
    else:
        print("错误！连续边界两侧节点数量需一致")
        return False



# 设置定压力边界条件
# nodes传入为节点编号
# 返回拉格朗日乘子阵及其非齐次项
def set_pressure_boundary(freedoms, nodes, p_set=0.5):
    q_nodes = freedoms
    l_nodes = len(nodes)
    kp1 = np.zeros((l_nodes, q_nodes), dtype=np.float32)
    fp1 = np.zeros(l_nodes)
    for i in range(l_nodes):
        kp1[i, nodes[i]] = 1
        fp1[i] = p_set
    return kp1, fp1

