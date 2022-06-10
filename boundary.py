import numpy as np
from base import System


def reynold_boundary(system):
    for i in range(len(system.p_result)):
        if system.p_result[i] < 0:
            system.p_result[i] = 0
