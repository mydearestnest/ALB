import numpy as np
import numba

a1 = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]

var = a1[1][:]
print(var)
var1 = a1[1:2][:]
print(var1)

a2 = np.array(a1)
var3 = a2[1, :]
var4 = a2[1:2, :]
print(var3, var3.shape)
print(var4, var4.shape)

a3 = np.zeros((3, 4))
print(a3)

a4 = np.ones((3, 4))
print(a4)

a5 = np.empty_like(a4)
print(a5)

a6 = np.shape(a5)
print(a6)


