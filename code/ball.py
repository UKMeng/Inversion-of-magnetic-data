import math
import numpy as np
# 全局参数
myu = 4 * math.pi * 1e-7
const = 1e-7
I = math.radians(75)
D = math.radians(25)
T = 50000

# 定义球体类型
class ball(object):
    def __init__(self, xc = 0, yc = 0, zc = 0, R = 0, k = 0):
        self.xc = xc
        self.yc = yc
        self.zc = zc
        self.R = R
        self.k = k
        self.M = k * T * (4 * math.pi * R * R * R / 3) / myu 

def deltaT(z, hx = 0, hy = 0):
    return hx * math.cos(I) * math.cos(D) + hy * math.cos(I) * math.sin(D) + z * math.sin(I)

def h(x, y, z, M):
    return const * M * ((2 * (x ** 2) - (y ** 2) - (z ** 2)) * \
        math.cos(I) * math.sin(D) - 3 * z * x * math.sin(I) +\
            3 * x * y * math.cos(I) * math.sin(D)) / (((x ** 2) \
                + (y ** 2) + (z ** 2)) ** 2.5)

def Za(x, y, z, M):
    return const * M * ((2 * (z ** 2) - (x ** 2) - (y ** 2)) * \
        math.sin(I) - 3 * z * x * math.cos(I) * math.cos(D) \
            - 3 * z * y * math.cos(I) * math.sin(D)) / (((x ** 2) + \
                (y ** 2) + (z ** 2)) ** 2.5)

def cal(x0, y0, m):
    x = x0 - m.xc
    y = y0 - m.yc
    z = m.zc
    M = m.M
    hax, hay, za = h(x, y, z, M), h(y, x, z, M), Za(x, y, z, M)
    return deltaT(za, hax, hay) 

def core(m, X, Y):
    Z = np.asarray(X, dtype= None, order= None)
    for x in range(0, Z.shape[0]):
        for y in range(0, Z.shape[1]):
            Z[x, y] = cal(Y[x, y], X[x, y], m)
    return Z