import math
import numpy as np
import sympy as sp
import multiprocessing as mp
import tqdm
# 全局参数
myu = 4 * math.pi * 1e-7
const = 1e-7
I = math.radians(75)
D = math.radians(25)
I0 = math.radians(75)
D0 = math.radians(25)
k1 = sp.cos(I0)*sp.sin(D0)*sp.sin(I) + sp.sin(I0)*sp.cos(I)*sp.sin(D)
k2 = sp.cos(I0)*sp.cos(D0)*sp.sin(I) + sp.sin(I0)*sp.cos(I)*sp.cos(D)
k3 = sp.cos(I0)*sp.cos(D0)*sp.cos(I)*sp.sin(D) + sp.cos(I0)*sp.sin(D0)*sp.cos(I)*sp.cos(D)
k4 = sp.cos(I0)*sp.cos(D0)*sp.cos(I)*sp.cos(D)
k5 = sp.cos(I0)*sp.sin(D0)*sp.cos(I)*sp.sin(D)
k6 = -sp.sin(I0)*sp.sin(I)
T = 50000

# 定义长方体
class cuboid(object):
    def __init__(self, xc = 0, yc = 0, zc = 0, a = 0, b = 0, c = 0, k = 0):
        self.xc = xc
        self.yc = yc
        self.zc = zc
        self.a = a
        self.b = b
        self.c = c
        self.k = k

def mag(k):
    m = k * T / myu
    return m * sp.cos(I) * sp.cos(D), m * sp.cos(I) * sp.sin(D), m * sp.sin(I) 

def deltaT(z, hx = 0, hy = 0):
    #return sp.sqrt(hx ** 2 + z ** 2 + hy ** 2)
    return hx * sp.cos(I) * sp.cos(D) + hy * sp.cos(I) * sp.sin(D) + z * sp.sin(I)

def hx(x0, y0, Mx, My, Mz, x1, x2, y1, y2, z1, z2):
    '''
    x, y, z = sp.symbols('x y z')
    r = sp.sqrt((x - x0) ** 2 + (y - y0) ** 2 + z ** 2)
    f = const * (Mx * sp.atan(((x - x0) * (y - y0))/((x - x0)**2 + r * z + z ** 2)) + My * sp.log(r + z) + Mz * sp.log((r + (y - y0))))
    f1 = f.subs(z, z1) - f.subs(z, z2)
    f2 = f1.subs(y, y1) - f1.subs(y, y2)
    f3 = f2.subs(x, x1) - f2.subs(x, x2)
    '''
    f = const * (Mx*sp.atan((-x0 + x1)*(-y0 + y1)/(z1**2 + z1*sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y1)**2) + (-x0 + x1)**2)) - Mx*sp.atan((-x0 + x1)*(-y0 + y1)/(z2**2 + z2*sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y1)**2) + (-x0 + x1)**2)) - Mx*sp.atan((-x0 + x1)*(-y0 + y2)/(z1**2 + z1*sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y2)**2) + (-x0 + x1)**2)) + Mx*sp.atan((-x0 + x1)*(-y0 + y2)/(z2**2 + z2*sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y2)**2) + (-x0 + x1)**2)) - Mx*sp.atan((-x0 + x2)*(-y0 + y1)/(z1**2 + z1*sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y1)**2) + (-x0 + x2)**2)) + Mx*sp.atan((-x0 + x2)*(-y0 + y1)/(z2**2 + z2*sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y1)**2) + (-x0 + x2)**2)) + Mx*sp.atan((-x0 + x2)*(-y0 + y2)/(z1**2 + z1*sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y2)**2) + (-x0 + x2)**2)) - Mx*sp.atan((-x0 + x2)*(-y0 + y2)/(z2**2 + z2*sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y2)**2) + (-x0 + x2)**2)) + My*sp.log(z1 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) - My*sp.log(z1 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) - My*sp.log(z1 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) + My*sp.log(z1 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y2)**2)) - My*sp.log(z2 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) + My*sp.log(z2 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) + My*sp.log(z2 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) - My*sp.log(z2 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y2)**2)) + Mz*sp.log(-y0 + y1 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) - Mz*sp.log(-y0 + y1 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) - Mz*sp.log(-y0 + y1 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) + Mz*sp.log(-y0 + y1 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) - Mz*sp.log(-y0 + y2 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) + Mz*sp.log(-y0 + y2 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y2)**2)) + Mz*sp.log(-y0 + y2 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) - Mz*sp.log(-y0 + y2 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y2)**2)))
    #print(x0, y0, f3)
    return f

def hy(x0, y0, Mx, My, Mz, x1, x2, y1, y2, z1, z2):
    '''
    x, y, z = sp.symbols('x y z')
    r = sp.sqrt((x - x0) ** 2 + (y - y0) ** 2 + z ** 2)
    f = const * (My * sp.atan(((x - x0) * (y - y0))/((y - y0)**2 + r * z + z ** 2)) + Mx * sp.log(r + z) + Mz * sp.log((r + (x - x0))))
    f1 = f.subs(z, z1) - f.subs(z, z2)
    f2 = f1.subs(y, y1) - f1.subs(y, y2)
    f3 = f2.subs(x, x1) - f2.subs(x, x2)
    '''
    f = const * (Mx*sp.log(z1 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) - Mx*sp.log(z1 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) - Mx*sp.log(z1 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) + Mx*sp.log(z1 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y2)**2)) - Mx*sp.log(z2 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) + Mx*sp.log(z2 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) + Mx*sp.log(z2 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) - Mx*sp.log(z2 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y2)**2)) + My*sp.atan((-x0 + x1)*(-y0 + y1)/(z1**2 + z1*sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y1)**2) + (-y0 + y1)**2)) - My*sp.atan((-x0 + x1)*(-y0 + y1)/(z2**2 + z2*sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y1)**2) + (-y0 + y1)**2)) - My*sp.atan((-x0 + x1)*(-y0 + y2)/(z1**2 + z1*sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y2)**2) + (-y0 + y2)**2)) + My*sp.atan((-x0 + x1)*(-y0 + y2)/(z2**2 + z2*sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y2)**2) + (-y0 + y2)**2)) - My*sp.atan((-x0 + x2)*(-y0 + y1)/(z1**2 + z1*sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y1)**2) + (-y0 + y1)**2)) + My*sp.atan((-x0 + x2)*(-y0 + y1)/(z2**2 + z2*sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y1)**2) + (-y0 + y1)**2)) + My*sp.atan((-x0 + x2)*(-y0 + y2)/(z1**2 + z1*sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y2)**2) + (-y0 + y2)**2)) - My*sp.atan((-x0 + x2)*(-y0 + y2)/(z2**2 + z2*sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y2)**2) + (-y0 + y2)**2)) + Mz*sp.log(-x0 + x1 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) - Mz*sp.log(-x0 + x1 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) - Mz*sp.log(-x0 + x1 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) + Mz*sp.log(-x0 + x1 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) - Mz*sp.log(-x0 + x2 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) + Mz*sp.log(-x0 + x2 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y2)**2)) + Mz*sp.log(-x0 + x2 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) - Mz*sp.log(-x0 + x2 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y2)**2)))
    #print(x0, y0, f3)
    return f

def z(x0, y0, Mx, My, Mz, x1, x2, y1, y2, z1, z2):
    '''
    x, y, z = sp.symbols('x y z')
    r = sp.sqrt((x - x0) ** 2 + (y - y0) ** 2 + z ** 2)
    f = const * (-Mz * sp.atan(((x - x0) * (y - y0))/(r * z)) + Mx * sp.log(r + (y - y0)) + My * sp.log(r + (x - x0)))
    f1 = f.subs(z, z1) - f.subs(z, z2)
    f2 = f1.subs(y, y1) - f1.subs(y, y2)
    f3 = f2.subs(x, x1) - f2.subs(x, x2)
    '''
    f = const * (Mx*sp.log(-y0 + y1 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) - Mx*sp.log(-y0 + y1 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) - Mx*sp.log(-y0 + y1 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) + Mx*sp.log(-y0 + y1 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) - Mx*sp.log(-y0 + y2 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) + Mx*sp.log(-y0 + y2 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y2)**2)) + Mx*sp.log(-y0 + y2 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) - Mx*sp.log(-y0 + y2 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y2)**2)) + My*sp.log(-x0 + x1 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) - My*sp.log(-x0 + x1 + sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) - My*sp.log(-x0 + x1 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y1)**2)) + My*sp.log(-x0 + x1 + sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y2)**2)) - My*sp.log(-x0 + x2 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) + My*sp.log(-x0 + x2 + sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y2)**2)) + My*sp.log(-x0 + x2 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y1)**2)) - My*sp.log(-x0 + x2 + sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y2)**2)) - Mz*sp.atan((-x0 + x1)*(-y0 + y1)/(z1*sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y1)**2))) + Mz*sp.atan((-x0 + x1)*(-y0 + y2)/(z1*sp.sqrt(z1**2 + (-x0 + x1)**2 + (-y0 + y2)**2))) + Mz*sp.atan((-x0 + x2)*(-y0 + y1)/(z1*sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y1)**2))) - Mz*sp.atan((-x0 + x2)*(-y0 + y2)/(z1*sp.sqrt(z1**2 + (-x0 + x2)**2 + (-y0 + y2)**2))) + Mz*sp.atan((-x0 + x1)*(-y0 + y1)/(z2*sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y1)**2))) - Mz*sp.atan((-x0 + x1)*(-y0 + y2)/(z2*sp.sqrt(z2**2 + (-x0 + x1)**2 + (-y0 + y2)**2))) - Mz*sp.atan((-x0 + x2)*(-y0 + y1)/(z2*sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y1)**2))) + Mz*sp.atan((-x0 + x2)*(-y0 + y2)/(z2*sp.sqrt(z2**2 + (-x0 + x2)**2 + (-y0 + y2)**2))))
    #print(x0, y0, f3)
    return f
'''
def hy(x, y, x0, y0, z0, Mx, My, Mz, r):
    return const * (-My * math.sp.atan(((x0 - x) * (y0 - y))/((y0 - y)**2 + r * z0 + z0 ** 2)) + Mx * math.sp.log(r + z0) + Mz * math.sp.log((r + (x0 - x))))

def z(x, y, x0, y0, z0, Mx, My, Mz, r):
    return const * (Mz * math.sp.atan(((x0 - x) * (y0 - y))/(r * z0)) + Mx * math.sp.log((r + (y0 - y))) + My * math.sp.log((r + (x0 - x))))
'''
def kernel(params):
    x0 = params[0]
    y0 = params[1]
    m = params[2]
    # xp = m.xc - x0
    # yp = m.xc - y0
    x1 = m.xc + m.a/2 - x0
    x2 = m.xc - m.a/2 - x0
    y1 = m.yc + m.b/2 - y0
    y2 = m.yc - m.b/2 - y0
    z1 = m.zc + m.c/2
    z2 = m.zc - m.c/2
    # r = sp.sqrt(xp**2 + yp ** 2 + m.zc ** 2)
    # print(x0, y0, x1, x2, y1, y2, z1, z2)
    # print(k1*sp.log(x2 + sp.sqrt(x2**2 + y2**2 + z2**2)))
    '''
    x, y, z = sp.symbols('x y z')
    r = sp.sqrt(x**2 + y**2 + z**2)
    g = -const * (k1 * sp.log(x + r) + k2 * sp.log(y + r) + k3 * sp.log(z + r) + k4 * sp.atan(x*y/(x**2 + r * z + z ** 2)) + k5 * sp.atan(x*y/(y ** 2 + r * z + z ** 2))+ k6 * sp.atan(x * y / (r * z)))
    '''
    
    g = const * (-k1*sp.log(x1 + sp.sqrt(x1**2 + y1**2 + z1**2)) + k1*sp.log(x1 + sp.sqrt(x1**2 + y1**2 + z2**2)) + k1*sp.log(x1 + sp.sqrt(x1**2 + y2**2 + z1**2)) - k1*sp.log(x1 + sp.sqrt(x1**2 + y2**2 + z2**2)) + k1*sp.log(x2 + sp.sqrt(x2**2 + y1**2 + z1**2)) - k1*sp.log(x2 + sp.sqrt(x2**2 + y1**2 + z2**2)) - k1*sp.log(x2 + sp.sqrt(x2**2 + y2**2 + z1**2)) + k1*sp.log(x2 + sp.sqrt(x2**2 + y2**2 + z2**2)) - k2*sp.log(y1 + sp.sqrt(x1**2 + y1**2 + z1**2)) + k2*sp.log(y1 + sp.sqrt(x1**2 + y1**2 + z2**2)) + k2*sp.log(y1 + sp.sqrt(x2**2 + y1**2 + z1**2)) - k2*sp.log(y1 + sp.sqrt(x2**2 + y1**2 + z2**2)) + k2*sp.log(y2 + sp.sqrt(x1**2 + y2**2 + z1**2)) - k2*sp.log(y2 + sp.sqrt(x1**2 + y2**2 + z2**2)) - k2*sp.log(y2 + sp.sqrt(x2**2 + y2**2 + z1**2)) + k2*sp.log(y2 + sp.sqrt(x2**2 + y2**2 + z2**2)) - k3*sp.log(z1 + sp.sqrt(x1**2 + y1**2 + z1**2)) + \
        k3*sp.log(z1 + sp.sqrt(x1**2 + y2**2 + z1**2)) + k3*sp.log(z1 + sp.sqrt(x2**2 + y1**2 + z1**2)) - k3*sp.log(z1 + sp.sqrt(x2**2 + y2**2 + z1**2)) + k3*sp.log(z2 + sp.sqrt(x1**2 + y1**2 + z2**2)) - k3*sp.log(z2 + sp.sqrt(x1**2 + y2**2 + z2**2)) - k3*sp.log(z2 + sp.sqrt(x2**2 + y1**2 + z2**2)) + k3*sp.log(z2 + sp.sqrt(x2**2 + y2**2 + z2**2)) - k4*sp.atan(x1*y1/(x1**2 + z1**2 + z1*sp.sqrt(x1**2 + y1**2 + z1**2))) + k4*sp.atan(x1*y1/(x1**2 + z2**2 + z2*sp.sqrt(x1**2 + y1**2 + z2**2))) + k4*sp.atan(x1*y2/(x1**2 + z1**2 + z1*sp.sqrt(x1**2 + y2**2 + z1**2))) - k4*sp.atan(x1*y2/(x1**2 + z2**2 + z2*sp.sqrt(x1**2 + y2**2 + z2**2))) + k4*sp.atan(x2*y1/(x2**2 + z1**2 + z1*sp.sqrt(x2**2 + y1**2 + z1**2))) - k4*sp.atan(x2*y1/(x2**2 + z2**2 + z2*sp.sqrt(x2**2 + y1**2 + z2**2))) - k4*sp.atan(x2*y2/(x2**2 + z1**2 + z1*sp.sqrt(x2**2 + y2**2 + z1**2))) + k4*sp.atan(x2*y2/(x2**2 + z2**2 + z2*sp.sqrt(x2**2 + y2**2 + z2**2))) - k5*sp.atan(x1*y1/(y1**2 + z1**2 + z1*sp.sqrt(x1**2 + y1**2 + z1**2))) + k5*sp.atan(x1*y1/(y1**2 + z2**2 + z2*sp.sqrt(x1**2 + y1**2 + z2**2))) + k5*sp.atan(x1*y2/(y2**2 + z1**2 + z1*sp.sqrt(x1**2 + y2**2 + z1**2))) - k5*sp.atan(x1*y2/(y2**2 + z2**2 + z2*sp.sqrt(x1**2 + y2**2 + z2**2))) + k5*sp.atan(x2*y1/(y1**2 + z1**2 + z1*sp.sqrt(x2**2 + y1**2 + z1**2))) - k5*sp.atan(x2*y1/(y1**2 + z2**2 + z2*sp.sqrt(x2**2 + y1**2 + z2**2))) - k5*sp.atan(x2*y2/(y2**2 + z1**2 + z1*sp.sqrt(x2**2 + y2**2 + z1**2))) + k5*sp.atan(x2*y2/(y2**2 + z2**2 + z2*sp.sqrt(x2**2 + y2**2 + z2**2))) - k6*sp.atan(x1*y1/(z1*sp.sqrt(x1**2 + y1**2 + z1**2))) + k6*sp.atan(x1*y1/(z2*sp.sqrt(x1**2 + y1**2 + z2**2))) + k6*sp.atan(x1*y2/(z1*sp.sqrt(x1**2 + y2**2 + z1**2))) - k6*sp.atan(x1*y2/(z2*sp.sqrt(x1**2 + y2**2 + z2**2))) + k6*sp.atan(x2*y1/(z1*sp.sqrt(x2**2 + y1**2 + z1**2))) - k6*sp.atan(x2*y1/(z2*sp.sqrt(x2**2 + y1**2 + z2**2))) - k6*sp.atan(x2*y2/(z1*sp.sqrt(x2**2 + y2**2 + z1**2))) + k6*sp.atan(x2*y2/(z2*sp.sqrt(x2**2 + y2**2 + z2**2))))

    '''
    g = const * (-k4*sp.atan(x1*y1/(r*z1 + x1**2 + z1**2)) + k4*sp.atan(x1*y1/(r*z2 + x1**2 + z2**2)) + k4*sp.atan(x1*y2/(r*z1 + x1**2 + z1**2)) - k4*sp.atan(x1*y2/(r*z2 + x1**2 + z2**2)) + k4*sp.atan(x2*y1/(r*z1 + \
        x2**2 + z1**2)) - k4*sp.atan(x2*y1/(r*z2 + x2**2 + z2**2)) - k4*sp.atan(x2*y2/(r*z1 + x2**2 + z1**2)) + k4*sp.atan(x2*y2/(r*z2 + x2**2 + z2**2)) - k5*sp.atan(x1*y1/(r*z1 + y1**2 + z1**2)) + k5*sp.atan(x1*y1/(r*z2 + y1**2 + z2**2)) + k5*sp.atan(x1*y2/(r*z1 + y2**2 + z1**2)) - k5*sp.atan(x1*y2/(r*z2 + y2**2 + z2**2)) + k5*sp.atan(x2*y1/(r*z1 + y1**2 + z1**2)) - k5*sp.atan(x2*y1/(r*z2 + y1**2 \
            + z2**2)) - k5*sp.atan(x2*y2/(r*z1 + y2**2 + z1**2)) + k5*sp.atan(x2*y2/(r*z2 + y2**2 + z2**2)) - k6*sp.atan(x1*y1/(r*z1)) + k6*sp.atan(x1*y1/(r*z2)) + k6*sp.atan(x1*y2/(r*z1)) - k6*sp.atan(x1*y2/(r*z2)) + k6*sp.atan(x2*y1/(r*z1)) - k6*sp.atan(x2*y1/(r*z2)) - k6*sp.atan(x2*y2/(r*z1)) + k6*sp.atan(x2*y2/(r*z2)))
            '''
    return g

def cal(params):
    x = params[0]
    y = params[1]
    m = params[2]
    Mx, My, Mz = mag(m.k)
    x1 = m.xc + m.a/2
    x2 = m.xc - m.a/2
    y1 = m.yc + m.b/2
    y2 = m.yc - m.b/2
    z1 = m.zc + m.c/2
    z2 = m.zc - m.c/2
    # r1 = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2 + z1 ** 2)
    # r2 = math.sqrt((x2 - x) ** 2 + (y2 - y) ** 2 + z2 ** 2)
    # r = math.sqrt((m.xc - x) ** 2 + (m.yc - y) ** 2 + m.zc ** 2)
    h_x = hx(x, y, Mx, My, Mz, x1, x2, y1, y2, z1, z2)
    h_y = hy(x, y, Mx, My, Mz, x1, x2, y1, y2, z1, z2)
    z_a = z(x, y, Mx, My, Mz, x1, x2, y1, y2, z1, z2)
    # return h_x, h_y, z_a
    return deltaT(z_a, h_x, h_y)
    
# core函数
def core(m, X, Y):
    # Z = np.asarray(X, dtype= None, order= None)
    params = []
    for x in range(0, X.shape[0]):
        for y in range(0, X.shape[1]):
            params.append((Y[x, y], X[x, y], m))
            #hax, hay, za = cal(Y[x, y], X[x, y], m)
            #Z[x, y] = deltaT(za, hax, hay)
    with mp.Pool(6) as p:
        result = list(tqdm.tqdm(p.imap(cal, params), total=len(params), desc='长方体正演计算中：'))
    p.close()
    p.join()
    #cal(Y[50, 50], X[50, 50], m)        
    # Hax = hx(m, )
    # Hax = hx(m.xc + m.a/2, m.yc + m.b/2, m.zc + c/2) - hx(m.xc + m.a/2, m.yc + m.b/2, m.zc + c/2)
    '''
    temp = []
    for res in result:
        temp.append(res.get())
    '''
    Z = np.asarray(result).reshape(X.shape[0], X.shape[1])
    # print(np.shape(Z))
    return Z

def main():
    m = cuboid(500, 500, 51, 200, 200, 100, 1)
    params = [(0, 0, m)]
    print(kernel(params[0]))

if __name__ == '__main__':
    main()