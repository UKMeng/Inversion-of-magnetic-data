# 用于计算核函数
import numpy as np
import cuboid
import multiprocessing as mp
import tqdm
'''
# 全局参数
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
'''
def c_list(x, y, z):
    cuboid_list = []
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    for X in x[2:len(x)-2]:
        for Y in y[2:len(y)-2]:
            for Z in z[1:len(z)-2]:
                x0 = X + dx/2
                y0 = Y + dy/2
                z0 = Z + dz/2
                m = cuboid.cuboid(x0, y0, z0, dx, dy, dz, 1)
                cuboid_list.append(m)
    return cuboid_list

def core(x, y, z): 
    cuboid_list = c_list(x, y, z)
    params = []
    for X in x:
        for Y in y:
            for m in cuboid_list:
                params.append((X, Y, m))
                # result.append(p.apply_async(cuboid.cal, args=(Y, X, m)))
    with mp.Pool(6) as p:
        result = list(tqdm.tqdm(p.imap(cuboid.kernel, params), total=len(params), desc='核函数矩阵计算中：'))
    p.close()
    p.join()
    print(result)
    '''
    temp = []
    for res in result:
        temp.append(res.get())
    '''
    kernel = np.asarray(result).reshape(x.shape[0] * y.shape[0], len(cuboid_list))
    print(kernel.shape)
    return kernel