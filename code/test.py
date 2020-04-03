import numpy as np
import draw
import ball
import cuboid
import time
import progressbar
import random
import kernel

def main():
    # 确定网格数量
    dx = 200
    dy = 200
    dz = 100
    # 构建网格
    time_start=time.time()
    x = np.arange(0, 1000 + dx, dx)
    y = np.arange(0, 1000 + dy, dy)
    z = np.arange(0, 500 + dz, dz)
    X, Y = np.meshgrid(x, y)
    # 选择模型，长方体, 参数从左到右依次是：xc, yc, zc, a, b, c, k
    '''
    m = cuboid.cuboid(500, 500, 300, 100, 100, 200, 0.015)
    data = cuboid.core(m, X, Y)
    '''

    # 生成多个长方体的模型
    '''
    p = progressbar.ProgressBar()
    N = 3
    data = np.zeros((X.shape[0], X.shape[1]))
    p.start()
    for i in range(N):
        m = cuboid.cuboid(500, 675 - i * 50, 75 + 50 * i, 300, 350, 50, 0.05)
        temp = cuboid.core(m, X, Y)
        data = data + temp
        time.sleep(0.01)
        p.update(i+1)
    p.finish()
    '''

    # 选择模型，球体, 参数从左到右依次是：xc, yc, zc, r, k
    # m = ball.ball(0, 0, 15, 10, 0.015)
    # Z = ball.core(m, X, Y)

    # 写入文件
    # np.savetxt('./res/data.txt', data, fmt='%f', delimiter=',')

    # 读取文件
    # data = np.loadtxt('./res/seven_cuboids.txt', delimiter=',')

    # 给数据添加高斯参数, 加3%的噪声，mu为均值，sigma为方差，数值还需要再确定
    '''
    mu = 1
    sigma = 0.02
    Noise =  np.zeros((X.shape[0], X.shape[1]))
    for item in Noise:
        item += random.gauss(mu, sigma)
    
    print(Noise)
    data = data + Noise
    '''

    # 生成核函数矩阵
    # 注意选取的单位不要在边界上
    k = kernel.core(x, y, z)
    # 数太小了，需要扩大数量级再存储
    k = k * 1e10
    np.savetxt('./res/kernel.txt', k, fmt='%f', delimiter=',')
    

    # 输出程序运行时间
    time_end=time.time()
    print('time cost:', time_end-time_start,'s')
    # 调用绘图函数
    #print(np.shape(Z))
    #draw.contour(x, y, data)


if __name__ == '__main__':
    main()