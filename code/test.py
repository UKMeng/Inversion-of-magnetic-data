import numpy as np
import draw
import ball
import cuboid
import time
import progressbar

def main():
    # 构建网格
    time_start=time.time()
    x = np.linspace(0, 1000, 100)
    y = np.linspace(0, 1000, 100)
    X, Y = np.meshgrid(x, y)
    # 选择模型，长方体, 参数从左到右依次是：xc, yc, zc, a, b, c, k
    # m = cuboid.cuboid(50, 50, 15, 40, 40, 10, 0.015)
    # Z = cuboid.core(m, X, Y)

    # 生成多个长方体的模型
    '''
    p = progressbar.ProgressBar()
    N = 7
    Z = np.zeros((X.shape[0], X.shape[1]))
    p.start()
    for i in range(N):
        m = cuboid.cuboid(500, 675 - i * 50, 75 + 50 * i, 300, 350, 50, 0.05)
        temp = cuboid.core(m, X, Y)
        Z = Z + temp
        time.sleep(0.01)
        p.update(i+1)
    p.finish()
    '''
    # 选择模型，球体, 参数从左到右依次是：xc, yc, zc, r, k
    # m = ball.ball(0, 0, 15, 10, 0.015)
    # Z = ball.core(m, X, Y)

    # 写入文件
    # np.savetxt('./res/data.txt', Z, fmt='%f', delimiter=',')

    # 读取文件
    Z = np.loadtxt('./res/data.txt', delimiter=',')

    # 输出程序运行时间
    time_end=time.time()
    print('time cost:', time_end-time_start,'s')
    # 调用绘图函数
    #print(np.shape(Z))
    draw.contour(x, y, Z)


if __name__ == '__main__':
    main()