import math
import numpy as np
import draw

def main():
    # 基本参数
    I = math.radians(75)
    D = math.radians(25)
    # 构建网格
    x = np.linspace(0, 1000, 100)
    y = np.linspace(0, 1000, 100)
    X, Y = np.meshgrid(x, y)
    # 计算
    
    # 调用绘图函数
    draw.contour(X, Y, Z)

if __name__ == '__main__':
    main()