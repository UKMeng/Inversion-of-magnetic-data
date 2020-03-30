import numpy as np
import matplotlib.pyplot as plt

def contour(X, Y, Z):
    #cset = plt.contourf(X, Y, Z) 
    #or cmap='hot'
    #画出8条线，并将颜色设置为黑色
    cset = plt.contourf(X, Y, Z, 15)
    #contour = plt.contour(X, Y, Z, 15, colors= 'k')
    #等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
    #plt.clabel(contour, fontsize=10, colors='k')
    #去掉坐标轴刻度
    #plt.xticks(())  
    #plt.yticks(())  
    #设置颜色条，（显示在图片右边）
    plt.colorbar(cset)
    #显示
    plt.show()
