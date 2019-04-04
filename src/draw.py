import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
with open('MAE_MSE','r') as f:
    lines = f.readlines()

maes = []
mses = []
iters = []

i=0
for line in lines[::5]:
    s = line.split('_')
    maes.append(float(s[1]))
    mses.append(float(s[-1]))
    iters.append(i)
    i = i+250
    
plt.plot(iters, maes, color='green', label='MAE')
plt.plot(iters, mses, color='red', label='MSE')

plt.legend()

plt.xlabel('迭代次数')
plt.ylabel('数值')
#添加标题
plt.title('训练过程')

plt.show()