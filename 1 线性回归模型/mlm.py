import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#导入数据集
data_set = pd.read_csv('mlm.csv')

#导入训练的样本
x, y, z = [], [], []
for i in range(0, 849):
    x.append([data_set.values[i][0], data_set.values[i][1]])
    z.append(data_set.values[i][2])

#对数据进行训练
result = linear_model.LinearRegression()
result.fit(x, z)

#导入测试样本
x_test, y_test, z_test = [], [], []
for i in range(850, 999):
    x_test.append(data_set.values[i][0])
    y_test.append(data_set.values[i][1])
    z_test.append(data_set.values[i][2])

#输出参数
print("Z=%f*X+%f*Y+%f" % (result.coef_[0], result.coef_[1], result.intercept_))

#计算损失
loss = 0
A = result.coef_[0]
B = result.coef_[1]
C = -1
D = result.intercept_
for i in range(0, 999):
    a = data_set.values[i][0]
    b = data_set.values[i][1]
    c = data_set.values[i][2]
    loss = loss + (A*a + B*b + C*c + D)/((A*A + B*B + C*C) ** 0.5)
print("loss = %f" % (loss/2))

#绘制三维坐标图
ax = plt.axes(projection="3d")
ax.scatter3D(x_test, y_test, z_test)

x_drawing = np.linspace(0, 100)
y_drawing = np.linspace(0, 100)
X_drawing, Y_drawing = np.meshgrid(x_drawing, y_drawing)
ax.plot_surface(X=X_drawing,
                Y=Y_drawing,
                Z=X_drawing * result.coef_[0] + Y_drawing * result.coef_[1] + result.intercept_,
                color='r',
                alpha=0.5)
#可视化
ax.view_init(elev=30, azim=30)
plt.show()