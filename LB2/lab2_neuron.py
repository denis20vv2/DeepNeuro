import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('data.csv')
print(df.head())
y = np.where(df.iloc[:, 4] == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 1, 2]].values

def neuron(w, x):
    if np.dot(w[1:], x) + w[0] >= 0:
        predict = 1
    else: 
        predict = -1
    return predict

w = np.random.random(4)  # w0, w1, w2, w3
eta = 0.01
w_iter = []

sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w,xi)
    sum_err += abs(target - predict)/2

print("Всего ошибок: ", sum_err)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], c='red', marker='o', label='Iris-setosa')
ax.scatter(X[y==-1, 0], X[y==-1, 1], X[y==-1, 2], c='blue', marker='x', label='Other')


xx, yy = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 10),
                     np.linspace(X[:,1].min(), X[:,1].max(), 10))
zz = (-w[0] - w[1]*xx - w[2]*yy) / w[3]

ax.plot_surface(xx, yy, zz, color='green', alpha=0.3)
ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')
ax.set_title('Нейрон с тремя признаками')
ax.legend()
plt.show()
