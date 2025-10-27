import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd

class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Tanh(),                       # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Tanh()
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred
    
df = pd.read_csv('dataset_simple.csv')
X = torch.Tensor(df.iloc[1:98, 0:2].values)
y = df.iloc[1:98, 2].values
y = torch.tensor(df.iloc[1:98, 2].values, dtype=torch.float32).reshape(-1,1)    

inputSize = X.shape[1] # количество признаков задачи 

hiddenSizes = 3 #  число нейронов скрытого слоя 

outputSize = 1

net = NNet(inputSize,hiddenSizes,outputSize)

with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >=0, 1, -1).reshape(-1,1))

err = sum(abs(y-pred))/2
lossFn = nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.0099)

epohs = 121
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred, y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%10==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >=0, 1, -1).reshape(-1,1))
err = sum(abs(y-pred))/2
print('\nОшибка (количество несовпавших ответов): ')
print(err) # обучение работает, не делает ошибок или делает их достаточно мало
