import torch
import torch.nn as nn
import pandas as pd

df = pd.read_csv('data.csv')


y = torch.where(torch.tensor(df.iloc[:, 4].values == "Iris-setosa"), 1.0, -1.0).unsqueeze(1)
X = torch.tensor(df.iloc[:, [0, 1, 2]].values, dtype=torch.float32)

print("Признаки X:", X.shape)
print("Метки y:", y.shape)


linear = nn.Linear(3, 1)  

lossFn = nn.MSELoss()           # среднеквадратичная ошибка
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)


num_epochs = 50

for epoch in range(num_epochs):
    optimizer.zero_grad()         
    pred = linear(X)               
    loss = lossFn(pred, y)          
    loss.backward()               
    optimizer.step()               

    if (epoch+1) % 10 == 0:
        print(f"Эпоха {epoch+1}, ошибка: {loss.item():.4f}")


with torch.no_grad():  
    predictions = torch.sign(linear(X)) 
    print("Предсказания после обучения:", predictions.squeeze())
    print("Реальные метки:", y.squeeze())


