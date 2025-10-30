import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# --- Загружаем данные ---
df = pd.read_csv('dataset_simple.csv')
X = df[['age']].values.astype(float)
y = df[['income']].values.astype(float)

# --- Нормализация вручную ---
X_min, X_max = X.min(), X.max()
y_min, y_max = y.min(), y.max()

X_norm = (X - X_min) / (X_max - X_min)   # масштабируем в [0,1]
y_norm = (y - y_min) / (y_max - y_min)

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y_norm, dtype=torch.float32)

# --- Сеть ---
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, X):
        return self.layers(X)

net = NNet(1, 5, 1)

lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)


epochs = 200
for i in range(epochs):
    pred = net(X_tensor)
    loss = lossFn(pred, y_tensor) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        print(f"Эпоха {i}: loss = {loss.item():.6f}")

with torch.no_grad():
    pred_norm = net(X_tensor)
  
    pred_income = pred_norm.numpy() * (y_max - y_min) + y_min

sorted_indices = np.argsort(X.flatten())
X_sorted = X.flatten()[sorted_indices]
pred_income_sorted = pred_income.flatten()[sorted_indices]


unique_ages, unique_indices = np.unique(X_sorted, return_index=True)
unique_pred_income = pred_income_sorted[unique_indices]

print("\nПредсказанный доход для каждого возраста (уникальные значения):")
for age_val, income_val in zip(unique_ages, unique_pred_income):
    print(f"Возраст: {age_val}, предсказанный доход: {income_val:.2f}")

