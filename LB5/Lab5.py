# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:05:20 2021

@author: AM4
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# Сначала определим на каком устройстве будем работать - GPU или CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Затем загружаем данные

data_transforms = transforms.Compose([
                        transforms.Resize(68),
                        transforms.CenterCrop(64),
                        transforms.ToTensor()])

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data2')

train_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(data_dir, 'train'),
    transform=data_transforms)

test_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(data_dir, 'test'),
    transform=data_transforms)


# посмотрим какие классы содержатся в наборе
train_dataset.classes

# сохраним названия этих классов
class_names = train_dataset.classes

# Список изображений можно получить следующим образом:
train_set = train_dataset.samples
print(train_set[1]) #  каждая строка списка содержит путь к изображению и метку класса

# посмотрим на размер нашего набора данных
print(len(train_set))

# В реальных задачах объем данных может быть очень большим, поэтому класс datasets
# содержит только описание набора данных и путь к ним
# Непосредственную загрузку данных в оперативную память осуществляет класс dataloader

batch_size = 10 # данные будут загружаться частями (batch)

# параметр shuffle указывает что данные будут выбираься случайно,
# num_workers - указывает сколько процессов (условно ядер процессора) будет
# задействовано при загрузке (т.к. при загрузке еще выполняются преобразования transform,
# то это может быть вычислительно затратно)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                           shuffle=True, num_workers=0)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                           shuffle=False, num_workers=0)



# Загрузим одну порцию данных.
# Каждое обращение к DataLoader возвращает изображения и их классы
inputs, classes = next(iter(train_loader))
inputs.shape # 10 изображений, 3 канала (RGB), размер каждого 224х224 пикселя

# построим на картинке
img = torchvision.utils.make_grid(inputs, nrow = 5) # метод делает сетку из картинок
img = img.numpy().transpose((1, 2, 0)) # для отображения через matplotlib 
plt.imshow(img)


# Теперь можно переходить к созданию сети
# Для этого будем использовать как и ранее метод Sequential
# который объединит несколько слоев в один стек
class CnNet(nn.Module):
    def __init__(self, num_classes=10):
        nn.Module.__init__(self)
        self.layer1 = nn.Sequential(
        # первый сверточный слой с ReLU активацией и maxpooling-ом
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=2), # 3 канала, 16 фильтров, размер ядра 7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # второй сверточный слой 
        # количество каналов второго слоя равно количеству фильтров предыдущего слоя
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # третий сверточный слой 
        # ядро фильтра от слоя к слою уменьшается
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # классификационный слой имеет нейронов: количество фильтров * размеры карты признаков
        self.fc = nn.Linear(8*8*64, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1) # флаттеринг
        out = self.fc(out)
        return out



# Количество классов
num_classes = 3
   
# создаем экземпляр сети
net = CnNet(num_classes).to(device)

# Задаем функцию потерь и алгоритм оптимизации
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

if __name__ == '__main__':

    import time
    t = time.time()

    num_epochs = 50
    save_loss = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = lossFn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            save_loss.append(loss.item())
            if i % 100 == 0:
                print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                      str(i) + ' Ошибка: ', loss.item())

    print(time.time() - t)

    plt.figure()
    plt.plot(save_loss)

    correct_predictions = 0
    num_test_samples = len(test_dataset)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images)
            _, pred_class = torch.max(pred.data, 1)
            correct_predictions += (pred_class == labels).sum().item()

    print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')

    torch.save(net.state_dict(), 'CnNet.ckpt')

    # --- Вторая часть (предобученная AlexNet) ---
    data_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]) ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=data_transforms)

    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=data_transforms)

    batch_size = 10

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                        shuffle=True,  num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                        shuffle=False, num_workers=0)

    net = torchvision.models.alexnet(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False

    num_classes = 3
    new_classifier = net.classifier[:-1]
    new_classifier.add_module('fc', nn.Linear(4096, num_classes))
    net.classifier = new_classifier

    net = net.to(device)

    correct_predictions = 0
    num_test_samples = len(test_dataset)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images)
            _, pred_class = torch.max(pred.data, 1)
            correct_predictions += (pred_class == labels).sum().item()
    print('Точность модели (до обучения): ' + str(100 * correct_predictions / num_test_samples) + '%')

    num_epochs = 2
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    t = time.time()
    save_loss = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = lossFn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            save_loss.append(loss.item())
            if i % 100 == 0:
                print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                      str(i) + ' Ошибка: ', loss.item())

    print(time.time() - t)
    plt.figure()
    plt.plot(save_loss)

    correct_predictions = 0
    num_test_samples = len(test_dataset)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images)
            _, pred_class = torch.max(pred.data, 1)
            correct_predictions += (pred_class == labels).sum().item()
    print('Точность модели (после обучения): ' + str(100 * correct_predictions / num_test_samples) + '%')

    import random

# Создаем словарь индексов по классам
class_indices = {i: [] for i in range(len(train_dataset.classes))}

# Собираем индексы изображений каждого класса
for idx, (_, label) in enumerate(test_dataset):
    class_indices[label].append(idx)

# Берем по 4 случайных изображения из каждого класса
sample_indices = []
for label, indices in class_indices.items():
    sample_indices += random.sample(indices, 4)

# Формируем батч изображений и меток
inputs = torch.stack([test_dataset[i][0] for i in sample_indices])
labels = torch.tensor([test_dataset[i][1] for i in sample_indices])

# Предсказание сети
pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)
class_names = train_dataset.classes

# --- Формируем батч: по 4 случайных изображения из каждого класса ---
class_indices = {i: [] for i in range(len(test_dataset.classes))}

# Собираем индексы изображений каждого класса
for idx, (_, label) in enumerate(test_dataset):
    class_indices[label].append(idx)

# Берем по 4 случайных изображения из каждого класса (или меньше, если их мало)
sample_indices = []
for label, indices in class_indices.items():
    if len(indices) < 4:
        sample_indices += indices
    else:
        sample_indices += random.sample(indices, 4)

# Формируем батч изображений и меток
inputs = torch.stack([test_dataset[i][0] for i in sample_indices])
labels = torch.tensor([test_dataset[i][1] for i in sample_indices])

# --- Предсказание сети ---
inputs_device = inputs.to(device)
pred = net(inputs_device)
_, pred_class = torch.max(pred.data, 1)
class_names = train_dataset.classes

# --- Визуализация изображений в сетке ---
num_classes = len(class_names)
fig, axes = plt.subplots(num_classes, 4, figsize=(12, 3*num_classes))

for idx, (img_tensor, pred_label) in enumerate(zip(inputs, pred_class.cpu())):
    row = idx // 4
    col = idx % 4
    img = img_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    axes[row, col].imshow(img)
    axes[row, col].set_title(class_names[pred_label])
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

