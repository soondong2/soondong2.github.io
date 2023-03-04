---
title: "[PyTorch] Logistic Regression"
date: 2023-03-01

categories:
  - AI
  - Deep Learning
tags:
    - 
    - PyTorch
---

## 학습 목표
- 로지스틱 회귀(Logistic Regression)
- 가설(Hypothesis)
- 손실함수(Cost Function)
- 평가(Evaluation)


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```


```python
# seed 고정
torch.manual_seed(1)
```


## Logistic Regression
![image](https://user-images.githubusercontent.com/100760303/222025583-78dfc8f6-4826-467d-ae3c-e81982bc9451.png)

![image](https://user-images.githubusercontent.com/100760303/222025661-76560b25-e7a8-4f27-a02e-fd66a96327b4.png)



```python
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
```


```python
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
```


```python
print(x_train.shape)
print(y_train.shape)
```

    torch.Size([6, 2])
    torch.Size([6, 1])
    


```python
# 모델 초기화
w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([w, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # H(x)
    hypothesis = torch.sigmoid(x_train.matmul(w) + b)
    # Cost
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}]{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
```

    Epoch    0]1000 Cost: 0.693147
    Epoch  100]1000 Cost: 0.134722
    Epoch  200]1000 Cost: 0.080643
    Epoch  300]1000 Cost: 0.057900
    Epoch  400]1000 Cost: 0.045300
    Epoch  500]1000 Cost: 0.037261
    Epoch  600]1000 Cost: 0.031672
    Epoch  700]1000 Cost: 0.027556
    Epoch  800]1000 Cost: 0.024394
    Epoch  900]1000 Cost: 0.021888
    Epoch 1000]1000 Cost: 0.019852
    

## Diabetes Dataset Ligistic Regression with `nn.Modul`


```python
import pandas as pd
import numpy as np

# csv 파일을 txt 파일로 읽어옴
df = np.loadtxt('C:/Users/USER/Desktop/Data/diabetes.csv', delimiter=',', dtype=np.float32)
```


```python
x_data = df[:, 0:-1]
y_data = df[:, [-1]] # 0, 1
```


```python
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
```


```python
print(x_train.dim())
print(x_train.shape)

print(y_train.dim())
print(y_train.shape)
```

    2
    torch.Size([759, 8])
    2
    torch.Size([759, 1])
    


```python
# nn.Module
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
```


```python
model = BinaryClassifier()
```


```python
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 100
for epoch in range(nb_epochs + 1):
    # H(x)
    hypothesis = model(x_train)
    # Cost
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 10번마다 로그 출력
    if epoch % 10 == 0:
        # 확률이 0.5보다 크면 1, 아니면 0
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}'.format(
            epoch, nb_epochs, cost.item(), accuracy
        ))
```

    Epoch    0/100 Cost: 0.625512 Accuracy 0.65
    Epoch   10/100 Cost: 0.567587 Accuracy 0.69
    Epoch   20/100 Cost: 0.536736 Accuracy 0.73
    Epoch   30/100 Cost: 0.518084 Accuracy 0.76
    Epoch   40/100 Cost: 0.506026 Accuracy 0.77
    Epoch   50/100 Cost: 0.497824 Accuracy 0.77
    Epoch   60/100 Cost: 0.492028 Accuracy 0.77
    Epoch   70/100 Cost: 0.487808 Accuracy 0.77
    Epoch   80/100 Cost: 0.484664 Accuracy 0.76
    Epoch   90/100 Cost: 0.482276 Accuracy 0.77
    Epoch  100/100 Cost: 0.480432 Accuracy 0.77
