---
title: "[PyTorch] Multivariable Linear Regression"
date: 2023-02-28

categories:
  - AI
  - Deep Learning
tags:
    - Python
    - PyTorch
---

## 학습 목표
- 다중 선형 회귀(Multivariable Linear Regression)
- 가설 함수(Hypothesis Function)
- 평균 제곱 오차(Mean Squared Error)
- 경사하강법(Gradient descent)

## Library


```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```


```python
# torch seed 값 고정
torch.manual_seed(1)
```




    <torch._C.Generator at 0x1cb8526b210>



## 다중 회귀분석 구현


```python
# Data
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
w = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([w, b], lr=1e-5)

nb_epochs = 30
for epoch in range(nb_epochs + 1):
    # H(x)
    hypothesis = x_train.matmul(w) + b

    # cost
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} cost: {:.6f}'.format(epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))
    
```

    Epoch    0/30 hypothesis: tensor([0., 0., 0., 0., 0.]) cost: 29661.800781
    Epoch    1/30 hypothesis: tensor([67.2578, 80.8397, 79.6523, 86.7394, 61.6605]) cost: 9298.520508
    Epoch    2/30 hypothesis: tensor([104.9128, 126.0990, 124.2466, 135.3015,  96.1821]) cost: 2915.712402
    Epoch    3/30 hypothesis: tensor([125.9942, 151.4381, 149.2133, 162.4896, 115.5097]) cost: 915.040527
    Epoch    4/30 hypothesis: tensor([137.7968, 165.6247, 163.1911, 177.7112, 126.3307]) cost: 287.936005
    Epoch    5/30 hypothesis: tensor([144.4044, 173.5674, 171.0168, 186.2332, 132.3891]) cost: 91.371010
    Epoch    6/30 hypothesis: tensor([148.1035, 178.0144, 175.3980, 191.0042, 135.7812]) cost: 29.758139
    Epoch    7/30 hypothesis: tensor([150.1744, 180.5042, 177.8508, 193.6753, 137.6805]) cost: 10.445305
    Epoch    8/30 hypothesis: tensor([151.3336, 181.8983, 179.2240, 195.1707, 138.7440]) cost: 4.391228
    Epoch    9/30 hypothesis: tensor([151.9824, 182.6789, 179.9928, 196.0079, 139.3396]) cost: 2.493135
    Epoch   10/30 hypothesis: tensor([152.3454, 183.1161, 180.4231, 196.4765, 139.6732]) cost: 1.897688
    Epoch   11/30 hypothesis: tensor([152.5485, 183.3610, 180.6640, 196.7389, 139.8602]) cost: 1.710541
    Epoch   12/30 hypothesis: tensor([152.6620, 183.4982, 180.7988, 196.8857, 139.9651]) cost: 1.651412
    Epoch   13/30 hypothesis: tensor([152.7253, 183.5752, 180.8742, 196.9678, 140.0240]) cost: 1.632387
    Epoch   14/30 hypothesis: tensor([152.7606, 183.6184, 180.9164, 197.0138, 140.0571]) cost: 1.625923
    Epoch   15/30 hypothesis: tensor([152.7802, 183.6427, 180.9399, 197.0395, 140.0759]) cost: 1.623412
    Epoch   16/30 hypothesis: tensor([152.7909, 183.6565, 180.9530, 197.0538, 140.0865]) cost: 1.622141
    Epoch   17/30 hypothesis: tensor([152.7968, 183.6643, 180.9603, 197.0618, 140.0927]) cost: 1.621253
    Epoch   18/30 hypothesis: tensor([152.7999, 183.6688, 180.9644, 197.0662, 140.0963]) cost: 1.620500
    Epoch   19/30 hypothesis: tensor([152.8014, 183.6715, 180.9666, 197.0686, 140.0985]) cost: 1.619770
    Epoch   20/30 hypothesis: tensor([152.8020, 183.6731, 180.9677, 197.0699, 140.1000]) cost: 1.619033
    Epoch   21/30 hypothesis: tensor([152.8022, 183.6741, 180.9683, 197.0706, 140.1009]) cost: 1.618346
    Epoch   22/30 hypothesis: tensor([152.8021, 183.6749, 180.9686, 197.0709, 140.1016]) cost: 1.617637
    Epoch   23/30 hypothesis: tensor([152.8019, 183.6754, 180.9687, 197.0710, 140.1022]) cost: 1.616930
    Epoch   24/30 hypothesis: tensor([152.8016, 183.6758, 180.9687, 197.0711, 140.1027]) cost: 1.616221
    Epoch   25/30 hypothesis: tensor([152.8012, 183.6762, 180.9686, 197.0710, 140.1032]) cost: 1.615508
    Epoch   26/30 hypothesis: tensor([152.8008, 183.6765, 180.9686, 197.0710, 140.1036]) cost: 1.614815
    Epoch   27/30 hypothesis: tensor([152.8004, 183.6768, 180.9684, 197.0709, 140.1041]) cost: 1.614109
    Epoch   28/30 hypothesis: tensor([152.8000, 183.6772, 180.9683, 197.0707, 140.1045]) cost: 1.613387
    Epoch   29/30 hypothesis: tensor([152.7995, 183.6775, 180.9682, 197.0706, 140.1049]) cost: 1.612701
    Epoch   30/30 hypothesis: tensor([152.7991, 183.6778, 180.9681, 197.0705, 140.1053]) cost: 1.611987
    

## nn.Module
- `nn.Module`을 상속해서 모델 생성
- `nn.Linear(3, 1)` : 입력차원 3, 출력차원 1
- Hypothesis 계산은 `forward()`에서 한다.
- Gradient 계산은 PyTorch가 알아서 해준다. `backward()`


```python
import torch.nn as nn

class MultiLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.linear(x)
```


```python
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
model = MultiLinearRegressionModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
```

    Epoch    0/20 Cost: 31667.599609
    Epoch    1/20 Cost: 9926.266602
    Epoch    2/20 Cost: 3111.513916
    Epoch    3/20 Cost: 975.451355
    Epoch    4/20 Cost: 305.908539
    Epoch    5/20 Cost: 96.042496
    Epoch    6/20 Cost: 30.260748
    Epoch    7/20 Cost: 9.641701
    Epoch    8/20 Cost: 3.178671
    Epoch    9/20 Cost: 1.152871
    Epoch   10/20 Cost: 0.517863
    Epoch   11/20 Cost: 0.318801
    Epoch   12/20 Cost: 0.256388
    Epoch   13/20 Cost: 0.236821
    Epoch   14/20 Cost: 0.230660
    Epoch   15/20 Cost: 0.228719
    Epoch   16/20 Cost: 0.228095
    Epoch   17/20 Cost: 0.227880
    Epoch   18/20 Cost: 0.227799
    Epoch   19/20 Cost: 0.227762
    Epoch   20/20 Cost: 0.227732
    
