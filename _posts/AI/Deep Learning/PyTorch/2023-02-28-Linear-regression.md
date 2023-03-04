---
title: "[PyTorch] Linear Regression"
date: 2023-02-28

categories:
  - AI
  - Deep Learning
tags:
    - ML
    - PyTorch
---

## 학습 목표
- 선형 회귀(Linear Regression)
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





```python
# Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# 모델 초기화
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([w, b], lr=0.01)

nb_epochs = 100
for epoch in range(nb_epochs + 1):
     # H(x) 계산
    hypothesis = x_train * w + b
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{}, W: {:.3f}, b: {:.3f}, cost: {:.6f}'.format(epoch, nb_epochs, w.item(), b.item(), cost.item()))
    
```

    Epoch    0/100, W: 0.093, b: 0.040, cost: 4.666667
    Epoch    1/100, W: 0.176, b: 0.075, cost: 3.692741
    Epoch    2/100, W: 0.250, b: 0.107, cost: 2.922885
    Epoch    3/100, W: 0.316, b: 0.135, cost: 2.314336
    Epoch    4/100, W: 0.374, b: 0.159, cost: 1.833292
    Epoch    5/100, W: 0.426, b: 0.181, cost: 1.453034
    Epoch    6/100, W: 0.473, b: 0.201, cost: 1.152441
    Epoch    7/100, W: 0.514, b: 0.218, cost: 0.914819
    Epoch    8/100, W: 0.551, b: 0.233, cost: 0.726974
    Epoch    9/100, W: 0.583, b: 0.246, cost: 0.578474
    Epoch   10/100, W: 0.612, b: 0.258, cost: 0.461073
    Epoch   11/100, W: 0.638, b: 0.268, cost: 0.368257
    Epoch   12/100, W: 0.661, b: 0.277, cost: 0.294872
    Epoch   13/100, W: 0.682, b: 0.285, cost: 0.236847
    Epoch   14/100, W: 0.700, b: 0.292, cost: 0.190963
    Epoch   15/100, W: 0.716, b: 0.298, cost: 0.154676
    Epoch   16/100, W: 0.731, b: 0.304, cost: 0.125976
    Epoch   17/100, W: 0.744, b: 0.309, cost: 0.103271
    Epoch   18/100, W: 0.755, b: 0.313, cost: 0.085307
    Epoch   19/100, W: 0.766, b: 0.316, cost: 0.071090
    Epoch   20/100, W: 0.775, b: 0.319, cost: 0.059834
    Epoch   21/100, W: 0.783, b: 0.322, cost: 0.050920
    Epoch   22/100, W: 0.791, b: 0.324, cost: 0.043856
    Epoch   23/100, W: 0.797, b: 0.326, cost: 0.038255
    Epoch   24/100, W: 0.803, b: 0.328, cost: 0.033811
    Epoch   25/100, W: 0.808, b: 0.329, cost: 0.030280
    Epoch   26/100, W: 0.813, b: 0.330, cost: 0.027473
    Epoch   27/100, W: 0.817, b: 0.331, cost: 0.025237
    Epoch   28/100, W: 0.821, b: 0.332, cost: 0.023452
    Epoch   29/100, W: 0.825, b: 0.332, cost: 0.022025
    Epoch   30/100, W: 0.828, b: 0.332, cost: 0.020880
    Epoch   31/100, W: 0.830, b: 0.333, cost: 0.019958
    Epoch   32/100, W: 0.833, b: 0.333, cost: 0.019213
    Epoch   33/100, W: 0.835, b: 0.333, cost: 0.018607
    Epoch   34/100, W: 0.837, b: 0.333, cost: 0.018112
    Epoch   35/100, W: 0.839, b: 0.333, cost: 0.017705
    Epoch   36/100, W: 0.841, b: 0.332, cost: 0.017366
    Epoch   37/100, W: 0.842, b: 0.332, cost: 0.017082
    Epoch   38/100, W: 0.844, b: 0.332, cost: 0.016842
    Epoch   39/100, W: 0.845, b: 0.331, cost: 0.016636
    Epoch   40/100, W: 0.846, b: 0.331, cost: 0.016457
    Epoch   41/100, W: 0.847, b: 0.330, cost: 0.016300
    Epoch   42/100, W: 0.848, b: 0.330, cost: 0.016160
    Epoch   43/100, W: 0.849, b: 0.329, cost: 0.016033
    Epoch   44/100, W: 0.850, b: 0.329, cost: 0.015918
    Epoch   45/100, W: 0.851, b: 0.328, cost: 0.015811
    Epoch   46/100, W: 0.852, b: 0.328, cost: 0.015711
    Epoch   47/100, W: 0.853, b: 0.327, cost: 0.015616
    Epoch   48/100, W: 0.853, b: 0.326, cost: 0.015526
    Epoch   49/100, W: 0.854, b: 0.326, cost: 0.015440
    Epoch   50/100, W: 0.855, b: 0.325, cost: 0.015356
    Epoch   51/100, W: 0.855, b: 0.324, cost: 0.015275
    Epoch   52/100, W: 0.856, b: 0.324, cost: 0.015196
    Epoch   53/100, W: 0.856, b: 0.323, cost: 0.015118
    Epoch   54/100, W: 0.857, b: 0.322, cost: 0.015042
    Epoch   55/100, W: 0.857, b: 0.322, cost: 0.014967
    Epoch   56/100, W: 0.858, b: 0.321, cost: 0.014892
    Epoch   57/100, W: 0.858, b: 0.320, cost: 0.014819
    Epoch   58/100, W: 0.859, b: 0.319, cost: 0.014746
    Epoch   59/100, W: 0.859, b: 0.319, cost: 0.014675
    Epoch   60/100, W: 0.859, b: 0.318, cost: 0.014603
    Epoch   61/100, W: 0.860, b: 0.317, cost: 0.014532
    Epoch   62/100, W: 0.860, b: 0.316, cost: 0.014462
    Epoch   63/100, W: 0.861, b: 0.316, cost: 0.014392
    Epoch   64/100, W: 0.861, b: 0.315, cost: 0.014323
    Epoch   65/100, W: 0.861, b: 0.314, cost: 0.014254
    Epoch   66/100, W: 0.862, b: 0.314, cost: 0.014185
    Epoch   67/100, W: 0.862, b: 0.313, cost: 0.014117
    Epoch   68/100, W: 0.862, b: 0.312, cost: 0.014049
    Epoch   69/100, W: 0.863, b: 0.311, cost: 0.013981
    Epoch   70/100, W: 0.863, b: 0.311, cost: 0.013914
    Epoch   71/100, W: 0.863, b: 0.310, cost: 0.013847
    Epoch   72/100, W: 0.864, b: 0.309, cost: 0.013780
    Epoch   73/100, W: 0.864, b: 0.308, cost: 0.013714
    Epoch   74/100, W: 0.865, b: 0.308, cost: 0.013648
    Epoch   75/100, W: 0.865, b: 0.307, cost: 0.013583
    Epoch   76/100, W: 0.865, b: 0.306, cost: 0.013518
    Epoch   77/100, W: 0.866, b: 0.305, cost: 0.013453
    Epoch   78/100, W: 0.866, b: 0.305, cost: 0.013388
    Epoch   79/100, W: 0.866, b: 0.304, cost: 0.013324
    Epoch   80/100, W: 0.867, b: 0.303, cost: 0.013260
    Epoch   81/100, W: 0.867, b: 0.303, cost: 0.013196
    Epoch   82/100, W: 0.867, b: 0.302, cost: 0.013133
    Epoch   83/100, W: 0.867, b: 0.301, cost: 0.013070
    Epoch   84/100, W: 0.868, b: 0.300, cost: 0.013007
    Epoch   85/100, W: 0.868, b: 0.300, cost: 0.012944
    Epoch   86/100, W: 0.868, b: 0.299, cost: 0.012882
    Epoch   87/100, W: 0.869, b: 0.298, cost: 0.012820
    Epoch   88/100, W: 0.869, b: 0.297, cost: 0.012759
    Epoch   89/100, W: 0.869, b: 0.297, cost: 0.012697
    Epoch   90/100, W: 0.870, b: 0.296, cost: 0.012637
    Epoch   91/100, W: 0.870, b: 0.295, cost: 0.012576
    Epoch   92/100, W: 0.870, b: 0.295, cost: 0.012515
    Epoch   93/100, W: 0.871, b: 0.294, cost: 0.012455
    Epoch   94/100, W: 0.871, b: 0.293, cost: 0.012396
    Epoch   95/100, W: 0.871, b: 0.293, cost: 0.012336
    Epoch   96/100, W: 0.872, b: 0.292, cost: 0.012277
    Epoch   97/100, W: 0.872, b: 0.291, cost: 0.012218
    Epoch   98/100, W: 0.872, b: 0.290, cost: 0.012159
    Epoch   99/100, W: 0.873, b: 0.290, cost: 0.012101
    Epoch  100/100, W: 0.873, b: 0.289, cost: 0.012043
    

## Cost Function
![image](https://user-images.githubusercontent.com/100760303/221807071-a45af6d7-978f-44ec-97ab-affeef568503.png)


## Gradient Descent
![image](https://user-images.githubusercontent.com/100760303/221806905-d64d9547-7e22-46aa-9b3e-a0a217864712.png)


## nn.Module
- `nn.Module`을 상속해서 모델 생성
- `nn.Linear(3, 1)` : 입력차원 3, 출력차원 1
- Hypothesis 계산은 `forward()`에서 한다.
- Gradient 계산은 PyTorch가 알아서 해준다. `backward()`


```python
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
```

- 모델의 `__init__`에서는 사용할 레이어들을 정의하게 됩니다.
- 여기서 우리는 linear regression 모델을 만들기 때문에 `nn.Linear` 를 이용할 것입니다.
- `forwar`d에서는 이 모델이 어떻게 입력값에서 출력값을 계산하는지 알려줍니다.


```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```


```python
# 모델 초기화
model = LinearRegressionModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        params = list(model.parameters())
        W = params[0].item()
        b = params[1].item()
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs, W, b, cost.item()))
```

    Epoch    0/1000 W: -0.784, b: 0.665 Cost: 13.291902
    Epoch  100/1000 W: 0.580, b: 0.955 Cost: 0.131521
    Epoch  200/1000 W: 0.670, b: 0.751 Cost: 0.081272
    Epoch  300/1000 W: 0.740, b: 0.590 Cost: 0.050221
    Epoch  400/1000 W: 0.796, b: 0.464 Cost: 0.031034
    Epoch  500/1000 W: 0.840, b: 0.365 Cost: 0.019177
    Epoch  600/1000 W: 0.874, b: 0.287 Cost: 0.011850
    Epoch  700/1000 W: 0.901, b: 0.225 Cost: 0.007323
    Epoch  800/1000 W: 0.922, b: 0.177 Cost: 0.004525
    Epoch  900/1000 W: 0.939, b: 0.139 Cost: 0.002796
    Epoch 1000/1000 W: 0.952, b: 0.109 Cost: 0.001728

