---
title: "[PyTorch] Overfitting 해결 방법"
date: 2023-03-02

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - PyTorch
---

## 학습 목표
- 최대 가능도 추정(Maximum Likelihood Estimation)
- 과적합(Overfitting)과 정규화(Regurlarization)
- 훈련 세트와 테스트 세트
- 학습률(Learning Rate)
- 데이터 전처리(Preprocessing)


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




    <torch._C.Generator at 0x1b5a29ad470>



## Train and Test Dataset


```python
x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]
                            ])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0]) # [0, 1, 2] -> 3개의 output
```


```python
print(x_train.dim(), x_train.shape)
print(y_train.dim(), y_train.shape)
```

    2 torch.Size([8, 3])
    1 torch.Size([8])
    


```python
x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])
```


```python
print(x_test.dim(), x_test.shape)
print(y_test.dim(), y_test.shape)
```

    2 torch.Size([3, 3])
    1 torch.Size([3])
    

## Model


```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)
```


```python
model = SoftmaxClassifierModel()
```


```python
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```

## Training


```python
def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        # H(x)
        prediction = model(x_train)
        # Cost
        cost = F.cross_entropy(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/20 Cost: 0.983424
    Epoch    1/20 Cost: 0.977591
    Epoch    2/20 Cost: 0.971864
    Epoch    3/20 Cost: 0.966240
    Epoch    4/20 Cost: 0.960718
    Epoch    5/20 Cost: 0.955295
    Epoch    6/20 Cost: 0.949968
    Epoch    7/20 Cost: 0.944736
    Epoch    8/20 Cost: 0.939596
    Epoch    9/20 Cost: 0.934546
    Epoch   10/20 Cost: 0.929585
    Epoch   11/20 Cost: 0.924709
    Epoch   12/20 Cost: 0.919918
    Epoch   13/20 Cost: 0.915210
    Epoch   14/20 Cost: 0.910582
    Epoch   15/20 Cost: 0.906032
    Epoch   16/20 Cost: 0.901561
    Epoch   17/20 Cost: 0.897164
    Epoch   18/20 Cost: 0.892841
    Epoch   19/20 Cost: 0.888590
    

## Test


```python
def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()

    cost = F.cross_entropy(prediction, y_test)

    print('Accuracy: {:.4f} Cost: {:.6f}'.format(correct_count / len(y_test), cost.item()))
```


```python
test(model, optimizer, x_test, y_test)
```

    Accuracy: 0.6667 Cost: 0.791733
    

## Data Preprocessing


```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```


```python
mu = x_train.mean(dim=0)
```


```python
sigma = x_train.std(dim=0)
```


```python
norm_x_train = (x_train - mu) / sigma
```


```python
print(norm_x_train)
```

    tensor([[-1.0674, -0.3758, -0.8398],
            [ 0.7418,  0.2778,  0.5863],
            [ 0.3799,  0.5229,  0.3486],
            [ 1.0132,  1.0948,  1.1409],
            [-1.0674, -1.5197, -1.2360]])
    


```python
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)
```


```python
model = MultivariateLinearRegressionModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```


```python
def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        # H(x)
        prediction = model(x_train)
        # Cost
        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
```


```python
# x_train 대신 정규화한 norm_x_train 입력
train(model, optimizer, norm_x_train, y_train)
```

    Epoch    0/20 Cost: 29770.843750
    Epoch    1/20 Cost: 18898.781250
    Epoch    2/20 Cost: 12050.423828
    Epoch    3/20 Cost: 7699.415527
    Epoch    4/20 Cost: 4924.062012
    Epoch    5/20 Cost: 3150.534668
    Epoch    6/20 Cost: 2016.257812
    Epoch    7/20 Cost: 1290.541992
    Epoch    8/20 Cost: 826.142700
    Epoch    9/20 Cost: 528.939758
    Epoch   10/20 Cost: 338.729218
    Epoch   11/20 Cost: 216.989227
    Epoch   12/20 Cost: 139.070099
    Epoch   13/20 Cost: 89.196045
    Epoch   14/20 Cost: 57.270782
    Epoch   15/20 Cost: 36.833179
    Epoch   16/20 Cost: 23.747566
    Epoch   17/20 Cost: 15.367709
    Epoch   18/20 Cost: 9.999623
    Epoch   19/20 Cost: 6.559284
    

## Regularization
- `L1` 정규화
- `L2` 정규화
- `L1 + L2` 정규화


```python
def train_with_regularization(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        # H(x) 
        prediction = model(x_train)
        # Cost 
        cost = F.mse_loss(prediction, y_train)
        
        # L2 norm 계산
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param)
            
        cost += l2_reg

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch + 1, nb_epochs, cost.item()))
```


```python
model = MultivariateLinearRegressionModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```


```python
train_with_regularization(model, optimizer, norm_x_train, y_train)
```

    Epoch    1/20 Cost: 29548.414062
    Epoch    2/20 Cost: 18835.623047
    Epoch    3/20 Cost: 12081.613281
    Epoch    4/20 Cost: 7787.242188
    Epoch    5/20 Cost: 5047.043457
    Epoch    6/20 Cost: 3295.697754
    Epoch    7/20 Cost: 2175.522217
    Epoch    8/20 Cost: 1458.805176
    Epoch    9/20 Cost: 1000.157837
    Epoch   10/20 Cost: 706.633972
    Epoch   11/20 Cost: 518.776978
    Epoch   12/20 Cost: 398.543579
    Epoch   13/20 Cost: 321.588684
    Epoch   14/20 Cost: 272.332123
    Epoch   15/20 Cost: 240.802841
    Epoch   16/20 Cost: 220.618927
    Epoch   17/20 Cost: 207.696640
    Epoch   18/20 Cost: 199.421722
    Epoch   19/20 Cost: 194.121582
    Epoch   20/20 Cost: 190.725433
