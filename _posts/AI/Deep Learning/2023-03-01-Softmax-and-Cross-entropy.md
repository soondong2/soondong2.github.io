---
title: "[PyTorch] Softmax & Cross Entropy"
date: 2023-03-01

categories:
  - AI
  - Deep Learning
tags:
    - DL
    - PyTorch
---

## 학습 목표
- 소프트맥스(Softmax)
- 크로스 엔트로피(Cross Entropy)


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

## Softmax


```python
z = torch.FloatTensor([1, 2, 3])
```


```python
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
```

    tensor([0.0900, 0.2447, 0.6652])
    

모든 확률을 더하면 1이 된다.


```python
hypothesis.sum()
```




    tensor(1.)



## Cross Entropy


```python
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
```

    tensor([[0.1703, 0.2118, 0.1857, 0.1963, 0.2359],
            [0.2328, 0.2268, 0.1060, 0.2356, 0.1987],
            [0.3189, 0.2760, 0.1258, 0.1234, 0.1559]], grad_fn=<SoftmaxBackward0>)
    


```python
y = torch.randint(5, (3,)).long()
print(y)
```

    tensor([2, 1, 0])
    

`F.null_loss(F.log_softmax)`


```python
# Cross Entropy 구하는 방법1
F.nll_loss(F.log_softmax(z, dim=1), y)
```




    tensor(1.4368, grad_fn=<NllLossBackward0>)



`F.cross_entropy`


```python
# Cross Entropy 구하는 방법1
F.cross_entropy(z, y)
```




    tensor(1.4368, grad_fn=<NllLossBackward0>)



## nn.Module


```python
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
```


```python
print(x_train.dim())
print(x_train.shape)

print(y_train.dim())
print(y_train.shape)
```

    2
    torch.Size([8, 4])
    1
    torch.Size([8])
    


```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output이 3
    
    def forward(self, x):
        return self.linear(x)
```


```python
model = SoftmaxClassifierModel()
```


```python
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # H(x)
    prediction = model(x_train)

    # Cost
    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
```

    Epoch    0/1000 Cost: 1.777960
    Epoch  100/1000 Cost: 0.654127
    Epoch  200/1000 Cost: 0.561501
    Epoch  300/1000 Cost: 0.505037
    Epoch  400/1000 Cost: 0.460010
    Epoch  500/1000 Cost: 0.420253
    Epoch  600/1000 Cost: 0.383131
    Epoch  700/1000 Cost: 0.347032
    Epoch  800/1000 Cost: 0.310779
    Epoch  900/1000 Cost: 0.274060
    Epoch 1000/1000 Cost: 0.244281
