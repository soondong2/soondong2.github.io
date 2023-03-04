---
title: "[PyTorch] Minibatch Gradient Descent & Custom Dataset"
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
- 미니배치 경사하강법(Minibatch Gradient descent)
- Dataset, DataLoader

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


## Minibatch Gradient Descent
- 전체 데이터를 균일하게 나누어 학습한다.
- 업데이트를 좀 더 빠르게 할 수 있다.
- 전체 데이터를 쓰지 않아서 잘못된 방향으로 업데이트 할 수도 있다.


```python
from torch.utils.data import Dataset
```


```python
class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [
            [73, 80, 75],
            [93, 88, 93],
            [89, 91, 90],
            [96, 98, 100],
            [73, 66, 70]
        ]

        self.y_data = [[152], [185], [180], [196], [142]]
    
    # 데이터 셋의 총 데이터 수
    def __len__(self):
        return len(self.x_data)
    
    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y
```

- `batch_size=2` : 각 Minibatch의 크기이다. 통상적으로 2의 제곱수로 설정한다.
- `shuffle=True` : Epoch마다 데이터 셋을 섞어서 데이터가 학습되는 순서를 바꾼다. 


```python
from torch.utils.data import DataLoader

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```


```python
model = nn.Linear(3 ,1)
optimizer = optim.SGD(model.parameters(), lr=1e-5) 
```


```python
nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        # H(x)
        prediction = model(x_train)

        # cost
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()))
```

    Epoch    0/20 Batch 1/3 Cost: 31655.191406
    Epoch    0/20 Batch 2/3 Cost: 10922.980469
    Epoch    0/20 Batch 3/3 Cost: 3584.120361
    Epoch    1/20 Batch 1/3 Cost: 756.364258
    Epoch    1/20 Batch 2/3 Cost: 414.346191
    Epoch    1/20 Batch 3/3 Cost: 215.288986
    Epoch    2/20 Batch 1/3 Cost: 35.517097
    Epoch    2/20 Batch 2/3 Cost: 16.300579
    Epoch    2/20 Batch 3/3 Cost: 1.422832
    Epoch    3/20 Batch 1/3 Cost: 0.155849
    Epoch    3/20 Batch 2/3 Cost: 16.210571
    Epoch    3/20 Batch 3/3 Cost: 35.381222
    Epoch    4/20 Batch 1/3 Cost: 20.472446
    Epoch    4/20 Batch 2/3 Cost: 10.149340
    Epoch    4/20 Batch 3/3 Cost: 15.655682
    Epoch    5/20 Batch 1/3 Cost: 12.491540
    Epoch    5/20 Batch 2/3 Cost: 20.897430
    Epoch    5/20 Batch 3/3 Cost: 11.950432
    Epoch    6/20 Batch 1/3 Cost: 15.870499
    Epoch    6/20 Batch 2/3 Cost: 7.127813
    Epoch    6/20 Batch 3/3 Cost: 21.390060
    Epoch    7/20 Batch 1/3 Cost: 21.590193
    Epoch    7/20 Batch 2/3 Cost: 2.824972
    Epoch    7/20 Batch 3/3 Cost: 21.442314
    Epoch    8/20 Batch 1/3 Cost: 14.949982
    Epoch    8/20 Batch 2/3 Cost: 23.845438
    Epoch    8/20 Batch 3/3 Cost: 14.746861
    Epoch    9/20 Batch 1/3 Cost: 21.042770
    Epoch    9/20 Batch 2/3 Cost: 8.507879
    Epoch    9/20 Batch 3/3 Cost: 3.986157
    Epoch   10/20 Batch 1/3 Cost: 9.826711
    Epoch   10/20 Batch 2/3 Cost: 15.381631
    Epoch   10/20 Batch 3/3 Cost: 16.729069
    Epoch   11/20 Batch 1/3 Cost: 29.537363
    Epoch   11/20 Batch 2/3 Cost: 14.819937
    Epoch   11/20 Batch 3/3 Cost: 2.593527
    Epoch   12/20 Batch 1/3 Cost: 11.980485
    Epoch   12/20 Batch 2/3 Cost: 6.603844
    Epoch   12/20 Batch 3/3 Cost: 39.579948
    Epoch   13/20 Batch 1/3 Cost: 13.779463
    Epoch   13/20 Batch 2/3 Cost: 17.290298
    Epoch   13/20 Batch 3/3 Cost: 11.465834
    Epoch   14/20 Batch 1/3 Cost: 9.671873
    Epoch   14/20 Batch 2/3 Cost: 19.929955
    Epoch   14/20 Batch 3/3 Cost: 7.452543
    Epoch   15/20 Batch 1/3 Cost: 16.040655
    Epoch   15/20 Batch 2/3 Cost: 10.869001
    Epoch   15/20 Batch 3/3 Cost: 21.206408
    Epoch   16/20 Batch 1/3 Cost: 15.156000
    Epoch   16/20 Batch 2/3 Cost: 13.026572
    Epoch   16/20 Batch 3/3 Cost: 6.086367
    Epoch   17/20 Batch 1/3 Cost: 12.933932
    Epoch   17/20 Batch 2/3 Cost: 15.502781
    Epoch   17/20 Batch 3/3 Cost: 7.013656
    Epoch   18/20 Batch 1/3 Cost: 27.403336
    Epoch   18/20 Batch 2/3 Cost: 15.444098
    Epoch   18/20 Batch 3/3 Cost: 2.836492
    Epoch   19/20 Batch 1/3 Cost: 14.594184
    Epoch   19/20 Batch 2/3 Cost: 16.887869
    Epoch   19/20 Batch 3/3 Cost: 5.393626
    Epoch   20/20 Batch 1/3 Cost: 18.676640
    Epoch   20/20 Batch 2/3 Cost: 12.567435
    Epoch   20/20 Batch 3/3 Cost: 3.024912
    


```python
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 

# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
```

    훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[154.8866]], grad_fn=<AddmmBackward0>)
