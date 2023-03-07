---
title: "[PyTorch] MNIST 딥러닝 예제"
date: 2023-03-03

categories:
  - AI
  - Deep Learning
tags:
    - DL
    - PyTorch
---

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

from torch.utils.data import DataLoader
```


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```


```python
# seed 고정
random.seed(1)
torch.manual_seed(1)
if device == 'cuda':
    torch.cuda.manual_seed_all(1)
```

60,000개의 이미지 데이터가 있는데 batch_size = 100으로 설정하였다. 따라서 600개의 batch가 생긴다. -> total_batch = 600


```python
# parameters
train_epochs = 15
batch_size = 100
```


```python
# MNIST Dataset
mnist_train = datasets.MNIST(root='./', train=True,
                             transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='./', train=False,
                            transform=transforms.ToTensor(), download=True)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./MNIST\raw\train-images-idx3-ubyte.gz
    

    100%|██████████| 9912422/9912422 [00:01<00:00, 9066483.55it/s] 
    

    Extracting ./MNIST\raw\train-images-idx3-ubyte.gz to ./MNIST\raw
    

      0%|          | 0/28881 [00:00<?, ?it/s]

    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./MNIST\raw\train-labels-idx1-ubyte.gz
    

    100%|██████████| 28881/28881 [00:00<00:00, 14446713.63it/s]
      0%|          | 0/1648877 [00:00<?, ?it/s]

    Extracting ./MNIST\raw\train-labels-idx1-ubyte.gz to ./MNIST\raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./MNIST\raw\t10k-images-idx3-ubyte.gz
    

    100%|██████████| 1648877/1648877 [00:00<00:00, 4511992.19it/s]
    

    Extracting ./MNIST\raw\t10k-images-idx3-ubyte.gz to ./MNIST\raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST\raw\t10k-labels-idx1-ubyte.gz
    

    100%|██████████| 4542/4542 [00:00<00:00, 4548836.86it/s]
    

    Extracting ./MNIST\raw\t10k-labels-idx1-ubyte.gz to ./MNIST\raw
    
    


```python
# DataLoader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size,
                                          shuffle=True, drop_last=True)
```

- 28 * 28 이미지 이므로 28 x 28 = 784개의 입력
- 0 ~ 9까지의 출력 데이터이므로 10개의 출력


```python
# MNIST data image of shape 28 * 28 = 784
linear = torch.nn.Linear(784, 10, bias=True).to(device)
```

`torch.nn.CrossEntropyLoss()`를 사용하면 내부적으로 `Softmax`가 계산된다.


```python
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
```


```python
from tqdm.notebook import tqdm

for epoch in tqdm(range(train_epochs)):
    avg_cost = 0
    total_batch = len(data_loader)

    # X : Image / y : Label
    for X, y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        y = y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X) # 분류 결과를 얻음

        cost = criterion(hypothesis, y) # 분류 결과와 실제 정답을 비교하여 cost 계산
        cost.backward() # cost를 이용해 gradient 계산
        optimizer.step() # gradient를 이용해 업데이트

        avg_cost += cost / total_batch
    print('Epoch {:4d}/{}, Cost: {:.4f}'.format(epoch, train_epochs, avg_cost))
```


      0%|          | 0/15 [00:00<?, ?it/s]


    Epoch    0/15, Cost: 0.3310
    Epoch    1/15, Cost: 0.3160
    Epoch    2/15, Cost: 0.3071
    Epoch    3/15, Cost: 0.3001
    Epoch    4/15, Cost: 0.2950
    Epoch    5/15, Cost: 0.2907
    Epoch    6/15, Cost: 0.2874
    Epoch    7/15, Cost: 0.2844
    Epoch    8/15, Cost: 0.2819
    Epoch    9/15, Cost: 0.2795
    Epoch   10/15, Cost: 0.2777
    Epoch   11/15, Cost: 0.2759
    Epoch   12/15, Cost: 0.2745
    Epoch   13/15, Cost: 0.2729
    Epoch   14/15, Cost: 0.2717
    


```python
import matplotlib.pyplot as plt
```

`with torch.no_grad()` : Gradient Update를 하지 않는다.
```python
# 모델을 이용하여 test
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct = torch.argmax(prediction, 1) == y_test # True / False
    accuracy = correct.float().mean() # True / False -> 1.0 / 0.0 -> mean
    print('Accuracy: ', accuracy.item())

    # 하나의 예측
    r = random.randint(0, len(mnist_test) -1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', y_single_data.item())
    
    single_pred = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_pred, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='gray', interpolation='nearest')
    plt.show()
```

    Accuracy:  0.8826000094413757
    Label:  0
    Prediction:  0
    


![image](https://user-images.githubusercontent.com/100760303/222608866-d6cd8a4a-106d-46c5-9c18-b0048b03e1a5.png)
   
