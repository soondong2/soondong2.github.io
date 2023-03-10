---
title: "[PyTorch] 활성화 함수, 손실 함수, 옵티마이저, 가중치 초기화 "
date: 2023-03-07

categories:
  - AI
  - Deep Learning
tags:
    - DL
    - PyTorch
---



```python
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
```


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed 고정
random.seed(111)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
```


```python
# parameters
learning_rate = 0.001
epochs = 15
batch_size = 100
```


```python
mnist_train = datasets.MNIST(root='./', train=True,
                             transform=transforms.ToTensor(),
                             download=True)

mnist_test = datasets.MNIST(root='./', train=False,
                            transform=transforms.ToTensor(),
                            download=True)
```


```python
# Dataset Loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
```

## Activation Function
- `torch.nn.relu` : ReLU
- `torch.nn.sigmoid` : Sigmoid
- `torch.nn.Leaky_relu` : Leaky ReLU


```python
# 784인 이유는 28 x 28
# nn.Linear(input_dim, output_dim)
linear1 = torch.nn.Linear(784, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 10, bias=True)

# activation function : ReLU
relu = torch.nn.ReLU()
```

## 가중치 초기화
- `torch.nn.init.normal_` : 기본적인 방법
- `torch.nn.init.xavier_uniform_` : Xvier


```python
# Initialization
torch.nn.init.normal_(linear1.weight)
torch.nn.init.normal_(linear2.weight)
torch.nn.init.normal_(linear3.weight)
```




    Parameter containing:
    tensor([[-2.3518, -1.0384,  0.3364,  ..., -0.0419, -2.8471, -0.7893],
            [-1.3143,  0.4450,  0.0218,  ..., -0.1175,  0.4807, -0.0046],
            [-1.0058, -1.2789,  0.4103,  ..., -0.0894, -0.4732, -1.7324],
            ...,
            [-0.3238, -1.3617,  0.9380,  ..., -0.4760, -0.0939, -0.4173],
            [ 0.4770, -0.8255,  1.8754,  ...,  1.0181, -0.2816, -1.3303],
            [ 0.8517,  0.9480,  0.4240,  ...,  0.5320,  0.6457, -1.3825]],
           requires_grad=True)




```python
# model
model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3).to(device)
```

## Loss & Optimizer
### Loss
- `torch.nn.BCELoss` : 이진 분류
- `torch.nn.CrossEntropyLoss` : 다중 클래스 분류
- `torch.nn.MSELoss` : MSE

### Optimizer
- `torch.optim.Adam` : Adam
- `torch.optim.SGD` : SGD
- `torch.optim.RMSprop` : RMSprop
- `torch.optim.Rprop` : Rprop


```python
# loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```


```python
total_batch = len(data_loader)

for epoch in range(epochs):
    avg_cost = 0

    for X, y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        y = y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)

        cost = criterion(hypothesis, y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    print('Epoch: {} Cost: {:.4f}'.format(epoch + 1, avg_cost))
```

    Epoch: 1 Cost: 164.3944
    Epoch: 2 Cost: 38.7940
    Epoch: 3 Cost: 24.0848
    Epoch: 4 Cost: 16.4266
    Epoch: 5 Cost: 11.6766
    Epoch: 6 Cost: 8.5664
    Epoch: 7 Cost: 6.3255
    Epoch: 8 Cost: 4.5684
    Epoch: 9 Cost: 3.4953
    Epoch: 10 Cost: 2.5707
    Epoch: 11 Cost: 1.9859
    Epoch: 12 Cost: 1.4931
    Epoch: 13 Cost: 1.1411
    Epoch: 14 Cost: 0.8788
    Epoch: 15 Cost: 0.6823
    


```python
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    y_test = mnist_test.test_labels.to(device)

    predict = model(X_test)
    correct = torch.argmax(predict, 1) == y_test
    accuracy = correct.float().mean()
    print('Accuracy: {}'.format(accuracy.item()))

    # 하나의 예측
    r = random.randint(0, len(mnist_test) - 1)
    X_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    y_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: {}'.format(y_data.item()))

    single_predict = model(X_data)
    print('Predict: {}'.format(torch.argmax(single_predict, 1).item()))
```

    c:\Users\USER\anaconda3\lib\site-packages\torchvision\datasets\mnist.py:80: UserWarning: test_data has been renamed data
      warnings.warn("test_data has been renamed data")
    c:\Users\USER\anaconda3\lib\site-packages\torchvision\datasets\mnist.py:70: UserWarning: test_labels has been renamed targets
      warnings.warn("test_labels has been renamed targets")
    

    Accuracy: 0.9441999793052673
    Label: 8
    Predict: 8
    
