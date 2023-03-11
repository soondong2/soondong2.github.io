---
title: "[PyTorch] Tutorial"
date: 2023-03-11

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - PyTorch
---

## Autograd(자동 미분)
- `torch.autograd` 패키지는 Tensor의 모든 연산에 대해 `자동 미분`을 제공
- 코드를 어떻게 작성하여 실행하느냐에 따라 `역전파`가 정의됨
- `backprop`를 위해 미분값을 자동으로 계산
- `requires_grad=True`로 설정하면 해당 텐서에서 이루어지는 모든 연산들을 추적
- 기록을 추적하는 것을 중단하려면 `.detach()`를 호출하여 연산 기록으로부터 분리
- `with torch.no_grad()`를 사용하여 기울기의 업데이트를 하지 않음
- 기울기 계산은 필요없지만 `requires_grad=True`로 설정되어 학습 가능한 매개변수를 갖는 모델을 평가할 때 유용

## Data Load
- `torch.utils.data`의 `Dataset`과 `DataLoader` 사용 가능
- `Dataset` : 다양한 데이터셋 존재(MNIST, FashionMNIST, CIFAR10)
- `batch_size`, `train` 여부, `transform` 등을 설정하여 데이터를 어떻게 로드할 건지 정의
- `torchvision`은 파이토치에서 제공하는 데이터셋들이 모여있는 패키지


```python
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch
```

- `ToTensor()`를 하는 이유는 torchvision이 PIL 형태로 Image를 받아오기 때문에 Tensor 형태로 바꿔주는 것


```python
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, ), std=(1.0))])
```


```python
trainset = datasets.MNIST(root='./', train=True, download=True, transform=mnist_transform)
testset = datasets.MNIST(root='./', train=False, download=True, transform=mnist_transform)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./MNIST/raw/train-images-idx3-ubyte.gz
    


      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting ./MNIST/raw/train-images-idx3-ubyte.gz to ./MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./MNIST/raw/train-labels-idx1-ubyte.gz
    


      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting ./MNIST/raw/train-labels-idx1-ubyte.gz to ./MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./MNIST/raw/t10k-images-idx3-ubyte.gz
    


      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting ./MNIST/raw/t10k-images-idx3-ubyte.gz to ./MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST/raw/t10k-labels-idx1-ubyte.gz
    


      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting ./MNIST/raw/t10k-labels-idx1-ubyte.gz to ./MNIST/raw
    
    

- `DataLoader`는 데이터 전체를 보관했다가 실제 모델을 학습할 때 지정된 `batch_size` 크기만큼 가져옴


```python
train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)
```


```python
# iter를 통해 next해서 한 번만 가져옴
dataiter = iter(train_loader)

imges, labels = next(dataiter)
print(imges.shape, labels.shape) # batch_size, channels, height, width
```

    torch.Size([8, 1, 28, 28]) torch.Size([8])
    


```python
torch_image = torch.squeeze(imges[0])
torch_image.shape
```




    torch.Size([28, 28])



## 신경망 구성
- `Layer` : 신경망의 핵심 데이터 구조로 하나 이상의 텐서를 입력 받아 하나 이상의 텐서를 출력
- `Module` : 한 개 이상의 계층이 모여서 구성
- `Model` : 한 개 이상의 모듈이 모여서 구성

### torch.nn 패키지
주로 가중치(weights), 편향(bias) 값들이 내부에서 자동으로 생성되는 레이어들을 사용할 때 사용


```python
import torch.nn as nn
```

#### nn.Linear


```python
nn.Linear(in_features=20, out_features=30, bias=True)
```




    Linear(in_features=20, out_features=30, bias=True)



#### nn.Conv2d


```python
nn.Conv2d(in_channels=16, out_channels=33, kernel_size=3, stride=1, padding=0)
```




    Conv2d(16, 33, kernel_size=(3, 3), stride=(1, 1))



## Convolution Layer
- `in_channels` : 입력 채널의 개수
- `out_channels` : 출력 채널의 개수
- `kernel_size` : 커널(필터) 사이즈


```python
layer = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1).to(torch.device('cpu'))
```

- `weight` 확인


```python
weight = layer.weight
weight.shape
```




    torch.Size([20, 1, 5, 5])



weight는 `detach()`를 통해 꺼내줘야 `numpy()` 변환이 가능


```python
weight = weight.detach()
```


```python
# numpy 변환
weight = weight.numpy()
weight.shape
```




    (20, 1, 5, 5)



## Pooling Layer
- `torch.nn.MaxPool2d`

## Linear Layer
1D만 가능하므로 `.view()`를 통해 펼쳐줘야함


```python
a = torch.FloatTensor(1, 28, 28)
a.size()
```




    torch.Size([1, 28, 28])




```python
flatten = a.view(1, 28 * 28)
flatten.shape
```




    torch.Size([1, 784])



## Non-Linear Layer
- `nn.ReLU()`

## 모델 정의

### nn.Module
- `nn.Mudule`을 상속 받는 클래스 정의
- `__init__()` : 모델에서 사용될 모듈과 활성화 함수 정의
- `forward()` : 모델에서 실행되어야 하는 연산 정의


```python
class Model(nn.Module):
    def __init__(self, inputs):
        super(Model, self).__init__()
        self.layer = nn.Linear(inputs, 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer(x)
        x = self.actiavtion(x)
        return x
```


```python
model = Model(1)
print(list(model.children()))
print(list(model.modules()))
```

    [Linear(in_features=1, out_features=1, bias=True), Sigmoid()]
    [Model(
      (layer): Linear(in_features=1, out_features=1, bias=True)
      (activation): Sigmoid()
    ), Linear(in_features=1, out_features=1, bias=True), Sigmoid()]
    

### nn.Sequential
- `nn.Sequential` : 객체로 그 안에 모듈을 순차적으로 실행
- `__init__()` : 사용할 네트워크 모델들을 nn.Sequential로 정의 가능
- `forward()` : 실행되어야 할 계산을 가독성 높게 작성 가능


```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
   
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=30 * 5 * 5, out_features=10, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.layer3(x)

        return x
```


```python
model = Model()
print(list(model.children()))
print(list(model.modules()))
```

    [Sequential(
      (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    ), Sequential(
      (0): Linear(in_features=750, out_features=10, bias=True)
      (1): ReLU()
    )]
    [Model(
      (layer1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (layer2): Sequential(
        (0): Linear(in_features=750, out_features=10, bias=True)
        (1): ReLU()
      )
    ), Sequential(
      (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    ), Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1)), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Sequential(
      (0): Linear(in_features=750, out_features=10, bias=True)
      (1): ReLU()
    ), Linear(in_features=750, out_features=10, bias=True), ReLU()]
    

## PyTorch 사전 학습 모델
https://pytorch.org/vision/stable/models.html

## Model Parameter

### 손실 함수(Loss Function)
- 예측값과 실제값 사이의 오차 측정
- 학습이 진행되면서 해당 과정이 얼마나 잘 되고 있는지 나타내는 지표
- 모델이 훈련되는 동안 최소화될 값으로 주어진 문제에 대한 성공 지표
- 손실 함수에 따른 결과를 통해 학습 파라미터 조정
- 최적화 이론에서 최소화 하고자 하는 함수
- 미분 가능한 함수 사용

<br>

- `torch.nn.BCELoss` : 이진 분류
- `torch.nn.CrossEntropyLoss` : 다중 클래스 분류
- `torch.nn.MSELoss` : 회귀


```python
criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
```

### 옵티마이저(Optimizer)
- 손실 함수를 기반으로 모델이 어떻게 업데이트 되어야 하는지를 결정
- 특정 종류의 확률적 경사 하강법 구현
- `step()` : 전달받은 파라미터를 모델 업데이트
- `zero_grad()` : 옵티마이저에 사용된 파라미터들의 기울이를 0으로 설정
- `torch.optim.lr_scheduler` : epochs에 따라 학습률 조절

<br>

- `optim.Adagrad`
- `optim.Adam`
- `optim.ReLU`
- `optim.SGD`
- `optim.RMSprop`

### 학습률 스케줄러(Learning rate scheduler)
- `optim.lr_scheduler.LambdaLR` : lambda 함수를 이용해 그 결과를 합습률로 설정
- `optim.lr_scheduler.StepLR` : 단계마다 학습률을 `gamma` 비율만큼 감소
- `optim.lr_scheduler.MultiStepLR` : 특정 단계가 아닌 지정된 epoch에만 감마 비율로 감소
- `optim.lr_scheduler.ExponentialLR` : epochs마다 이전 학습률에 감마만큼 곱함
- `optim.lr_scheduler.CosineAnnealingLR` : 학습률을 코사인 함수 형태로 변화시켜 학습률이 키지기도, 작아지기도 함
- `optim.lr_scheduler..ReduceLROnPlateau` : 학습이 잘 되는지 아닌지에 따라 동적으로 학습률 번화

## 지표(Metrics)
- 모델의 학습과 테스트 단계를 모니터링


```python
# !pip install torchmetrics
```
