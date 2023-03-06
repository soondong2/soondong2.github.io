---
title: "[PyTorch] Multi Layer Perceptron"
date: 2023-03-06

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - PyTorch
---

## 학습 목표
- 다중 퍼셉트론(Multi Layer Perceptron)
- 오차역전파(Backpropagation)


```python
import torch
```


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed 고정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
```


```python
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)
```

## nn Layers
두 개의 레이어를 가지는 다중 퍼셉트론


```python
# nn Layers
linear1 = torch.nn.Linear(2, 2, bias=True)
linear2 = torch.nn.Linear(2, 1, bias=True)
sigmoid = torch.nn.Sigmoid()
```


```python
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)
```


```python
# Binary Cross-Entropy Loss
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    # 100번에 한 번씩 로그 출력
    if step % 100 == 0:
        print(step, cost.item())
```

    0 0.7434073090553284
    100 0.693165123462677
    200 0.6931577920913696
    300 0.6931517124176025
    400 0.6931463479995728
    500 0.6931411027908325
    600 0.6931357383728027
    700 0.6931294798851013
    800 0.6931220889091492
    900 0.6931126117706299
    1000 0.6930999755859375
    1100 0.693082332611084
    1200 0.6930569410324097
    1300 0.6930190324783325
    1400 0.6929606199264526
    1500 0.6928660273551941
    1600 0.6927032470703125
    1700 0.6923960447311401
    1800 0.6917302012443542
    1900 0.6899653673171997
    2000 0.6838315725326538
    2100 0.6561667919158936
    2200 0.4311023950576782
    2300 0.1348935216665268
    2400 0.06630441546440125
    2500 0.042168211191892624
    2600 0.030453883111476898
    2700 0.023665912449359894
    2800 0.01927776448428631
    2900 0.01622403785586357
    3000 0.01398380845785141
    3100 0.012273931875824928
    3200 0.010928118601441383
    3300 0.009842472150921822
    3400 0.008949032984673977
    3500 0.008201336488127708
    3600 0.007566767744719982
    3700 0.007021686062216759
    3800 0.006548595614731312
    3900 0.006134224124252796
    4000 0.005768375005573034
    4100 0.0054430365562438965
    4200 0.005151905119419098
    4300 0.004889918025583029
    4400 0.0046528722159564495
    4500 0.004437457304447889
    4600 0.004240859299898148
    4700 0.004060701932758093
    4800 0.003895031288266182
    4900 0.0037421947345137596
    5000 0.0036007347516715527
    5100 0.003469479735940695
    5200 0.0033473046496510506
    5300 0.0032333978451788425
    5400 0.0031268750317394733
    5500 0.0030270610004663467
    5600 0.0029333550482988358
    5700 0.0028452035039663315
    5800 0.00276215560734272
    5900 0.002683777129277587
    6000 0.0026096487417817116
    6100 0.0025394847616553307
    6200 0.0024729417636990547
    6300 0.0024097643326967955
    6400 0.002349698217585683
    6500 0.0022925634402781725
    6600 0.002238075714558363
    6700 0.002186085097491741
    6800 0.0021364721469581127
    6900 0.002089011948555708
    7000 0.00204361486248672
    7100 0.0020001311786472797
    7200 0.0019584265537559986
    7300 0.0019184107659384608
    7400 0.0018799942918121815
    7500 0.0018430722411721945
    7600 0.0018075549742206931
    7700 0.0017733527347445488
    7800 0.0017404207028448582
    7900 0.0017087137093767524
    8000 0.001678097527474165
    8100 0.0016485571395605803
    8200 0.0016200175741687417
    8300 0.0015924491453915834
    8400 0.0015657918993383646
    8500 0.0015400308184325695
    8600 0.0015150614781305194
    8700 0.0014909137971699238
    8800 0.0014674977865070105
    8900 0.001444813678972423
    9000 0.0014228165382519364
    9100 0.0014014765620231628
    9200 0.0013806892093271017
    9300 0.0013606036081910133
    9400 0.0013410557294264436
    9500 0.001322030322626233
    9600 0.001303557539358735
    9700 0.001285637030377984
    9800 0.0012681199004873633
    9900 0.0012511102249845862
    10000 0.0012345188297331333
    


```python
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('\nHypothesis: ', hypothesis.detach().cpu().numpy(), 
          '\nCorrect: ', predicted.detach().cpu().numpy(),
          '\nAccuracy: ', accuracy.item())
```

    
    Hypothesis:  [[0.00106364]
     [0.99889404]
     [0.99889404]
     [0.00165862]] 
    Correct:  [[0.]
     [1.]
     [1.]
     [0.]] 
    Accuracy:  1.0
    

## nn Wide Deep Layers
네 개의 레이어를 가지는 다중 퍼셉트론


```python
# nn wide deep Layers
linear1 = torch.nn.Linear(2, 10, bias=True)
linear2 = torch.nn.Linear(10, 10, bias=True)
linear3 = torch.nn.Linear(10, 10, bias=True)
linear4 = torch.nn.Linear(10, 1, bias=True)
sigmoid = torch.nn.Sigmoid()
```


```python
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid, linear3, sigmoid, linear4, sigmoid).to(device)
```


```python
# Binary Cross-Entropy Loss
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    # 1000번에 한 번씩 로그 출력
    if step % 1000 == 0:
        print(step, cost.item())
```

    0 0.6978083848953247
    1000 0.6929439902305603
    2000 0.6797102689743042
    3000 0.002333962358534336
    4000 0.0006081871688365936
    5000 0.00033300856011919677
    6000 0.0002254210994578898
    7000 0.0001689193450147286
    8000 0.00013438795576803386
    9000 0.00011119883856736124
    10000 9.461208537686616e-05
    


```python
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('\nHypothesis: ', hypothesis.detach().cpu().numpy(), 
          '\nCorrect: ', predicted.detach().cpu().numpy(),
          '\nAccuracy: ', accuracy.item())
```

    
    Hypothesis:  [[7.4425239e-05]
     [9.9988043e-01]
     [9.9990392e-01]
     [8.8329281e-05]] 
    Correct:  [[0.]
     [1.]
     [1.]
     [0.]] 
    Accuracy:  1.0
