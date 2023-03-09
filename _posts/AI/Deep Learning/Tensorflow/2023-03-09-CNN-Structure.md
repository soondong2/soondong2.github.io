---
title: "[Tensorflow] CNN 개념"
date: 2023-03-09

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## 컨볼루션 신경망(Convolution Neural Networks, CNN)
- 완전 연결 네트워크의 문제점으로부터 시작
    - 매개변수의 폭발적인 증가와 공간 추론의 부족의 문제점
- 동물의 시각피질의 구조에서 영감을 받아 만들어진 딥러닝 모델
- 영상 분류, 문자 인식 등 인식 문제에 높은 성능

<img src="https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png">

## 컨볼루션 연산(Convolution Operation)
### 필터(filter) 연산
- 입력 데이터에 필터를 통한 연산을 수행
- 필터에 대응하는 원소끼리 곱하고 그 합을 구함
- 연산이 완료된 결과 데이터를 `특징 맵(feature map)`이라고 함

### 필터(filter)
- `커널(kernel)`이라고도 함
- 필터 사이즈는 거의 항상 `홀수`
    - 짝수이면 패딩이 비대칭이 됨
    - 왼쪽, 오른쪽을 다르게 주어야 함
- 필터의 학습 파라미터 개수는 입력 데이터의 크기와 상관없이 일정
- 아래 이미지는 3 x 3의 컨볼루션 연산 수행

<img src="https://theano-pymc.readthedocs.io/en/latest/_images/numerical_padding_strides.gif">



- 일반적으로 합성곱 연산을 한 후의 데이터 사이즈는 다음과 같다.
- n : 입력 데이터의 크기
- f : 필터(커널의 크기)
- (n - f + 1) x (n - f + 1)

<img src="https://miro.medium.com/max/1400/1*Fw-ehcNBR9byHtho-Rxbtw.gif" width="400">

- 위 이미지의 경우 (5 - 3 + 1) = 3이므로 출력 데이터 크기는 3


```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.datasets import load_sample_image
from tensorflow.keras.layers import Conv2D
```


```python
# 255 : nomarization
flower = load_sample_image('flower.jpg') / 255

print(flower.dtype)
print(flower.shape)
plt.imshow(flower)
plt.show()
```

    float64
    (427, 640, 3)
    
![image](https://user-images.githubusercontent.com/100760303/223910596-1754e505-b42f-4e8d-a714-454dbe187fec.png)
    



```python
china = load_sample_image('china.jpg') / 255

print(china.dtype)
print(china.shape) # height, width, channel
plt.imshow(china)
plt.show()
```

    float64
    (427, 640, 3)
    


![image](https://user-images.githubusercontent.com/100760303/223910617-6e92fedb-6439-4250-beed-854c89d390d2.png)
    



```python
images = np.array([china, flower])
batch_size, height, width, channels = images.shape

# 이미지 두개, 높이, 너비, 채널
print(batch_size, height, width, channels)
```

    2 427 640 3
    

## 패딩과 스트라이드
- 필터(커널) 사이즈와 함께 입력 이미지와 출력 이미지의 사이즈를 결정하기 위해 사용
- 사용자가 결정할 수 있음

### 패딩(Padding) 
- **입력 데이터의 주변을 특정 값으로 채우는 기법**
- 주로 `0`으로 채움 
- 출력 데이터의 크기 : (n + 2p - f + 1) x (n + 2p - f + 1)
- 아래 이미지에서는 (5 + 2(1) - 3 + 1) = 5

#### valid
- `padding=0` : 패딩을 주지 않음

#### same
- 패딩을 주어 **입력 이미지의 크기와 연산 후의 이미지 크기를 같도록 유지**

<img src="https://miro.medium.com/max/395/1*1okwhewf5KCtIPaFib4XaA.gif" width="300">

### Stride
- 필터를 적용하는 `간격`을 의미

### 출력 데이터의 크기

  $\qquad OH = \frac{H + 2P - FH}{S} + 1 $ 
  
  $\qquad OW = \frac{W + 2P - FW}{S} + 1 $ 

  - 입력 크기 : $(H, W)$
  - 필터 크기 : $(FH, FW)$
  - 출력 크기 : $(OH, OW)$
  - 패딩, 스트라이드 : $P, S$
- 위 식의 값에서 $\frac{H + 2P - FH}{S}$ 또는 $\frac{W + 2P - FW}{S}$가 정수로 나누어 떨어지는 값이어야 함
- 정수로 나누어 떨어지지 않으면, 패딩, 스트라이드 값을 조정하여 정수로 나누어 떨어지게 해야함
  
  


```python
conv = Conv2D(filters=16, kernel_size=3, padding='same', strides=1, activation='relu')
```

## Pooling
### Max Pooling
- 가장 많이 사용되는 방법
- 모델이 물체의 주요한 특징을 학습할 수 있도록 해줌
- 일반적으로 `stride=2`, `kernel_size=2`를 통해 특징맵의 크기를 절반으로 줄이는 역할
- 출력 데이터의 사이즈 계산은 컨볼루션 연산과 동일

  $\quad OH = \frac{H + 2P - FH}{S} + 1 $ 
  
  $\quad OW = \frac{W + 2P - FW}{S} + 1 $ 

<img src="https://cs231n.github.io/assets/cnn/maxpool.jpeg" width="600">


```python
from tensorflow.keras.layers import MaxPool2D
```


```python
print(flower.shape) # 원본 이미지 크기(height, width, channel)
flower = np.expand_dims(flower, axis=0)
print(flower.shape) # batch size = 1 추가

output = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(flower)
output = MaxPool2D(pool_size=2)(output)

print(output.shape) # MaxPooling 후 사이즈
```

    (427, 640, 3)
    (1, 427, 640, 3)
    (1, 213, 320, 32)
    


```python
# 임의의 이미지 출력
plt.imshow(output[0, :, :, 4], cmap='gray')
plt.show()
```

![image](https://user-images.githubusercontent.com/100760303/223910661-0c8cb3b8-1f79-4524-b09e-ec02138cc487.png)
    


## 완전 연결 계층(Fully-Connected Layer)
- 입력 받은 텐서를 1차원으로 `평면화(flatten)`
- `밀집 계층(Dense Layer)` 라고도 함
- 일반적으로 분류기로서 네트워크 마지막 계층에서 사용


```python
from tensorflow.keras.layers import Dense
```


```python
output_size = 64
fc = Dense(units=output_size, activation='relu')
```

## 유효 수용 영역(ERF)
- 입력 이미지에서 거리가 먼 요소를 상호 참조하여 결합하여 네트워크 능력에 영향을 줌
- 입력 이미지의 영역을 정의해 주어진 계층을 위한 뉴런의 활성화에 영향을 미침
- 한 계층의 필터 크기나 윈도우 크기로 불리기 때문에 `수용 영역`이라는 용어를 흔히 볼 수 있음
- 가우시안 분포를 따름
