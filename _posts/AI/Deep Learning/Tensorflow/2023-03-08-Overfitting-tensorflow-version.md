---
title: "[Tensorflow] 과대적합 방지 방법"
date: 2023-03-08

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## 과소적합(Underfitting)
- 학습 데이터를 충분히 학습하지 않아 성능이 매우 안 좋은 경우
- 모델이 지나치게 단순한 경우

### 해결 방안
- 충분한 학습 데이터 수집
- 복잡한 모델 사용
- epochs를 늘려 충분히 학습

## 과대적합(Overfitting)
- 모델이 학습 데이터에 지나치게 맞추어진 상태
- 새로운 데이터에서는 성능 저하
- 데이터에 잡음이나 오류가 포함
- 학습 데이터가 매우 적을 경우
- 모델이 지나치게 복잡할 경우
- 학습 횟수가 매우 많을 경우

### 해결 방안
- 다양한 학습 데이터 수집 및 학습
- 정규화(Regularization)
- 하이퍼 파라미터

## 과대적합과 과소적합 방지 방법
- 모델의 크기 축소
- 가중치 초기화(Weight Initializer)
- 옵티마이저(Optimizer)
- 배치 정규화(Batch Normalization)
- 규제화(Regularization)
- 드롭아웃(Dropout)

### 1.모델 크기 조절
#### 모델 크기 감소
- 모델의 크기를 줄인하는 것은 학습 파라미터의 수를 줄이는 것

#### 모델 크기 증가
- 모델의 크기를 증가시키는 것은 학습 파라미터의 수를 늘리는 것

### 2. 옵티마이저(Optimizer)
#### 확률적 경사하강법(SGD)
- Stochastic Gradient Descent
- 전체를 한 번에 계산하지 않고 확률적으로 일부 샘플을 뽑아 조금씩 나누어 학습

#### Momentum
- 운동량을 의미
- 공이 그릇의 경사면을 따라서 내려가는 듯한 모습


```python
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

optimizer = SGD(learning_rate=0.001, momentum=0.9)
```

#### AdaGrad
- 가장 가파른 경사를 따라 빠르게 하강하는 방법
- 학습률이 너무 감소되어 전역 최소값에 도달하기 전에 학습이 빨리 종료되는 문제


```python
from tensorflow.keras.optimizers import Adagrad

optimizer = Adagrad(learning_rate=0.001)
```

#### RMSProp
- AdaGrad를 보완하기 위한 방법


```python
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(learning_rate=0.001, rho=0.9)
```

#### Adam
- 모멘텀 최적화와 RMSProp의 아이디어를 합친 것
- 가장 많이 사용되는 최적화 방법


```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
```

### 3. 가중치 초기화
#### Xavier
- 활성화 함수 : tanh, sigmoid, softmax

#### He
- 활성화 함수 : ReLU, LeakyReLU


```python
from tensorflow.keras.layers import Dense, Input, Flatten, Softmax, LeakyReLU, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
```


```python
# 함수형 API
inputs = Input(shape=(10, 10), name='input')
x = Dense(units=30, kernel_initializer='he_normal', name='dense1')(inputs)
x = LeakyReLU(alpha=0.2, name='leaky')(x)
x = Dense(units=1, kernel_initializer='he_normal', name='dens2')(x)
x = Softmax(name='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()
```

    Model: "model_17"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input (InputLayer)          [(None, 10, 10)]          0         
                                                                     
     dense1 (Dense)              (None, 10, 30)            330       
                                                                     
     leaky (LeakyReLU)           (None, 10, 30)            0         
                                                                     
     dens2 (Dense)               (None, 10, 1)             31        
                                                                     
     softmax (Softmax)           (None, 10, 1)             0         
                                                                     
    =================================================================
    Total params: 361
    Trainable params: 361
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Sequential API
model = Sequential([
    Dense(30, kernel_initializer='he_normal', input_shape=[10, 10], name='dens1'),
    LeakyReLU(alpha=0.2),
    Dense(1, kernel_initializer='he_normal', name='dens2'),
    Activation('softmax')
])
model.summary()
```

    Model: "sequential_7"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dens1 (Dense)               (None, 10, 30)            330       
                                                                     
     leaky_re_lu_14 (LeakyReLU)  (None, 10, 30)            0         
                                                                     
     dens2 (Dense)               (None, 10, 1)             31        
                                                                     
     activation_5 (Activation)   (None, 10, 1)             0         
                                                                     
    =================================================================
    Total params: 361
    Trainable params: 361
    Non-trainable params: 0
    _________________________________________________________________
    

### 4. 배치 정규화(Batch Normalization)
- 모델에 주입되는 샘플들을 균일하게 만드는 방법
- 가중치의 활성화 값이 적당히 퍼지게끔 적용
- 과대적합 방지
- 주로 **Dense 또는 Conv2D Layer후, 활성화 함수 이전에 놓임**


```python
from tensorflow.keras.layers import BatchNormalization, Flatten
```


```python
# 함수형 API
inputs = Input(shape=(28, 28), name='input')
x = Flatten(input_shape=(28, 28), name='flatten')(inputs)
x = Dense(units=32, kernel_initializer='he_normal', name='dense1')(x)
x = BatchNormalization(name='batch')(x)
x = Activation('relu', name='relu')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()
```

    Model: "model_23"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input (InputLayer)          [(None, 28, 28)]          0         
                                                                     
     flatten (Flatten)           (None, 784)               0         
                                                                     
     dense1 (Dense)              (None, 32)                25120     
                                                                     
     batch (BatchNormalization)  (None, 32)                128       
                                                                     
     relu (Activation)           (None, 32)                0         
                                                                     
    =================================================================
    Total params: 25,248
    Trainable params: 25,184
    Non-trainable params: 64
    _________________________________________________________________
    


```python
# Sequential API
model = Sequential()

model.add(Dense(units=32, input_shape=(28 * 28, ), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.summary()
```

    Model: "sequential_11"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_12 (Dense)            (None, 32)                25120     
                                                                     
     batch_normalization_3 (Batc  (None, 32)               128       
     hNormalization)                                                 
                                                                     
     activation_7 (Activation)   (None, 32)                0         
                                                                     
    =================================================================
    Total params: 25,248
    Trainable params: 25,184
    Non-trainable params: 64
    _________________________________________________________________
    

### 5. 규제화(Regularization)
- 과대적합 방지하는 방법
- 큰 가중치 값에 큰 규제를 가하는 것
- `L1` : 가중치의 절댓값에 비례하는 비용이 추가
- `L2` : 가중치의 제곱에 비례하는 비용이 추가
- `L1 + L2`


```python
from tensorflow.keras.regularizers import l1, l2, l1_l2
```

#### L2 규제
- `kernel_regularizer=l2(0.001)`


```python
# 함수형 API
inputs = Input(shape=(10000, ), name='input')
x = Dense(units=16, kernel_regularizer=l2(0.001), activation='relu', name='dense1')(inputs)
x = Dense(units=16, kernel_regularizer=l2(0.001), activation='relu', name='dense2')(x)
x = Dense(units=1, activation='sigmoid', name='output')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()
```

    Model: "model_22"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input (InputLayer)          [(None, 10000)]           0         
                                                                     
     dense1 (Dense)              (None, 16)                160016    
                                                                     
     dense2 (Dense)              (None, 16)                272       
                                                                     
     output (Dense)              (None, 1)                 17        
                                                                     
    =================================================================
    Total params: 160,305
    Trainable params: 160,305
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Sequential API
l2_model = Sequential([
    Dense(units=16, kernel_regularizer=l2(0.001), activation='relu', input_shape=(10000, ), name='dense1'),
    Dense(units=16, kernel_regularizer=l2(0.001), activation='relu', name='dense2'),
    Dense(units=1, activation='sigmoid', name='output')
])
model.summary()
```

    Model: "model_22"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input (InputLayer)          [(None, 10000)]           0         
                                                                     
     dense1 (Dense)              (None, 16)                160016    
                                                                     
     dense2 (Dense)              (None, 16)                272       
                                                                     
     output (Dense)              (None, 1)                 17        
                                                                     
    =================================================================
    Total params: 160,305
    Trainable params: 160,305
    Non-trainable params: 0
    _________________________________________________________________
    

#### L1 규제
- `kernel_regularizer=l1(0.0001)`

#### L1 + L2
- `kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)`

### 6. Dropout
- 과대적합 방지 방법
- 학습할 때 사용하는 노드의 수를 전체 노드 중 일부만을 사용
- 훈련하는 동안 무작위로 층의 일부 노드를 제외
- 테스트 단계에서는 드롭아웃 되지 않음


```python
from tensorflow.keras.layers import Dropout
```


```python
# 함수형 API
inputs = Input(shape=(10000, ), name='input')
x = Dense(units=16, activation='relu', name='dense1')(inputs)
x = Dropout(0.5, name='dropout1')(x)
x = Dense(units=16, activation='relu', name='dense2')(x)
x = Dropout(0.5, name='dropout2')(x)
x = Dense(units=1, activation='sigmoid', name='output')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()
```

    Model: "model_20"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input (InputLayer)          [(None, 10000)]           0         
                                                                     
     dense1 (Dense)              (None, 16)                160016    
                                                                     
     dropout1 (Dropout)          (None, 16)                0         
                                                                     
     dense2 (Dense)              (None, 16)                272       
                                                                     
     dropout2 (Dropout)          (None, 16)                0         
                                                                     
     output (Dense)              (None, 1)                 17        
                                                                     
    =================================================================
    Total params: 160,305
    Trainable params: 160,305
    Non-trainable params: 0
    _________________________________________________________________
    

### 7. 하이퍼 파라미터
- 사람이 직접 설정해야 하는 매개변수
- `Learning Rete`, `Epochs`, `Mini Batch Size`, `Validation Data`

