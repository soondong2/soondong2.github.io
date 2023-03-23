---
title: "[Chatbot] Chapter6 양방향 LSTM"
date: 2023-03-23

categories:
  - AI
  - Deep Learning
tags:
  - NLP
  - Chatbot
---


## 양방향 LSTM
- RNN이나 LSTM은 일반 신경망과 다르게 시퀀스 또는 시계열 데이터 처리에 특화되어 은닉층에서 과거의 정보를 기억할 수 있음
- 순환 신경망 구조 특성상 데이터가 입력 순을오 처리되기 때문에 이전 시점의 정보만 활용할 수 밖에 없음 (단점)
- 문장이 길어질수록 성능이 저하됨
- 기존 LSTM 계층에 `역방향`으로 처리하는 LSTM 계층을 하나 더 추가해 `양방향`에서 문장의 패턴을 분석할 수 있도록 구성된 게 `양방향 LSTM`
- 양방향에서 처리하므로 시퀀스 길이가 길어져도 정보 손실 없이 처리 가능

## Library Call


```python
import numpy as np
from random import random
from tensorflow.keras.models import  Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, TimeDistributed
```

## Sequence


```python
# 시퀀스 생성
def get_sequence(n_timesteps):
    # 0~1 사이 랜덤 시퀀스 생성
    X = np.array([random() for _ in range(n_timesteps)])

    # 클래스 분류 기준
    limit = n_timesteps / 4.0

    # 누적합 시퀀스에서 클래스 결정
    # 누적합 항목이 limit보다 작은 경우 0, 아닌 경우 1로 분류
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])

    # LSTM 입력을 위해 3차원 텐서로 변경
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)

    return X, y
```


```python
# 하이퍼 파라미터
n_units = 20
n_timesteps = 4
```

## Modeling

랜덤으로 시퀀스를 생성해 임의의 분류 기준에 맞는 클래스를 예측하는 양방향 LSTM 모델 예제


```python
# 양방향 LSTM 모델 정의 - Functional Model
input = Input(shape=(n_timesteps, 1))
x = Bidirectional(LSTM(units=n_units, return_sequences=True))(input)
output = TimeDistributed(Dense(units=1, activation='sigmoid'))(x)

model = Model(inputs=input, outputs=output)
model.summary()
```

    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_5 (InputLayer)        [(None, 4, 1)]            0         
                                                                     
     bidirectional_8 (Bidirectio  (None, 4, 40)            3520      
     nal)                                                            
                                                                     
     time_distributed_8 (TimeDis  (None, 4, 1)             41        
     tributed)                                                       
                                                                     
    =================================================================
    Total params: 3,561
    Trainable params: 3,561
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# 양방향 LSTM 모델 정의 - Sequential Model
model = Sequential()
model.add(Input(shape=(n_timesteps, 1)))
model.add(Bidirectional(LSTM(units=n_units, return_sequences=True)))
model.add(TimeDistributed(Dense(units=1, activation='sigmoid')))

model.summary()
```

    Model: "sequential_9"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     bidirectional_10 (Bidirecti  (None, 4, 40)            3520      
     onal)                                                           
                                                                     
     time_distributed_10 (TimeDi  (None, 4, 1)             41        
     stributed)                                                      
                                                                     
    =================================================================
    Total params: 3,561
    Trainable params: 3,561
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```


```python
# 모델 학습
# epoch마다 학습 데이터를 생성해서 학습
for epoch in range(1000):
    X, y = get_sequence(n_timesteps)
    
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)
```

    1/1 - 3s - loss: 0.6984 - accuracy: 0.0000e+00 - 3s/epoch - 3s/step
    1/1 - 0s - loss: 0.7006 - accuracy: 0.0000e+00 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.7051 - accuracy: 0.0000e+00 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.7013 - accuracy: 0.2500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.7038 - accuracy: 0.0000e+00 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6930 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6959 - accuracy: 0.2500 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.6946 - accuracy: 0.5000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6963 - accuracy: 0.2500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6952 - accuracy: 0.2500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6905 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6926 - accuracy: 0.2500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6914 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6871 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6883 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6867 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6858 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6817 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6849 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6807 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6827 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6801 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6847 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6671 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6697 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6830 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6774 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6809 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6688 - accuracy: 1.0000 - 5ms/epoch - 5ms/step
    1/1 - 0s - loss: 0.6616 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6692 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6632 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6594 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6802 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6853 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6704 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6632 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6617 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6639 - accuracy: 1.0000 - 31ms/epoch - 31ms/step
    1/1 - 0s - loss: 0.6481 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6575 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6786 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6519 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6628 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6784 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6477 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.7119 - accuracy: 0.2500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6495 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6540 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6475 - accuracy: 0.7500 - 27ms/epoch - 27ms/step
    1/1 - 0s - loss: 0.6499 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6260 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6361 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6764 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6657 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6459 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6415 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6376 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6199 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6276 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6311 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6265 - accuracy: 0.7500 - 30ms/epoch - 30ms/step
    1/1 - 0s - loss: 0.6270 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6170 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6102 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6239 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6523 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6107 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5973 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6009 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6119 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6094 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5898 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5831 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5880 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6310 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5903 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5905 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5843 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5623 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5784 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5654 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5779 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6191 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5419 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6182 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5625 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5361 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5380 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5207 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5039 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5128 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4880 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4803 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4712 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5382 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5052 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4679 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4737 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5047 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4439 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4572 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4429 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4797 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4625 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4394 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3894 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4136 - accuracy: 1.0000 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.6607 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5939 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4086 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4232 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5777 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3471 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.9532 - accuracy: 0.2500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4128 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3852 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3378 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4538 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3861 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5925 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3791 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3808 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3645 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3417 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3156 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3407 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3259 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3573 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4622 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2957 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.3395 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3864 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.5204 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3003 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5371 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3239 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3392 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2739 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2734 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3117 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3070 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2922 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2781 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3209 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3312 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2211 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2950 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2835 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3996 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2522 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3346 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2651 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2447 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2921 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2172 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2671 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.3100 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2658 - accuracy: 0.7500 - 5ms/epoch - 5ms/step
    1/1 - 0s - loss: 0.5259 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2915 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2672 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2387 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2279 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2156 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2954 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2532 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2859 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2044 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2138 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1754 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2724 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2081 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3453 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2114 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2439 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2011 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2166 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2511 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2046 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2344 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2094 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2014 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2679 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2407 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2284 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2474 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2022 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.7422 - accuracy: 0.7500 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.1973 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6227 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2630 - accuracy: 0.7500 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.2150 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1823 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1473 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3088 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1758 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1584 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1743 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2193 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.3018 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6382 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2521 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2581 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2191 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2121 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5518 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1594 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1599 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1863 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1661 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1282 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1793 - accuracy: 1.0000 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.1676 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2083 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4973 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2453 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2546 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1433 - accuracy: 1.0000 - 5ms/epoch - 5ms/step
    1/1 - 0s - loss: 0.2265 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1520 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6806 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1277 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5583 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1500 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2777 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3858 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2514 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1109 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2262 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3356 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1489 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2084 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1846 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5845 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2749 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1248 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.2363 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2840 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1210 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2640 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1951 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2774 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2551 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1578 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3051 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.6943 - accuracy: 0.2500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1161 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5584 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1197 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5232 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1426 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2872 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1723 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1210 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1600 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5744 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1508 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.4990 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1436 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1468 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2599 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.3617 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1162 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.0533 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2300 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2400 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1538 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1109 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1074 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1586 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3735 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3716 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3961 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4051 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4837 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2475 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.1362 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4388 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3113 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1310 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1496 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3149 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1716 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1692 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1398 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3272 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3574 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1247 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1546 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1662 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2423 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1965 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1600 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3495 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1367 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1460 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1308 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4425 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1222 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4724 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1273 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1890 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.3426 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.0383 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1466 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2383 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.1661 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2943 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3434 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3269 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1224 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3701 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6268 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2216 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1578 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3273 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1649 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2936 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2020 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.8737 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1401 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3677 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4247 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1351 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1074 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3847 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1526 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1595 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3662 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2700 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4347 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3301 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3539 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1896 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1475 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1613 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1985 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4081 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3059 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4060 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2366 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3708 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1825 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3716 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1836 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1270 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2978 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5899 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1121 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2283 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1903 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1978 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1513 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1812 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2339 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1269 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1987 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1885 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1853 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2135 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1308 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2034 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1450 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2581 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1477 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6195 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1851 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1281 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1539 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1515 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.3796 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1721 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2162 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2370 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1554 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2090 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1370 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1721 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2164 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2135 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4986 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1397 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1611 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2174 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4678 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2503 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2228 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1912 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1167 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1767 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2069 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1818 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1769 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1714 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1215 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2131 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1710 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1921 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1859 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1632 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2061 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6067 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1858 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2048 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1791 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2778 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.4944 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2273 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2116 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1582 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1207 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2013 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1454 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2436 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1983 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.4527 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2583 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.7888 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1539 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6030 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2431 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1213 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1704 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1510 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 1.4731 - accuracy: 0.5000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1189 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1595 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1291 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2233 - accuracy: 0.7500 - 9ms/epoch - 9ms/step
    1/1 - 0s - loss: 0.1283 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1757 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1902 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2519 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1873 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1508 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5432 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1201 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1267 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5190 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2014 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1482 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2393 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2842 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1940 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1221 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2218 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1904 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1499 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1605 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1093 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1390 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1064 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1321 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1331 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.5454 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1573 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.5074 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1866 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.0640 - accuracy: 0.5000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1404 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.9563 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2736 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2379 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2609 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.7749 - accuracy: 0.5000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2384 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.4696 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3008 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.0307 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2883 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1225 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1206 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1524 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2809 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3071 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1731 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1290 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1729 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3439 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3405 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1442 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2037 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2987 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2528 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1026 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3467 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2548 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2824 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1685 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2855 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.3103 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2904 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1805 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4630 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1892 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2973 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1297 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1695 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1358 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1441 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2446 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5390 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1460 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1760 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1456 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1365 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1104 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1324 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1674 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4847 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.8640 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2146 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2713 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2574 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5142 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2459 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1524 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2048 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1129 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1313 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1584 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1693 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.2294 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1656 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2539 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1298 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1670 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1502 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4265 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1542 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1253 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1287 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1281 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1666 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1799 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2260 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1555 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1987 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1081 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1570 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1621 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1080 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1448 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1199 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1125 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1638 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1167 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1964 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1057 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2141 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1463 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2536 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1667 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2352 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1818 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1428 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1397 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1980 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1063 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4606 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2304 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1589 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.0112 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1021 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1195 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1971 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1396 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0964 - accuracy: 1.0000 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.1282 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.4634 - accuracy: 0.2500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1158 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2596 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1839 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1480 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3805 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2141 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1727 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2227 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1113 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1493 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4002 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1530 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1886 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1370 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1517 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3639 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2100 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4073 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1128 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1550 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2420 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.3076 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2351 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1258 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5368 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1194 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0952 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1467 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3268 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0997 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1811 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1095 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1447 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1407 - accuracy: 1.0000 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.1439 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1403 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1582 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1424 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1082 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1654 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1361 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3544 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1438 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1926 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1197 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2070 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0989 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1499 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1646 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2048 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2151 - accuracy: 0.7500 - 9ms/epoch - 9ms/step
    1/1 - 0s - loss: 0.1037 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.0789 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.4339 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1766 - accuracy: 1.0000 - 10ms/epoch - 10ms/step
    1/1 - 0s - loss: 0.2113 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1129 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0976 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1106 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1438 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2194 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2452 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 1.1044 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5313 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1287 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1067 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1787 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4712 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5936 - accuracy: 0.7500 - 9ms/epoch - 9ms/step
    1/1 - 0s - loss: 0.1425 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1308 - accuracy: 1.0000 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.1896 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1911 - accuracy: 1.0000 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.5216 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1158 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1932 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2273 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1809 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1584 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1739 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4798 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2127 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2234 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1896 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1237 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2313 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1369 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.2401 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1169 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1656 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0959 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2705 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1885 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1447 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1570 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1524 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.2518 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1760 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1295 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1719 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1631 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1368 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1237 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2502 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6061 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1062 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1207 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1523 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1242 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1815 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1085 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1784 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1256 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1939 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1364 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1524 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1914 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2491 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1087 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2272 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1458 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.8259 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1183 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.4140 - accuracy: 0.2500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1062 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0679 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1196 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2178 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1162 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1046 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1762 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2008 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2119 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1933 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1322 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0731 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1400 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1265 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1799 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1748 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1127 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5631 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3491 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1868 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0918 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1566 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1852 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1388 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1594 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1321 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0930 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0887 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1808 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6651 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1167 - accuracy: 1.0000 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.1634 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1477 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0572 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2297 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0919 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1308 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2516 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1183 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1968 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1565 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5240 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1843 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1403 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.4203 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1228 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1824 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1170 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1070 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2850 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1902 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0891 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1550 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1247 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2918 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1928 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4606 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1196 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1681 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0819 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1006 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1849 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1964 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1399 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1489 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0679 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1867 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4290 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1593 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1594 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1165 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1143 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0951 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1529 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0776 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1242 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1925 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1376 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0675 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1389 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4042 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0631 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1657 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0717 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1075 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5191 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1359 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1942 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1437 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0725 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2608 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1056 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0626 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5672 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1343 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1848 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1675 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1496 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1779 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0727 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1717 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1593 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1652 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1379 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1711 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1258 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1302 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1504 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1043 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0629 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1958 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1675 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1195 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4556 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2027 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1556 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1070 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2813 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3167 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4850 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5146 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1055 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1320 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2629 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1791 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5119 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2321 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1679 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1249 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.0863 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1811 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1364 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1978 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0968 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1162 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.3680 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0969 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0948 - accuracy: 1.0000 - 11ms/epoch - 11ms/step
    1/1 - 0s - loss: 0.2161 - accuracy: 0.7500 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.4262 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2735 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.4294 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2222 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4410 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.7491 - accuracy: 0.5000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.6723 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2129 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0637 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1732 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1716 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1146 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5867 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0732 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2190 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3054 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1301 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1671 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.4159 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.3482 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0998 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1830 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1215 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.7002 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1600 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1533 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2297 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0954 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1313 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3094 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0868 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2129 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0938 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3343 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3585 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0936 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2077 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0829 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1060 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1733 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1704 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0980 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3599 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2620 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1914 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1198 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2112 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4589 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1161 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2459 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0519 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1545 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2991 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1391 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0520 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0682 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1889 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1598 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0835 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1636 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1424 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1022 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2556 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0983 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0843 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1622 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1497 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1420 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0631 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 1.3731 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1519 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0597 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1270 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1702 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1133 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1562 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1184 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0832 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1216 - accuracy: 1.0000 - 8ms/epoch - 8ms/step
    1/1 - 0s - loss: 0.1748 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1199 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1279 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1003 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2042 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1226 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1447 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4803 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1331 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1790 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1402 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0808 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1507 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2035 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6201 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0467 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0697 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0577 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1696 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2047 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3883 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0711 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1582 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1156 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2718 - accuracy: 0.7500 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.4237 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1353 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0921 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1342 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4633 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0897 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0906 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4983 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1668 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1072 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1272 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0955 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0998 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1494 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1004 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1116 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.5357 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0486 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6201 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1860 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0751 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1704 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2394 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1409 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4273 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0606 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1636 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2528 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0697 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.2576 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2888 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1300 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1762 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1393 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0974 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0754 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1893 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0977 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4277 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1984 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0908 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2031 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.6500 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0825 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1145 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1111 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.7472 - accuracy: 0.5000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.4791 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1096 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1726 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1468 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.2281 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1502 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.1066 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1556 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0916 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0578 - accuracy: 1.0000 - 7ms/epoch - 7ms/step
    1/1 - 0s - loss: 0.4093 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.3721 - accuracy: 0.7500 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.0691 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1712 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    1/1 - 0s - loss: 0.1692 - accuracy: 1.0000 - 6ms/epoch - 6ms/step
    


```python
# get_sequence 함수의 X 부분 예시
print(random())
print(np.array([random() for _ in range(5)]))
```

    0.18470805796753775
    [0.3674843  0.05786435 0.32903239 0.15887972 0.91681222]
    

## Evaluate


```python
# 모델 평가
X, y = get_sequence(n_timesteps)
yhat = model.predict(X, verbose=0)

for i in range(n_timesteps):
    if yhat[0, i] > 0.5:
        pred = 1
    else: 
        pred = 0
    print('실제값 : ', y[0, i], '예측값 : ', pred)
```

    실제값 :  [0] 예측값 :  0
    실제값 :  [0] 예측값 :  0
    실제값 :  [1] 예측값 :  1
    실제값 :  [1] 예측값 :  1
