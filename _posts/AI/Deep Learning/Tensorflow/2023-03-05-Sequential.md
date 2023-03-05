---
title: "[Tensorflow] Sequential Model"
date: 2023-03-05

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## Model
- `Sequential()`
- 서브 클래싱(Subclassing)
- 함수형 API

### Sequential()
- 모델이 순차적인 구조로 진행할 때 사용
- 간단한 방법
    - Sequential 객체 생성 후 `add()`를 이용한 방법
    - Sequential 인자에 한 번에 추가 방법
- 다중 입력 및 출력이 존재하는 복잡한 모델을 구성할 수 없음


```python
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
```


```python
# add 사용
model = Sequential()
model.add(Input(shape=(28, 28)))
model.add(Dense(units=300, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_1 (Dense)             (None, 28, 300)           8700      
                                                                     
     dense_2 (Dense)             (None, 28, 100)           30100     
                                                                     
     dense_3 (Dense)             (None, 28, 10)            1010      
                                                                     
    =================================================================
    Total params: 39,810
    Trainable params: 39,810
    Non-trainable params: 0
    _________________________________________________________________
    


```python
plot_model(model)
```

![image](https://user-images.githubusercontent.com/100760303/222938049-fbe7ec96-00c0-4a26-b01b-6e6c34b73958.png)

    




```python
# 인자에 리스트로 추가
model = Sequential([Input(shape=(28, 28)),
                    Dense(units=300, activation='relu', name='Dense1'),
                    Dense(units=100, activation='relu', name='Dense2'),
                    Dense(units=10, activation='softmax', name='Ouput')])
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     Dense1 (Dense)              (None, 28, 300)           8700      
                                                                     
     Dense2 (Dense)              (None, 28, 100)           30100     
                                                                     
     Ouput (Dense)               (None, 28, 10)            1010      
                                                                     
    =================================================================
    Total params: 39,810
    Trainable params: 39,810
    Non-trainable params: 0
    _________________________________________________________________
    
