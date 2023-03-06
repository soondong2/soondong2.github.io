---
title: "[Tensorflow] Subclassing Model"
date: 2023-03-06

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## 서브클래싱(Subclassing)
- 커스터마이징에 최적화된 방법
- Model 클래스를 상속받아 Model이 포함하는 기능을 사용할 수 있음
    - `fit()`, `evaluate()`, `predict()`
    - `save()`, `load()`
- 주로 `call()` 메소드 안에서 원하는 계산 가능
- 권장되는 방법은 아니지만 모델의 구현 코드를 참고할 때 해석할 수 있어야 함


```python
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Concatenate
```


```python
class MyModel(Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense_layer1 = Dense(units=300, activation=activation)
        self.dense_layer2 = Dense(units=100, activation=activation)
        self.dense_layer3 = Dense(units=units, activation=activation)
        self.output_layer = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        x = self.dense_layer3(x)
        x = self.output_layer(x)
        return x
```


