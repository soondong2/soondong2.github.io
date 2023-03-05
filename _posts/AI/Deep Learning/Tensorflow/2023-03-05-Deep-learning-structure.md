---
title: "[Tensorflow] Deep Learning Structure"
date: 2023-03-05

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

```python
import tensorflow as tf
```

## 딥러닝 구조 및 학습
- 모델(네트워크)를 구성하는 `레이어(layer)`
- `입력 데이터`와 그에 대한 `목적(결과)`
- 학습에 사용할 피드백을 정의하는 `손실함수(loss function)`
- 학습 진행 방식을 결정하는 `옵티마이저(optimizer)`

### Layer
- 신경망의 핵심 데이터 구조
- 하나 이상의 텐서를 입력 받아 하나 이상의 텐서를 출력하는 데이터 처리 모듈
- 상태가 없는 레이어도 있지만, 대부분 `가중치(weight)`라는 레이어 상태를 가짐
- 가중치는 확률적 경사 하강법에 의해 학습되는 하나 이상의 텐서

[주요 레이어]
- Dense
- Activation
- Flatten
- Input


```python
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
```

#### Dense
- 완전연결계층(Fully-Connected Layer) : 모든 노드가 서로 완전하게 연결되어 있는 상태
- 노드 수(units), 활성화 함수(activation) 등을 지정
- name을 통해 레이어간 구분 가능
- 가중치 초기화(kernel_initializer)
    - 신경망의 성능에 큰 영향을 주는 요소
    - 보통 가중치의 초기값으로 0에 가까운 무작위 값 사용


```python
Dense(units=10, activation='softmax')
```




    <keras.layers.core.dense.Dense at 0x205c0cacd90>




```python
Dense(units=10, activation='relu', name='Dense Layer')
```




    <keras.layers.core.dense.Dense at 0x205c0d735e0>




```python
Dense(units=10, kernel_initializer='he_normal', name='Dense Layer')
```




    <keras.layers.core.dense.Dense at 0x205c0d735b0>



#### Activation
- Sigmoid
- tanh
- ReLU
- Leaky ReLU


```python
dense = Dense(units=10, activation='relu', name='Dense Layer')
Activation(dense)
```




    <keras.layers.core.activation.Activation at 0x205c0d73c40>



#### Flatten
- 배치 크기를 제외하고 데이터를 1차원으로 쭉 펼치는 작업
- 예시 : (128, 3, 2, 2) -> (128, 12)


```python
Flatten(input_shape=(128, 3, 2, 2))
```




    <keras.layers.core.flatten.Flatten at 0x205c0d73430>



#### Input
- 모델의 입력을 정의
- `shape`, `dtype`을 포함
- 하나의 모델은 여러 개의 입력을 가질 수 있음
- `summary()` 메소드를 통해서는 보이지 않음


```python
Input(shape=(28, 28), dtype=tf.float32)
```




    <KerasTensor: shape=(None, 28, 28) dtype=float32 (created by layer 'input_1')>




```python
Input(shape=(8, ), dtype=tf.int32)
```




    <KerasTensor: shape=(None, 8) dtype=int32 (created by layer 'input_2')>
