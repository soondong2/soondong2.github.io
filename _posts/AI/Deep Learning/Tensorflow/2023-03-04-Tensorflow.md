---
title: "[Tensorflow] Tensor"
date: 2023-03-04

categories:
  - AI
  - Deep Learning
tags:
    - Python
    - Tensorflow
---

```python
import numpy as np
import tensorflow as tf
```

## 텐서(Tensor)
- `Rank` : 축의 개수
- `Shape` : 형상(각 축에 따른 차원 개수)
- `Type` : 데이터 타입

### 0D Tensor(Scalar)
- 하나의 숫자를 담고 있는 텐서
- 축과 형상이 없음


```python
t0 = tf.constant(1)

print(t0)
print(tf.rank(t0))
print(t0.shape)
```

    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(0, shape=(), dtype=int32)
    ()
    

### 1D Tensor(Vector)
- 값들을 저장한 리스트와 유사한 텐서
- 하나의 축이 존재


```python
t1 = tf.constant([1, 2, 3])

print(t1)
print(tf.rank(t1))
print(t1.shape)
```

    tf.Tensor([1 2 3], shape=(3,), dtype=int32)
    tf.Tensor(1, shape=(), dtype=int32)
    (3,)
    

### 2D Tensor(Matrix)
- 행렬과 같은 모양의로 두 개의 축이 존재
- 주로 샘플과 특성을 가진 구조로 사용


```python
t2 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(t2)
print(tf.rank(t2))
print(t2.shape)
```

    tf.Tensor(
    [[1 2 3]
     [4 5 6]
     [7 8 9]], shape=(3, 3), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    (3, 3)
    

### 3D Tensor
- 큐브와 같은 모양으로 세 개의 축이 존재
- 데이터가 연속된 시퀀스 데이터나 시간 축이 포함된 시계열 데이터에 해당
- samples, timesteps, features를 가진 구조


```python
t3 = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

print(t3)
print(tf.rank(t3))
print(t3.shape)
```

    tf.Tensor(
    [[[1 2 3]
      [4 5 6]
      [7 8 9]]
    
     [[1 2 3]
      [4 5 6]
      [7 8 9]]
    
     [[1 2 3]
      [4 5 6]
      [7 8 9]]], shape=(3, 3, 3), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)
    (3, 3, 3)
    

### 4D Tensor
- 4개의 축
- 컬러 이미지가 대표적인 사례(흑백 이미지 데이터는 3D Tensor)
- samples, height, width, channel을 가진 구조

### 5D Tensor
- 5개의 축
- 비디오 데이터가 대표적인 사례
- samples, frames, height, width, channel을 가진 구조

## 텐서 데이터 타입
- 정수형(int32), 실수형(float32), 문자열(string)
- 타입 변환에는 `tf.cast()` 사용


```python
f16 = tf.constant(2., dtype=tf.float16)
print(f16)
```

    tf.Tensor(2.0, shape=(), dtype=float16)
    


```python
# 타입 변환 (float16 -> float32)
t32 = tf.cast(f16, tf.float32)
print(t32)
```

    tf.Tensor(2.0, shape=(), dtype=float32)
    

## 텐서 연산
- `tf.add` : 더하기
- `tf.subtract` : 빼기
- `tf.multiply` : 곱하기
- `tf.divide` : 나누기


```python
# 더하기
print(tf.constant(2) + tf.constant(2))
print(tf.add(tf.constant(2), tf.constant(2)))
```

    tf.Tensor(4, shape=(), dtype=int32)
    tf.Tensor(4, shape=(), dtype=int32)
    


```python
# 빼기
print(tf.constant(2) - tf.constant(2))
print(tf.subtract(tf.constant(2), tf.constant(2)))
```

    tf.Tensor(0, shape=(), dtype=int32)
    tf.Tensor(0, shape=(), dtype=int32)
    


```python
# 곱하기
print(tf.constant(2) * tf.constant(2))
print(tf.multiply(tf.constant(2), tf.constant(2)))
```

    tf.Tensor(4, shape=(), dtype=int32)
    tf.Tensor(4, shape=(), dtype=int32)
    


```python
# 나누기
print(tf.constant(2) / tf.constant(2))
print(tf.divide(tf.constant(2), tf.constant(2)))
```

    tf.Tensor(1.0, shape=(), dtype=float64)
    tf.Tensor(1.0, shape=(), dtype=float64)
