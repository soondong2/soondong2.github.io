---
title: "[Tensorflow] 모델 저장 및 복원"
date: 2023-03-08

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## 모델 저장 및 불러오기
- `save()` : 저장
- `models.load_model()` : 복원
- Sequential API, 함수형 API에서는 모델의 저장 및 로드가 가능하지만 서브 클래싱 방법에서는 불가능
- JSON 형식
  - `model.to_json()` : 저장
  - `tf.keras.models.model_from_json(file_path)` : 복원
- YAML로 직렬화
  - `model.to_yaml()` : 저장
  - `tf.keras.models.model_from_yaml(file_path)`: 복원

### 모델 저장
```python
# 모델 저장
model.save('mnist_model.h5')
```

### 모델 복원
```python
# 모델 불러오기
load_model = models.load_model('mnist_model.h5')
```


```python
# 복원한 모델 확인
load_model.summary()
```
