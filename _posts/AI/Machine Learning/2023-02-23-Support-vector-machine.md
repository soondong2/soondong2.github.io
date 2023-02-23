---
title: SVM(Support Vector Machine)
date: 2023-02-23

categories:
  - AI
  - Machine Learning
tags:
    - Python
    - ML
---

## 서포트벡터머신
서포트 벡터 머신은 **선형이나 비선형 분류, 회귀, 이상치 탐색에 사용할 수 있는 다목적 머신러닝 모델**이다.
복잡한 분류 문제에 특히 유용하며 작거나 중간 크기의 데이터에 적합하다. 또한 서포트 벡터 머신은 비확률적 이진 선형 분류 모델을 생성한다.
데이터가 사상된 공간에서 경계로 표현되며, 공간상에 존재하는 여러 경계 중 가장 큰 폭을 가진 경계를 찾는다. **마진을 최대화하는 초평면을 찾아 분류와 회귀를 수행**한다.
비선형 분류에도 사용되는데, 비선형 분류에서는 입력자료를 다차원 공간상으로 맵핑할 때 `커널 트릭(kernel trick)`을 사용하기도 한다.

SVM는 **스케일링에 민감**하다. scaler를 활용할 경우 결정 경계가 훨씬 좋아진다. 모든 데이터가 경계선 바깥에 올바르게 분류되어 있다면 `하드 마진` 분류라고 한다. 하드 마진 분류에는 두 가지 문제점이 있다.

1. 데이터가 선형적으로 구분될 수 있어야 한다.
2. 이상치에 민감하다.

이런 문제를 피하기 위해 클래스 간의 결정선을 넓게 유지하는 것과 마진 오류 사이에 적절한 균형을 잡아야하는데, 이를 `소프트 마진` 분류라고 한다. `C` 파라미터를 사용해 균형을 조절한다.

SVM은 클래스에 대한 **확률을 제공하지 않는다.** 비선형 데이터를 다루는 방법은 다항 특성과 같은 특성을 추가하는 것이다. 다항식 특성을 추가하는 것은 낮은 차수는 복잡한 데이터를 잘 표현하지 못하고, 높은 차수는 모델의 과접합이나 속도를 느리게 만든다. 이 때 `커널 트릭`을 사용해 실제로 특성을 추가하지는 않지만 특성을 추가한 것과 같은 결과를 얻을 수 있다.

## 용어
- `초평면(decision hyperline)` : 각 그룹을 구분하는 분류자
- `서포트 벡터(support vector)` : 각 그룹에 속한 데이터 중에서도 초평면에 가장 가까이에 붙어있는 최전방 데이터들
- `마진(margin)` : 서포트 벡터와 초평면 사이의 수직 거리

## 파라미터
- `C` : 클수록 하드마진(오류 허용X), 작을수록 소프트마진(오류 허용O)
- `gamma` : 결정경계를 얼마나 유연하게 그을 것인지를 결정한다. 클수록 경계가 복잡(과대적합), 작을수록 경계가 단순(과소적합)하다.

![image](https://user-images.githubusercontent.com/100760303/220839054-04020ab2-e09f-4eb3-877e-e9ff050e6cfb.png)

## 장점 
- 분류와 예측에 모두 사용 가능하다.
- 신경망 기법에 비해 과적합 정도가 낮다.
- 예측의 정확도가 높다.
- 저차원과 고차원의 데이터에 대해 모두 잘 작동한다.

## 단점
- 전처리와 파라미터에 따라 정확도가 달라진다.
- 예측이 어떻게 이루어지는지에 대한 이해와 모델에 대한 해석이 어렵다.
- 대용량 데이터에 대한 모형 구축시 속도가 느리며 메모리 할달량이 크다.

## Library Import


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

# pip install graphviz
import graphviz
import warnings
warnings.filterwarnings('ignore')
```


```python
from sklearn.datasets import load_breast_cancer, load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
```

## LinearSVC


```python
X = pd.DataFrame(load_breast_cancer().data, columns=load_breast_cancer().feature_names)
y = load_breast_cancer().target
```


```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
```


```python
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=42)
```


```python
lsvc = LinearSVC(random_state=42)
lsvc.fit(X_train, y_train)
y_pred = lsvc.predict(X_test)
```


```python
print('학습 데이터 Score : {}'.format(lsvc.score(X_train, y_train)))
print('평가 데이터 Score : {}'.format(lsvc.score(X_test, y_test)))
```

    학습 데이터 Score : 0.9912087912087912
    평가 데이터 Score : 0.9649122807017544
    


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.95      0.95      0.95        42
               1       0.97      0.97      0.97        72
    
        accuracy                           0.96       114
       macro avg       0.96      0.96      0.96       114
    weighted avg       0.96      0.96      0.96       114
    
    


```python
plt.figure(figsize=(10, 6))
sns.barplot(y=X.columns.tolist(), x=lsvc.coef_.tolist()[0], edgecolor=(0, 0, 0))
plt.show()
```

![image](https://user-images.githubusercontent.com/100760303/220839148-3d73ea84-9c99-420d-9fd6-f75c103e67a4.png)
    


## SVC


```python
svc = SVC(random_state=42, kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
```


```python
print('학습 데이터 Score : {}'.format(svc.score(X_train, y_train)))
print('평가 데이터 Score : {}'.format(svc.score(X_test, y_test)))
```

    학습 데이터 Score : 0.9912087912087912
    평가 데이터 Score : 0.9736842105263158
    


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.95      0.98      0.96        42
               1       0.99      0.97      0.98        72
    
        accuracy                           0.97       114
       macro avg       0.97      0.97      0.97       114
    weighted avg       0.97      0.97      0.97       114
    
    


```python
# kernel = 'linear'일 때만 coef_ 출력 가능
plt.figure(figsize=(10, 6))
sns.barplot(y=X.columns.tolist(), x=svc.coef_.tolist()[0], edgecolor=(0, 0, 0))
plt.show()
```

![image](https://user-images.githubusercontent.com/100760303/220839191-9f876635-5c5d-4b21-994f-1d7fdf20f73c.png)

    


## 회귀
- LinearSVR
- SVR
