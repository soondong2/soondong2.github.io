---
title: 의사결정나무(Decision Tree)
date: 2023-02-21

categories:
  - AI
  - Machine Learning
tags:
    - Python
    - ML
---

## 의사결정나무
전체 집단을 계속 양분하는 분류기법으로써 분기가 발생하는 노드에는 기준이 되는 질문이 있어 기준 질문에 부합하냐, 부합하지 않느냐에 따라 노드 이동의 방향이 결정된다. 분류(classification)와 회귀(regression) 예측이 모두 가능한 알고리즘이다. 분류나무 모형은 불연속적(이산형)인 값을 예측한다. 회귀나무 모형은 연속적인 값을 예측한다.

## 분할 규칙
데이터 집단을 나눌 경우에는 분할 기준이 있다. 분할 기준은 분할 변수와 목표 변수를 통해 산포된 데이터들을 가장 잘 구분해줄 수 있는 지점(기준)을 찾는 기준이 된다.

## 순수도/불순도(Purity/Impurity)
분할점은 순수도가 최대(=불순도가 최소)가 되도록 설정된다. 불순도가 낮으면 불확실성도 감소하는데 이를 정보이론에서는 정보획득(Impormation Gain)이라고 한다.

## 불순도와 엔트로피(Entropy)
결정트리에서 불순도를 측정하는 지표로 엔트로피가 적용되어 있다. 정보이론에서 엔트로피가 높을수록 정보 내용의 기대 수준이 떨어지는 것이다. 데이터의 혼잡도가 높을수록 엔트로피 값도 높게 나타난다.

## 결정트리의 장점
- 결과를 설명하기에 용이하다.
- 대용량 데이터에서도 빠르게 만들 수 있다.
- 비정상 잡음 데이터에 대해서도 민감함 없이 분류가 가능하다.
- 상관성이 높은 변수가 있어도 크게 영향을 받지 않는다.
- 전처리가 거의 필요하지 않으며 스케일링 작업이 필요하지 않다.

## 결정트리의 단점
- 과대적합 가능성이 높다.
- 분류 경계선 부근의 자료값에 대해서 오차가 크다.
- 설명변수 간의 중요도를 판단하기 쉽지 않다.

## 파라미터
- `min_samples_split` : 노드를 분할하기 위한 최소 샘플 수
- `min_samples_leaf` : 리프노드가 되기 위한 최소한의 샘플 데이터 수
- `max_features` : 최대 feature 개수
- `max_depth` : 트리의 최대 깊이
- `max_leaf_nodes` : 리프노드의 최대 개수

## Library Import


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import io
import graphviz
import pydot
from IPython.core.display import Image
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
```

## Dataset


```python
iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

x.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



## Data Split


```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
```

## Decision Tree Classification


```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Confusion Matrix


```python
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predict'], margins=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predict</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>All</th>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        10
               1       0.90      0.90      0.90        10
               2       0.90      0.90      0.90        10
    
        accuracy                           0.93        30
       macro avg       0.93      0.93      0.93        30
    weighted avg       0.93      0.93      0.93        30
    
    

## Feature Importance


```python
plt.figure(figsize=(10, 6))
sns.barplot(y=x.columns.tolist(), x=model.feature_importances_, edgecolor=(0, 0, 0))
plt.show()
```

![image](https://user-images.githubusercontent.com/100760303/220313904-b2a6911e-2ddb-4f91-8e0d-aa9c963f6d90.png)


## Tree 시각화


```python
def draw_decision_tree(model):
    dot = io.StringIO()
    export_graphviz(model, out_file=dot, feature_names=iris.feature_names)
    graph = pydot.graph_from_dot_data(dot.getvalue())[0]
    image = graph.create_png()

    return Image(image)
```


```python
draw_decision_tree(model)
```

![image](https://user-images.githubusercontent.com/100760303/220314058-cef410dc-fd19-4fa0-bba2-ab9eadcf731b.png)


```python
# 텍스트를 통한 시각화
r = tree.export_text(decision_tree=model, feature_names=iris.feature_names)
print(r)
```

    |--- petal length (cm) <= 2.45
    |   |--- class: 0
    |--- petal length (cm) >  2.45
    |   |--- petal width (cm) <= 1.65
    |   |   |--- petal length (cm) <= 4.95
    |   |   |   |--- class: 1
    |   |   |--- petal length (cm) >  4.95
    |   |   |   |--- sepal length (cm) <= 6.15
    |   |   |   |   |--- sepal width (cm) <= 2.45
    |   |   |   |   |   |--- class: 2
    |   |   |   |   |--- sepal width (cm) >  2.45
    |   |   |   |   |   |--- class: 1
    |   |   |   |--- sepal length (cm) >  6.15
    |   |   |   |   |--- class: 2
    |   |--- petal width (cm) >  1.65
    |   |   |--- petal length (cm) <= 4.85
    |   |   |   |--- sepal width (cm) <= 3.00
    |   |   |   |   |--- class: 2
    |   |   |   |--- sepal width (cm) >  3.00
    |   |   |   |   |--- class: 1
    |   |   |--- petal length (cm) >  4.85
    |   |   |   |--- class: 2
    
    


```python
# plot tree를 이용한 시각화
tree.plot_tree(model)
```




    [Text(133.92000000000002, 199.32, 'X[2] <= 2.45\ngini = 0.667\nsamples = 120\nvalue = [40, 40, 40]'),
     Text(100.44000000000001, 163.07999999999998, 'gini = 0.0\nsamples = 40\nvalue = [40, 0, 0]'),
     Text(167.40000000000003, 163.07999999999998, 'X[3] <= 1.65\ngini = 0.5\nsamples = 80\nvalue = [0, 40, 40]'),
     Text(66.96000000000001, 126.83999999999999, 'X[2] <= 4.95\ngini = 0.133\nsamples = 42\nvalue = [0, 39, 3]'),
     Text(33.480000000000004, 90.6, 'gini = 0.0\nsamples = 38\nvalue = [0, 38, 0]'),
     Text(100.44000000000001, 90.6, 'X[0] <= 6.15\ngini = 0.375\nsamples = 4\nvalue = [0, 1, 3]'),
     Text(66.96000000000001, 54.359999999999985, 'X[1] <= 2.45\ngini = 0.5\nsamples = 2\nvalue = [0, 1, 1]'),
     Text(33.480000000000004, 18.119999999999976, 'gini = 0.0\nsamples = 1\nvalue = [0, 0, 1]'),
     Text(100.44000000000001, 18.119999999999976, 'gini = 0.0\nsamples = 1\nvalue = [0, 1, 0]'),
     Text(133.92000000000002, 54.359999999999985, 'gini = 0.0\nsamples = 2\nvalue = [0, 0, 2]'),
     Text(267.84000000000003, 126.83999999999999, 'X[2] <= 4.85\ngini = 0.051\nsamples = 38\nvalue = [0, 1, 37]'),
     Text(234.36, 90.6, 'X[1] <= 3.0\ngini = 0.444\nsamples = 3\nvalue = [0, 1, 2]'),
     Text(200.88000000000002, 54.359999999999985, 'gini = 0.0\nsamples = 2\nvalue = [0, 0, 2]'),
     Text(267.84000000000003, 54.359999999999985, 'gini = 0.0\nsamples = 1\nvalue = [0, 1, 0]'),
     Text(301.32000000000005, 90.6, 'gini = 0.0\nsamples = 35\nvalue = [0, 0, 35]')]





```python
# graphviz를 이용한 시각화
dot_data = tree.export_graphviz(decision_tree=model, feature_names=iris.feature_names,
                                filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)
graph
```

![image](https://user-images.githubusercontent.com/100760303/220314268-38dc439f-9ccd-4266-a471-06fb25b59b0c.png)
