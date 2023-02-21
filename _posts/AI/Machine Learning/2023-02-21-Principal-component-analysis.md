---
title: 주성분 분석(Principal Component Analysis)
date: 2023-02-21

categories:
  - AI
  - Machine Learning
tags:
    - Python
    - ML
---

## 주성분분석
여러 특성(feature) 가운데 대표 특성을 찾아 분석하는 방식으로, 자료의 차원을 고차원에서 하위 차원으로 축소하는 `차원축소` 기법을 활용한다. 

## 분산, 차원축소를 위한 주성분의 선택 기준
차원축소를 위한 정사영의 시작은 무엇을 기준으로 선택되는 걸까?<br>
선택에 따라 데이터의 실제 특성을 보존할 수도 있고, 잃을 수도 있다.<br>
주성분 선택에 있어 최초로 고려되는 요소는 8**분산이 가장 큰 하나의 데이터 선**이 된다. 분산을 기준으로 첫 번째 주성분을 찾아낸다. **분산이 가장 큰 경우가 데이터의 변동 방향을 가장 잘 설명하는 첫 번째 주성분 값이 되며, 그것에 직교하는 선이 두 번째 주성분이 된다.**

## 직교, 그 다음 주성분을 찾는 기준
두 번째 주성분은 첫 번째 주성분과 `직교`하는 또 하나의 선이다. 첫 번째 선택 방향과 직교하면서 첫 번째로 분산이 큰 쪽이 선택된다. 두 선이 직교하고 있다면 하나의 선과 다른 하나의 선은 서로 가장 독립적인 상태라고 말할 수 있다. 이를 내적이 0인 상태라고 하는데, 좌표상에 두 선이 수직을 이루며 교차함을 뜻한다.

## 차원축소의 3가지 순기능
주성분 분석을 사용하는 이유는 데이터가 가진 특성의 수가 지나치게 많을 때, 그 수를 적절하게 줄임으로써 얻어지는 이점이 있기 때문이다. 특성의 수를 줄일 때 우리는 크게 3가지 순기능을 기대해볼 수 있다.

1. 차원이 낮아지면 대상에 대한 이해가 보다 쉬워지게 된다.
2. 연산속도가 개선된다. 분산값을 유지하면서 정보의 크기 자체를 줄이기 때문에 데이터 특성을 훼손시키지 않고도 보다 빠른 연산을 기대할 수 있게 된다.
3. 차원축소는 `차원의 저주`를 해결하는 열쇠가 된다. 만약 데이터 양이 동일한 경우에 보다 상위 차원 속에 데이터를 위치시키면 데이터간의 거리가 점점 멀어지게 된다. 면적당 데이터의 수가 떨어진다는 것이다. 이런 문제를 차원의 저주라고 한다. 차원 증가에 따라 요구되는 데이터의 양이 기하급수적으로 늘어나기 때문에 차원축소를 통해 이와 같은 문제를 해결한다. 또한 차원축소는 데이터가 부족한 상태에서 과적합을 예방하는 전처리 기법으로도 사용 가능하다.


<br>

## Library Import


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

<br>

## Data Load


```python
cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, dtype='category')

df['class'] = y
```


```python
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



양성/악성 정보 바탕의 진단 결과를 속성별 관계로 플로팅하여 환자 분포의 분명한 차이를 시각적으로 확인한다.


```python
sns.pairplot(vars=['mean radius', 'mean texture', 'mean area', 'mean concavity'], hue='class', data=df)
plt.show()
```


    

    
<br>

## Scailing
주성분 분석을 사용하기 전 표준화 또는 정규화 작업이 필요하다. 속성마다 데이터값의 범위가 다르기 때문에 이를 주성분 선별 전에 전처리해야 각 속성의 영향도를 같은 선상에서 비교할 수 있다.

```python
# 표준화
scale = StandardScaler()
x_scaled = scale.fit_transform(cancer.data)
```

<br>

## PCA


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=8)
comp = pca.fit_transform(x_scaled)
```


```python
pca_df = pd.DataFrame(comp, columns=['Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5', 'Comp6', 'Comp7', 'Comp8'])

print(pca_df.shape)
pca_df.head()
```

    (569, 8)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Comp1</th>
      <th>Comp2</th>
      <th>Comp3</th>
      <th>Comp4</th>
      <th>Comp5</th>
      <th>Comp6</th>
      <th>Comp7</th>
      <th>Comp8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.192837</td>
      <td>1.948583</td>
      <td>-1.123166</td>
      <td>3.633731</td>
      <td>-1.195110</td>
      <td>1.411424</td>
      <td>2.159342</td>
      <td>-0.398406</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.387802</td>
      <td>-3.768172</td>
      <td>-0.529293</td>
      <td>1.118264</td>
      <td>0.621775</td>
      <td>0.028656</td>
      <td>0.013357</td>
      <td>0.240987</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.733896</td>
      <td>-1.075174</td>
      <td>-0.551748</td>
      <td>0.912083</td>
      <td>-0.177086</td>
      <td>0.541451</td>
      <td>-0.668195</td>
      <td>0.097374</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.122953</td>
      <td>10.275589</td>
      <td>-3.232790</td>
      <td>0.152547</td>
      <td>-2.960878</td>
      <td>3.053423</td>
      <td>1.429936</td>
      <td>1.059571</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.935302</td>
      <td>-1.948072</td>
      <td>1.389767</td>
      <td>2.940639</td>
      <td>0.546748</td>
      <td>-1.226493</td>
      <td>-0.936165</td>
      <td>0.636379</td>
    </tr>
  </tbody>
</table>
</div>

<br>


```python
# 표준편차
pca_std = pd.DataFrame(pca_df.std()).T

# 각 주성분의 설명된 분산 비율
pca_var = pd.DataFrame(pca.explained_variance_ratio_.T).set_index(pca_std.columns).T

# 누적 기여율
pca_cumsum = pd.DataFrame(pca.explained_variance_ratio_.cumsum()).set_index(pca_std.columns).T

pca_summary = pd.concat([pca_std, pca_var, pca_cumsum])
pca_summary.index = ['Standard deviation', 'Propotion of variance', 'Cumulative proportion']
pca_summary
     
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Comp1</th>
      <th>Comp2</th>
      <th>Comp3</th>
      <th>Comp4</th>
      <th>Comp5</th>
      <th>Comp6</th>
      <th>Comp7</th>
      <th>Comp8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Standard deviation</th>
      <td>3.647601</td>
      <td>2.387755</td>
      <td>1.680152</td>
      <td>1.408591</td>
      <td>1.285159</td>
      <td>1.099765</td>
      <td>0.822441</td>
      <td>0.690982</td>
    </tr>
    <tr>
      <th>Propotion of variance</th>
      <td>0.442720</td>
      <td>0.189712</td>
      <td>0.093932</td>
      <td>0.066021</td>
      <td>0.054958</td>
      <td>0.040245</td>
      <td>0.022507</td>
      <td>0.015887</td>
    </tr>
    <tr>
      <th>Cumulative proportion</th>
      <td>0.442720</td>
      <td>0.632432</td>
      <td>0.726364</td>
      <td>0.792385</td>
      <td>0.847343</td>
      <td>0.887588</td>
      <td>0.910095</td>
      <td>0.925983</td>
    </tr>
  </tbody>
</table>
</div>




```python
per_var = np.round(pca.explained_variance_, 1)
```

<br>

## Scree Plot

Scree Plot을 활용하여 고유값이 수평을 유지하기 전단계로 주성분의 수를 결정한다.


```python
plt.figure(figsize=(8, 6))
plt.title('Scree Plot', fontsize=15)
plt.plot(range(1, 9), per_var,
         marker='o', markerfacecolor='w', markersize=6, markeredgecolor='k', color='r')
plt.show()
```

![image](https://user-images.githubusercontent.com/100760303/220266153-84c4b0c6-c361-4e4a-99ac-396d006e6425.png)

    
<br>


## PCA 계산 과정

- `고유벡터` : 고유벡터는 주성분(P)와 표준화된 독립변수(Z) 사이의 관계를 보여준다. <br>
- PC1 = 0.218902(X1) + 0.103725(X2) + ... + 0.131784(X30) <br>
- cancer 데이터가 target 변수를 제외하고 총 30개의 변수를 갖고 있으므로 index 0~29를 갖는 고유벡터가 생성된다.


```python
pca_vector = pd.DataFrame(pca.components_.T)
pca_vector.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']
pca_vector
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.218902</td>
      <td>-0.233857</td>
      <td>-0.008531</td>
      <td>0.041409</td>
      <td>0.037786</td>
      <td>0.018741</td>
      <td>-0.124086</td>
      <td>-0.007452</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.103725</td>
      <td>-0.059706</td>
      <td>0.064550</td>
      <td>-0.603050</td>
      <td>-0.049469</td>
      <td>-0.032179</td>
      <td>0.011400</td>
      <td>0.130675</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.227537</td>
      <td>-0.215181</td>
      <td>-0.009314</td>
      <td>0.041983</td>
      <td>0.037375</td>
      <td>0.017308</td>
      <td>-0.114474</td>
      <td>-0.018687</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.220995</td>
      <td>-0.231077</td>
      <td>0.028700</td>
      <td>0.053434</td>
      <td>0.010331</td>
      <td>-0.001888</td>
      <td>-0.051652</td>
      <td>0.034674</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.142590</td>
      <td>0.186113</td>
      <td>-0.104292</td>
      <td>0.159383</td>
      <td>-0.365089</td>
      <td>-0.286375</td>
      <td>-0.140670</td>
      <td>-0.288975</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.239285</td>
      <td>0.151892</td>
      <td>-0.074092</td>
      <td>0.031795</td>
      <td>0.011704</td>
      <td>-0.014131</td>
      <td>0.030920</td>
      <td>-0.151396</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.258400</td>
      <td>0.060165</td>
      <td>0.002734</td>
      <td>0.019123</td>
      <td>0.086375</td>
      <td>-0.009344</td>
      <td>-0.107521</td>
      <td>-0.072827</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.260854</td>
      <td>-0.034768</td>
      <td>-0.025564</td>
      <td>0.065336</td>
      <td>-0.043861</td>
      <td>-0.052050</td>
      <td>-0.150484</td>
      <td>-0.152323</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.138167</td>
      <td>0.190349</td>
      <td>-0.040240</td>
      <td>0.067125</td>
      <td>-0.305941</td>
      <td>0.356458</td>
      <td>-0.093892</td>
      <td>-0.231531</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.064363</td>
      <td>0.366575</td>
      <td>-0.022574</td>
      <td>0.048587</td>
      <td>-0.044424</td>
      <td>-0.119431</td>
      <td>0.295760</td>
      <td>-0.177121</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.205979</td>
      <td>-0.105552</td>
      <td>0.268481</td>
      <td>0.097941</td>
      <td>-0.154456</td>
      <td>-0.025603</td>
      <td>0.312490</td>
      <td>0.022540</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.017428</td>
      <td>0.089980</td>
      <td>0.374634</td>
      <td>-0.359856</td>
      <td>-0.191651</td>
      <td>-0.028747</td>
      <td>-0.090755</td>
      <td>-0.475413</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.211326</td>
      <td>-0.089457</td>
      <td>0.266645</td>
      <td>0.088992</td>
      <td>-0.120990</td>
      <td>0.001811</td>
      <td>0.314642</td>
      <td>-0.011897</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.202870</td>
      <td>-0.152293</td>
      <td>0.216007</td>
      <td>0.108205</td>
      <td>-0.127574</td>
      <td>-0.042864</td>
      <td>0.346678</td>
      <td>0.085805</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.014531</td>
      <td>0.204430</td>
      <td>0.308839</td>
      <td>0.044664</td>
      <td>-0.232066</td>
      <td>-0.342917</td>
      <td>-0.244024</td>
      <td>0.573410</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.170393</td>
      <td>0.232716</td>
      <td>0.154780</td>
      <td>-0.027469</td>
      <td>0.279968</td>
      <td>0.069197</td>
      <td>0.023462</td>
      <td>0.117460</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.153590</td>
      <td>0.197207</td>
      <td>0.176464</td>
      <td>0.001317</td>
      <td>0.353982</td>
      <td>0.056343</td>
      <td>-0.208822</td>
      <td>0.060567</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.183417</td>
      <td>0.130322</td>
      <td>0.224658</td>
      <td>0.074067</td>
      <td>0.195548</td>
      <td>-0.031224</td>
      <td>-0.369645</td>
      <td>-0.108319</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.042498</td>
      <td>0.183848</td>
      <td>0.288584</td>
      <td>0.044073</td>
      <td>-0.252869</td>
      <td>0.490246</td>
      <td>-0.080383</td>
      <td>0.220149</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.102568</td>
      <td>0.280092</td>
      <td>0.211504</td>
      <td>0.015305</td>
      <td>0.263297</td>
      <td>-0.053195</td>
      <td>0.191394</td>
      <td>0.011168</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.227997</td>
      <td>-0.219866</td>
      <td>-0.047507</td>
      <td>0.015417</td>
      <td>-0.004407</td>
      <td>-0.000291</td>
      <td>-0.009711</td>
      <td>0.042619</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.104469</td>
      <td>-0.045467</td>
      <td>-0.042298</td>
      <td>-0.632808</td>
      <td>-0.092883</td>
      <td>-0.050008</td>
      <td>0.009870</td>
      <td>0.036251</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.236640</td>
      <td>-0.199878</td>
      <td>-0.048547</td>
      <td>0.013803</td>
      <td>0.007454</td>
      <td>0.008501</td>
      <td>-0.000446</td>
      <td>0.030558</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.224871</td>
      <td>-0.219352</td>
      <td>-0.011902</td>
      <td>0.025895</td>
      <td>-0.027391</td>
      <td>-0.025164</td>
      <td>0.067828</td>
      <td>0.079394</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.127953</td>
      <td>0.172304</td>
      <td>-0.259798</td>
      <td>0.017652</td>
      <td>-0.324435</td>
      <td>-0.369255</td>
      <td>-0.108830</td>
      <td>0.205852</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.210096</td>
      <td>0.143593</td>
      <td>-0.236076</td>
      <td>-0.091328</td>
      <td>0.121804</td>
      <td>0.047706</td>
      <td>0.140473</td>
      <td>0.084020</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.228768</td>
      <td>0.097964</td>
      <td>-0.173057</td>
      <td>-0.073951</td>
      <td>0.188519</td>
      <td>0.028379</td>
      <td>-0.060489</td>
      <td>0.072468</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.250886</td>
      <td>-0.008257</td>
      <td>-0.170344</td>
      <td>0.006007</td>
      <td>0.043332</td>
      <td>-0.030873</td>
      <td>-0.167969</td>
      <td>-0.036171</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.122905</td>
      <td>0.141883</td>
      <td>-0.271313</td>
      <td>-0.036251</td>
      <td>-0.244559</td>
      <td>0.498927</td>
      <td>-0.018489</td>
      <td>0.228225</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.131784</td>
      <td>0.275339</td>
      <td>-0.232791</td>
      <td>-0.077053</td>
      <td>0.094423</td>
      <td>-0.080223</td>
      <td>0.374659</td>
      <td>0.048361</td>
    </tr>
  </tbody>
</table>
</div>




```python
# StandardScaler로 표준화된 x값들에 대한 데이터 프레임
x_df = pd.DataFrame(x_scaled, columns=cancer.feature_names)

```

- `고유값`
- 각 주성분의 분산과 고유값은 일치한다.


```python
# 상관 행렬
# 스케일링한 x 데이터를 이용한다.
corr = x_df.corr(method = 'pearson')

# (w = 고유값),(v = 고유벡터), (corr.values=DataFrame인 corr를 numpy array로 변환)
w, v = np.linalg.eig(corr.values)

# 고유값
eig_value = pd.DataFrame(w[:8], columns=['eigenvalue'], index=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
eig_value = eig_value.T  # or eig_value.transpose()
eig_value
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>eigenvalue</th>
      <td>13.281608</td>
      <td>5.691355</td>
      <td>2.817949</td>
      <td>1.98064</td>
      <td>1.648731</td>
      <td>1.207357</td>
      <td>0.67522</td>
      <td>0.476617</td>
    </tr>
  </tbody>
</table>
</div>


<br>


```python
# 각 주성분의 분산
pca.explained_variance_
```




    array([13.30499079,  5.7013746 ,  2.82291016,  1.98412752,  1.65163324,
            1.20948224,  0.67640888,  0.47745625])


<br>

- `기여율`
- 전체 정보량 중 자기 정보량의 비율이다.
- ex) PC1의 기여율 = (13.304990794374552 / 30) = 1.660201


```python
# 기여율
eig_value / 30
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>eigenvalue</th>
      <td>0.44272</td>
      <td>0.189712</td>
      <td>0.093932</td>
      <td>0.066021</td>
      <td>0.054958</td>
      <td>0.040245</td>
      <td>0.022507</td>
      <td>0.015887</td>
    </tr>
  </tbody>
</table>
</div>

<br>

- `누적 기여율`
- 첫번째 주성분부터 자기 정보량까지 총합의 비율이다.



```python
# PC2의 누적 기여율 계산 방법
(eig_value['PC1'] + eig_value['PC2']) / 30
```




    eigenvalue    0.632432
    dtype: float64



직접 계산한 결과값과 앞서 계산한 pca_summary와 일치함을 확인할 수 있다.


```python
pca_summary
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Comp1</th>
      <th>Comp2</th>
      <th>Comp3</th>
      <th>Comp4</th>
      <th>Comp5</th>
      <th>Comp6</th>
      <th>Comp7</th>
      <th>Comp8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Standard deviation</th>
      <td>3.647601</td>
      <td>2.387755</td>
      <td>1.680152</td>
      <td>1.408591</td>
      <td>1.285159</td>
      <td>1.099765</td>
      <td>0.822441</td>
      <td>0.690982</td>
    </tr>
    <tr>
      <th>Propotion of variance</th>
      <td>0.442720</td>
      <td>0.189712</td>
      <td>0.093932</td>
      <td>0.066021</td>
      <td>0.054958</td>
      <td>0.040245</td>
      <td>0.022507</td>
      <td>0.015887</td>
    </tr>
    <tr>
      <th>Cumulative proportion</th>
      <td>0.442720</td>
      <td>0.632432</td>
      <td>0.726364</td>
      <td>0.792385</td>
      <td>0.847343</td>
      <td>0.887588</td>
      <td>0.910095</td>
      <td>0.925983</td>
    </tr>
  </tbody>
</table>
</div>
