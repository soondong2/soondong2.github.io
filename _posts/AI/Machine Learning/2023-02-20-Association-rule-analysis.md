---
title: 연관규칙분석(Association Rule Analysis)
date: 2023-02-20

categories:
  - AI
  - Machine Learning
tags:
    - Python
    - ML
---

## 연관분석
대량의 트랜잭션 정보로부터 개별 데이터 사이에서 연관규칙(x면 y가 발생)을 찾는 것이다. 가령 슈퍼마켓의 구매내역에서 특정 물건의 판매 발생 빈도를 기반으로 **'A 물건을 구매하는 사람들을 B 물건을 구매하는 경향이 있다'** 라는 규칙을 찾을 수 있다. 다른 말로는 `장바구니 분석(Market Basket Analysis)`라고 한다.

<br>

## 연관규칙
조건 결과의 빈도수를 기반으로 표현되기 때문에 비교적 결과를 쉽게 이해할 수 있다. 넷플릭스도 연관규칙을 추천 알고리즘에 적용했다. A 영화에 대한 시청 결과가 B나 C 영화를 선택할 가능성에 얼마나 영향을 미치는지 계산하는 **조건부 확률**로 콘텐츠 추천 모델을 만들었다.

<br>

### 1. 지지도(Support)
전체 거래 중 항목 A와 B를 동시에 포함하는 거래의 비율이다.<br>
ex) 장을 본 목록을 확인했을 때 우유와 식빵이 꼭 함께 있을 확률
- Support=A와 B가 동시에 포함된 거래 수/전체 거래 수

<br>

### 2. 신뢰도(Confidence)
항목 A를 포함한 거래 중에서 항목 A와 항목 B가 동시에 포함될 확률을 구한다.<br>
ex) 우유를 구매했을 때 식빵이 장바구니로 함께 들어갈 확률

- Confidence=A와 B가 동시에 포함된 거래 수/A를 포함하는 거래 수
- 지지도/P(A)

<br>

### 3. 향상도(Lift)
A가 주어지지 않은 상태에서 B의 확률에 대하여 A사 주어졌을 때 B의 확률의 증가비율이다.
- 지지도/신뢰도

## Library Import

- `mlxtend` : 통계분석 기능을 지원해주는 파이썬 라이브러리

연관규칙을 적용하기 위해 각 항목들이 어떤 빈도로 나타났는지 또는 어떤 항목과 함께 나왔는지를 파악하는 것이 필수다. 하지만 데이터 셋이 큰 경우 모든 항목들에 대해 검사하는 것은 비효율 적이므로 이를 해결하기 위해 연관규칙분석의 대표적인 알고리즘인 `Apriori`을 사용한다.


```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
```

## Dataset
```python
dataset = [
    ['Milk', 'Onion', 'Nutmeg', 'Eggs', 'Yogurt'],
    ['Onion', 'Nutmeg', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Yogurt'],
    ['Corn', 'Onion', 'Onion', 'Ice cream', 'Eggs']
]
```

## Asocitaion Rule Analysis
```python
# TransactionEncoder() : 기계학습에 적합한 배열 형식으로 변환
te = TransactionEncoder()

# One-hot encoding
te_arr = te.fit_transform(dataset)

df = pd.DataFrame(te_arr, columns=te.columns_)
```


```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Corn</th>
      <th>Eggs</th>
      <th>Ice cream</th>
      <th>Milk</th>
      <th>Nutmeg</th>
      <th>Onion</th>
      <th>Unicorn</th>
      <th>Yogurt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

<br>

- Eggs를 구매할 확률은 0.8이다.
- Apple, Eggs를 함께 구매할 확률은 0.2이다.


```python
# min_support : 최소 지지도가 0.05 이상인 규칙 집합
freq_items = apriori(df, min_support=0.05, use_colnames=True)
freq_items
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.2</td>
      <td>(Apple)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.4</td>
      <td>(Corn)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.8</td>
      <td>(Eggs)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.2</td>
      <td>(Ice cream)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>(Milk)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.4</td>
      <td>(Nutmeg)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>(Onion)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.2</td>
      <td>(Unicorn)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.6</td>
      <td>(Yogurt)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.2</td>
      <td>(Eggs, Apple)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.2</td>
      <td>(Milk, Apple)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.2</td>
      <td>(Corn, Eggs)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.2</td>
      <td>(Ice cream, Corn)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.2</td>
      <td>(Corn, Milk)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.2</td>
      <td>(Onion, Corn)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.2</td>
      <td>(Corn, Unicorn)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.2</td>
      <td>(Corn, Yogurt)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.2</td>
      <td>(Ice cream, Eggs)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.4</td>
      <td>(Milk, Eggs)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.4</td>
      <td>(Nutmeg, Eggs)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.6</td>
      <td>(Onion, Eggs)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.4</td>
      <td>(Eggs, Yogurt)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.2</td>
      <td>(Ice cream, Onion)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.2</td>
      <td>(Milk, Nutmeg)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.2</td>
      <td>(Onion, Milk)</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.2</td>
      <td>(Unicorn, Milk)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.4</td>
      <td>(Milk, Yogurt)</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.4</td>
      <td>(Onion, Nutmeg)</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.4</td>
      <td>(Nutmeg, Yogurt)</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.4</td>
      <td>(Onion, Yogurt)</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.2</td>
      <td>(Unicorn, Yogurt)</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.2</td>
      <td>(Milk, Eggs, Apple)</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.2</td>
      <td>(Ice cream, Corn, Eggs)</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.2</td>
      <td>(Onion, Corn, Eggs)</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.2</td>
      <td>(Ice cream, Onion, Corn)</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.2</td>
      <td>(Corn, Unicorn, Milk)</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.2</td>
      <td>(Corn, Milk, Yogurt)</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.2</td>
      <td>(Corn, Unicorn, Yogurt)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.2</td>
      <td>(Ice cream, Onion, Eggs)</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.2</td>
      <td>(Nutmeg, Milk, Eggs)</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.2</td>
      <td>(Onion, Milk, Eggs)</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.2</td>
      <td>(Milk, Eggs, Yogurt)</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.4</td>
      <td>(Nutmeg, Onion, Eggs)</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.4</td>
      <td>(Nutmeg, Eggs, Yogurt)</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.4</td>
      <td>(Onion, Eggs, Yogurt)</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.2</td>
      <td>(Onion, Milk, Nutmeg)</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.2</td>
      <td>(Nutmeg, Milk, Yogurt)</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.2</td>
      <td>(Onion, Milk, Yogurt)</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.2</td>
      <td>(Unicorn, Milk, Yogurt)</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.4</td>
      <td>(Nutmeg, Onion, Yogurt)</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.2</td>
      <td>(Ice cream, Onion, Corn, Eggs)</td>
    </tr>
    <tr>
      <th>51</th>
      <td>0.2</td>
      <td>(Corn, Unicorn, Milk, Yogurt)</td>
    </tr>
    <tr>
      <th>52</th>
      <td>0.2</td>
      <td>(Nutmeg, Onion, Milk, Eggs)</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0.2</td>
      <td>(Nutmeg, Milk, Eggs, Yogurt)</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0.2</td>
      <td>(Onion, Milk, Eggs, Yogurt)</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.4</td>
      <td>(Nutmeg, Onion, Eggs, Yogurt)</td>
    </tr>
    <tr>
      <th>56</th>
      <td>0.2</td>
      <td>(Nutmeg, Onion, Milk, Yogurt)</td>
    </tr>
    <tr>
      <th>57</th>
      <td>0.2</td>
      <td>(Nutmeg, Yogurt, Milk, Eggs, Onion)</td>
    </tr>
  </tbody>
</table>
</div>

<br>

lift(향상도)가 1보다 클수록 우연히 일어나지 않았다는 의미다. 아무런 관계가 없을 경우 1로 표시된다.

- Apple과 Eggs를 모두 구매할 확률은 0.2이다.
- Apple을 구매했을 때 Eggs도 함께 구매할 가능성은 1(100%)이다.
- Eggs를 구매했을 때 Apple도 함께 구매할 가능성은 0.25(25%)이다.


```python
association_rules(freq_items, metric='lift', min_threshold=1)
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Eggs)</td>
      <td>(Apple)</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.250000</td>
      <td>1.250000</td>
      <td>0.04</td>
      <td>1.066667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Apple)</td>
      <td>(Eggs)</td>
      <td>0.2</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>1.000000</td>
      <td>1.250000</td>
      <td>0.04</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Milk)</td>
      <td>(Apple)</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.333333</td>
      <td>1.666667</td>
      <td>0.08</td>
      <td>1.200000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Apple)</td>
      <td>(Milk)</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>1.000000</td>
      <td>1.666667</td>
      <td>0.08</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Ice cream)</td>
      <td>(Corn)</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>0.2</td>
      <td>1.000000</td>
      <td>2.500000</td>
      <td>0.12</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>231</th>
      <td>(Onion, Eggs)</td>
      <td>(Yogurt, Milk, Nutmeg)</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.333333</td>
      <td>1.666667</td>
      <td>0.08</td>
      <td>1.200000</td>
    </tr>
    <tr>
      <th>232</th>
      <td>(Nutmeg)</td>
      <td>(Onion, Eggs, Milk, Yogurt)</td>
      <td>0.4</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.500000</td>
      <td>2.500000</td>
      <td>0.12</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>233</th>
      <td>(Yogurt)</td>
      <td>(Onion, Eggs, Milk, Nutmeg)</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.333333</td>
      <td>1.666667</td>
      <td>0.08</td>
      <td>1.200000</td>
    </tr>
    <tr>
      <th>234</th>
      <td>(Eggs)</td>
      <td>(Onion, Yogurt, Milk, Nutmeg)</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.250000</td>
      <td>1.250000</td>
      <td>0.04</td>
      <td>1.066667</td>
    </tr>
    <tr>
      <th>235</th>
      <td>(Onion)</td>
      <td>(Yogurt, Eggs, Milk, Nutmeg)</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.333333</td>
      <td>1.666667</td>
      <td>0.08</td>
      <td>1.200000</td>
    </tr>
  </tbody>
</table>
<p>236 rows × 9 columns</p>
</div>
