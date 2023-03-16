---
title: "[NLP] 카운트 기반의 단어 표현"
date: 2023-03-16

categories:
  - AI
  - Deep Learning
tags:
  - NLP
---



## 카운트 기반의 단어 표현
- 자연어 처리에서 텍스트를 표현하는 방법은 여러가지
- 정보 검색과 텍스트 마이닝분야에서 주로 사용되는 카운트 기반 텍스트 표현 방법
- `DTM(Document Term Matrix)`
- `TF-IDF(Term Frequency-Inverse Document Frequency)`
- `BoW(Bag of Words)`는 국소 표현에 속하며 단어의 빈도수를 카운트하여 단어를 수치화 하는 단어 표현 방법

## 다양한 단어의 표현 방법
- 카운트 기반의 단어 표현 방법 외 다양한 단어 표현 방법
- 국소 표현 방법과 분산 표현 방법으로 나뉨
- 국소 표현 방법 : 단어 자체만 보고 특정값을 맵핑하여 단어를 표현
- 분산 표현 방법 : 단어를 표현하고자 주변을 참고하여 표현

## BoW(Bag of Words)
- 단어들의 가방
- 단어들의 순서를 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하는 텍스트 데이터의 수치화 표현 방법
- (1) 각 단어에 고유한 정수 인덱스를 부여
- (2) 각 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터를 만듦


```python
from konlpy.tag import Okt

okt = Okt()

def build_bag_of_words(document):
    # 온점 제거 및 형태소 분석
    document = document.replace('.', '')
    tokenized_document = okt.morphs(document)

    word_to_index = {}
    bow = []

    for word in tokenized_document:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)

            # BoW에 전부 기본값 1을 넣음
            bow.insert(len(word_to_index) - 1, 1)
        
        else:
            # 재등장하는 단어의 인덱스 위치에 +1
            index = word_to_index.get(word)
            bow[index] = bow[index] + 1

    return word_to_index, bow
```


```python
document = '정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.'
vocab, bow = build_bag_of_words(document)

print('Vocabulary : ', vocab)
print('Bag of Words : ', bow)
```

    Vocabulary :  {'정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9}
    Bag of Words :  [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
    

### CountVectorizer
- scikit-learn에서 제공하는 CountVectorizer


```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I wart your love. because I love you.']
vector = CountVectorizer()

# 단어 빈도수
print('BoW : ', vector.fit_transform(corpus).toarray())

# 각 단어의 인덱스
print('Vocab : ', vector.vocabulary_)
```

    BoW :  [[1 1 2 1 2 1]]
    Vocab :  {'you': 4, 'know': 1, 'wart': 3, 'your': 5, 'love': 2, 'because': 0}
    

### 불용어를 제거한 BoW


```python
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
```

#### 1. 사용자가 직접 정의한 불용어


```python
text = ['Family is not an important thing. It\'s everything.']

stop_words = ['the', 'a', 'an', 'is', 'not']
vector = CountVectorizer(stop_words=stop_words)

print('BoW : ', vector.fit_transform(text).toarray()) # 단어 빈도수
print('Vocab : ', vector.vocabulary_) # 각 단어의 인덱스
```

    BoW :  [[1 1 1 1 1]]
    Vocab :  {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}
    

#### 2. CountVectorizer 에서 제공하는 불용어


```python
text = ['Family is not an important thing. It\'s everything.']
vector = CountVectorizer(stop_words='english')

print('BoW : ', vector.fit_transform(text).toarray()) # 단어 빈도수
print('Vocab : ', vector.vocabulary_) # 각 단어의 인덱스
```

    BoW :  [[1 1 1]]
    Vocab :  {'family': 0, 'important': 1, 'thing': 2}
    

#### 3. NLTK에서 제공하는 불용어


```python
from nltk.corpus import stopwords

text = ['Family is not an important thing. It\'s everything.']

stop_words = stopwords.words('english')
vector = CountVectorizer(stop_words=stop_words)

print('BoW : ', vector.fit_transform(text).toarray()) # 단어 빈도수
print('Vocab : ', vector.vocabulary_) # 각 단어의 인덱스
```

    BoW :  [[1 1 1 1]]
    Vocab :  {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
    

## 문서 단어 행렬(DTM)
- Document-Term Matrix, DTM
- 서로 다른 문서들의 BoW들을 결합한 표현 방법
- 다수의 문서에서 등장하는 각 단어들의 빈도를 행렬로 표현한 것
- 각 문서에 대한 BoW를 하나의 행렬로 만든 것
- BoW와 다른 표현 방법이 아닌, BoW 표현을 다수의 문서에 행렬로 표현한 것
- 문서 단어 행렬은 문서들을 서로 비교할 수 있도록 수치화할 수 있다는 점에 의의


```python
corpus = ['Think like a man of action and act like man of thought.',
          'Try not to become a man of success but rather try to becom a man of value.',
          'Give me liberty, of give me death']

vector = CountVectorizer(stop_words='english')
bow = vector.fit_transform(corpus)

print(bow.toarray())
print(vector.vocabulary_)
```

    [[1 1 0 0 0 2 2 0 1 1 0 0]
     [0 0 1 0 0 0 2 1 0 0 2 1]
     [0 0 0 1 1 0 0 0 0 0 0 0]]
    {'think': 8, 'like': 5, 'man': 6, 'action': 1, 'act': 0, 'thought': 9, 'try': 10, 'success': 7, 'becom': 2, 'value': 11, 'liberty': 4, 'death': 3}
    


```python
import pandas as pd

columns = []
for k, v in sorted(vector.vocabulary_.items(), key=lambda item:item[1]):
    columns.append(k)

df = pd.DataFrame(bow.toarray(), columns=columns)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>act</th>
      <th>action</th>
      <th>becom</th>
      <th>death</th>
      <th>liberty</th>
      <th>like</th>
      <th>man</th>
      <th>success</th>
      <th>think</th>
      <th>thought</th>
      <th>try</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### DTM의 한계
##### 1. 희소 표현
- 원-핫 벡터는 공간적 낭비와 계산 리소스를 증가시킬 수 있다는 점에서 단점
- 각 문서 벡터의 차원은 원-핫 벡터와 마찬가지로 전체 단어 집합의 크기를 가짐
- 전체 코퍼스가 방대한 데이터라면 문서 벡터의 차원은 수만 이상의 차원을 가질 수 있음
- 또한 많은 문서 벡터가 대부분의 값이 0을 가질 수도 있음

#### 2. 단순 빈도수 기반 접근
- 빈도 표기법은 때로 한계를 갖기도 함
- 영어에 대해 DTM을 만들었을 때, 불용어인 the는 어떤 문서이든 자주 등장
- 문서1, 문서2, 문서3에 the 빈도수가 높다고 해서 이 문자들이 유사한 문서는 아님
- 불용어와 중요한 단어에 대해 가중치를 줄 수 있는 방법인 `TF-IDF` 사용

## TF-IDF
- 단어의 빈도와 역문서 빈도를 사용하여 DTM 내 각 단어들마다 중요한 정도를 가중치로 주는 방법
- 단순히 빈도수가 높은 단어가 핵심이 아닌, 특정 문서에서만 집중적으로 등장할 때 해당 단어가 문서의 주제를 잘 담고 있는 핵심이라고 가정
- 특정 문서에서 특정 단어가 많이 등장하고, 그 단어가 다른 문서에서 적게 등장할 때, 그 단어를 특정 문서의 핵심어로 간주
- 어휘 빈도-문서 역빈도는 어휘 빈도와 역문서 빈도를 곱해 계산
- `어휘 빈도` : 특정 문서에서 특정 단어가 많이 등장하는 것을 의미
- `역문서 빈도` : 다른 문서에서 등장하지 않는 단어의 빈도를 의미
- `TF-IDF`를 계산하기 위해 `scikit-learn`의 `Tfidfvectorizer` 이용

### 1. DTM


```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'you know I want your love',
    'I line you',
    'what should I do'
]

vector = CountVectorizer()

# 단어 빈도 수
print(vector.fit_transform(corpus).toarray())

# 각 단어와 맵핑된 인덱스
print(vector.vocabulary_)
```

    [[0 1 0 1 0 1 0 1 1]
     [0 0 1 0 0 0 0 1 0]
     [1 0 0 0 1 0 1 0 0]]
    {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'line': 2, 'what': 6, 'should': 4, 'do': 0}
    

### 2. TF-IDF


```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'you know I want your love',
    'I line you',
    'what should I do'
]

tfidf = TfidfVectorizer().fit(corpus)

print(tfidf.transform(corpus).toarray())
print(tfidf.vocabulary_)
```

    [[0.         0.46735098 0.         0.46735098 0.         0.46735098 0.         0.35543247 0.46735098]
     [0.         0.         0.79596054 0.         0.         0.         0.         0.60534851 0.        ]
     [0.57735027 0.         0.         0.         0.57735027 0.         0.57735027 0.         0.        ]]
    {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'line': 2, 'what': 6, 'should': 4, 'do': 0}
