---
title: "[Chatbot] Chapter4 워드 임베딩"
date: 2023-03-17

categories:
  - AI
  - Deep Learning
tags:
  - NLP
  - Chatbot
---

## 임베딩
- 자연어를 숫자나 벡터 형태로 변환할 필요가 있음
- 임베딩이란 단어나 문장을 수치화해 벡터 공간으로 표현하는 과정
- 말뭉치 의미에 따라 벡터화하므로 문법 정보를 포함
- 문장 임베딩 / 단어 임베딩

### 단어 임베딩
- 말뭉치에서 각각의 단어를 벡터로 변환하는 기법

#### 1. 원-핫 인코딩
- 단 하나의 값만 `1`이고, 나머지는 `0` -> 희소벡터
- 단어 집합을 먼저 만들어야 함
- 고유한 인덱스 번호를 부여함
- 인덱스 번호가 원-핫 인코딩에서 1의 값을 갖는 위치가 됨


```python
from konlpy.tag import Komoran
import numpy as np

komoran = Komoran()
text = '오늘 날씨는 구름이 많아요.'

# 명사만 추출
print(komoran.nouns(text))

# 단어 사전 구축 및 단어별 인덱스 부여
dics = {}
for word in komoran.nouns(text):
    if word not in dics.keys():
        dics[word] = len(dics)
print(dics)

# 원-핫 인코딩
nb_classes = len(dics)
targets = list(dics.values())
one_hot = np.eye(nb_classes)[targets] # np.eye() : 단위 행렬
print(one_hot)
```

    ['오늘', '날씨', '구름']
    {'오늘': 0, '날씨': 1, '구름': 2}
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    

#### 2. 희소 표현과 분산 표현
- 원-핫 인코딩은 `희소 벡터(희소 행렬)`에 해당
- 단어가 희소 벡터로 표현되는 방식을 `희소 표현`이라고 함
- 각 차원이 독립적인 정보를 갖고 있지만, 단어 사전 크기가 커질수록 메모리 낭비와 계산 복잡도 증가
- 단어 간 연관성이 없음

<br>

- 희소표현의 단점을 해결하기 위해 단어 간 유사성을 잘 표현하고 공간을 절약하는 `분산 표현` 방법 등장
- 분산 표현이란 한 단어의 정보가 특정 차원에 표현되지 않고, 여러 차원에 분산되어 표현된다는 뜻
- 데이터 손실을 최소화 하면서 벡터 차원이 압축됨
- 임베딩 벡터에는 단어 의미, 주변 단어 간의 관계 등 많은 정보가 내포되어 일반화 능력이 뛰어남
- 벡터 공간 상에서 유사한 의미를 갖는 단어들은 비슷한 위치에 분포됨

#### 3. Word2Vec
- 신경망 기반 단어 임베딩의 대표적인 방법
- 단어 임베딩 모델
- `CBOW`와 `skip-gram` 두 가지 모델로 제안됨
- `CBOW` : 주변 단어를 이용해 타깃 단어를 예측
- `skip-gram` : 타깃 단어를 이용해 주변 단어들을 예측
- `윈도우` : 앞뒤로 몇 개의 단어까지 확인할지의 범위
- Word2Vec의 단어 임베딩은 해당 단어를 밀집 벡터로 표현하여 학습을 통해 의미상 비슷한 단어들을 비슷한 벡터 공간에 위치
- 의미에 따라 방향성을 가짐


```python
from gensim.models import Word2Vec
from konlpy.tag import Komoran
import time
```

```python
# 네이버 영화 리뷰 데이터
def read_review(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] # header 제거
    return data
```

```python
# 리뷰 파일 읽어오기
df = read_review('C:/Users/USER/Desktop/ratings.txt')

print(len(df)) # 데이터 개수
```
    200000
    16.601213693618774
    

```python
# 데이터 확인
df[0]
```

    ['8112052', '어릴때보고 지금다시봐도 재밌어요ㅋㅋ', '1']



```python
# 문장 단위로 명사만 추출해 학습 입력 데이터로 만듦
komoran = Komoran()

docs = [komoran.nouns(sentence[1]) for sentence in df]
print(docs)
```
[['때'], ['디자인', '학생', '외국', '디자이너', '전통', '발전', '문화', '산업', '사실', '우리나라', '시절', '끝', '열정', '노라', '노', '전통', '사람', '꿈', '수', '것', '감사'], ['폴리스', '스토리', '시리즈', '뉴', '최고'], ['연기', '것', '라고', '생각', '몰입', '영', '화지'], ['안개', '밤하늘', '초승달', '영화'], ... , [], ['완전', '사이코', '영화', '마지막', '영화', '질'], ['라따뚜이', '스머프'], ['포', '풍', '저그', '가나', '신다영', '차영', '차영', '차']]

- `sentences` : 모델 학습에 필요한 문장 데이터
- `size` : 단어 임베딩 벡터의 차원(크기)
- `window` : 주변 단어 윈도우의 크기
- `hs` : 0(0이 아닌 경우 음수 샘플링 사용), 1(모델 학습에 softmax 사용)
- `min_count` : 단어 최소 빈도수 제한(빈도수 이하 단어들은 학습하지 않음)
- `sg` : 0(CBOW모델), 1(skip-gram 모델)


```python
# Word2Vec 모델 학습
model = Word2Vec(sentences=docs, size=200, window=4, hs=1, min_count=2, sg=1)
```

```python
# 모델 저장
model.save('nvmc.model')

# 학습된 말뭉치 수, 코퍼스 내 전체 단어 수
print('Corpus Count : ', model.corpus_count)
print('Corpus Total Words : ', model.corpus_total_words)
```
    Corpus Count :  200000
    Corpus Total Words :  1076896
    
