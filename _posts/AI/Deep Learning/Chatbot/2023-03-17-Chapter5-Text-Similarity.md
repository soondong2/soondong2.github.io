---
title: "[Chatbot] Chapter5 텍스트 유사도"
date: 2023-03-17

categories:
  - AI
  - Deep Learning
tags:
  - NLP
  - Chatbot
---

## 텍스트 유사도
- 임베딩으로 각 단어들의 벡터를 구한 다음 벡터 간의 거리를 계산하여 단어 간의 유사성 계산 가능
- 유사도 계산을 위해 단어들을 수치화해야 함
- 언어 모델에 따라 통계 / 인공 신경망 이용 방법으로 나뉨

### 1. n-gram
- n-gram은 주어진 문장에서 n개의 연속적인 단어 시퀀스를 의미
- 문장에서 n개의 단어를 토큰으로 사용
- 이웃한 단어의 출현 횟수를 통계적으로 표현해 텍스트의 유사도를 계산
- 논문 인용, 도용 정도 조사 가능
- 모든 단어의 출현 빈도가 아닌 연속되는 문장에서 일부 단어(n만큼)만 확인하므로 정확도가 떨어질 수 있음
- 일반적으로 n은 1 ~ 5 사이 값을 사용


```python
from konlpy.tag import Komoran

# 어절 단위 n-gram
def word_ngram(bow, num_gram):
    text = tuple(bow)
    ngrams = [text[x:x + num_gram] for x in range(0, len(text))]
    return tuple(ngrams)
```


```python
# 유사도 계산
def similarity(doc1, doc2):
    cnt = 0
    for token in doc1:
        if token in doc2:
            cnt += 1
    return cnt / len(doc1)
```


```python
# 문장 정의
sentence1 = '6월에 뉴턴은 선생님의 제안으로 트리니티에 입학했다.'
sentence2 = '6월에 뉴턴은 선생님의 제안으로 대학교에 입학했다.'
sentence3 = '나는 맛있는 밥을 뉴턴 선생님과 함께 먹었다.'
```


```python
# 형태소 분석기에서 명사 추출
komoran = Komoran()
bow1 = komoran.nouns(sentence1)
bow2 = komoran.nouns(sentence2)
bow3 = komoran.nouns(sentence3)

print(bow1)
```

    ['6월', '뉴턴', '선생님', '제안', '트리니티', '입학']
    


```python
# 단어 n-gram 토큰 추출 (2-gram 방식)
doc1 = word_ngram(bow1, 2)
doc2 = word_ngram(bow2, 2)
doc3 = word_ngram(bow3, 2)

print(doc1)
```

    (('6월', '뉴턴'), ('뉴턴', '선생님'), ('선생님', '제안'), ('제안', '트리니티'), ('트리니티', '입학'), ('입학',))
    


```python
# 유사도 계산
r1 = similarity(doc1, doc2)
r2 = similarity(doc3, doc1)

print(r1)
print(r2)
```

    0.6666666666666666
    0.0
    

### 2. 코사인 유사도
- 두 벡터 간 코사인 각도를 이용한 유사도 측정 방법
- 출현 빈도를 통해 유사도를 계산하면 동일한 단어가 많이 포함될수록 벡터의 크기가 커짐
- 코사인 유사도는 벡터 크기와 상관 없이 안정적

<br>

- 두 벡터의 방향이 동일할 경우 `1`
- 두 벡터의 방향이 반대일 경우 `-1`
- 두 벡터가 서로 직각일 경우 `0`
- 코사인은 `-1 ~ 1` 사이의 값을 가짐 


```python
from konlpy.tag import Komoran
import numpy as np
from numpy.linalg import norm
```


```python
# 코사인 유사도 계산
def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
```


```python
# DTM 만들기
def make_term_doc_mat(sentence_bow, word_dics):
    freq_mat = {}

    for word in word_dics:
        freq_mat[word] = 0
    
    for word in word_dics:
        if word in sentence_bow:
            freq_mat[word] += 1
    
    return freq_mat
```


```python
# 단어 벡터 만들기
def make_vector(dtm):
    vec = []
    for key in dtm:
        vec.append(dtm[key])
    return vec
```


```python
# 문장 정의
sentence1 = '6월에 뉴턴은 선생님의 제안으로 트리니티에 입학했다.'
sentence2 = '6월에 뉴턴은 선생님의 제안으로 대학교에 입학했다.'
sentence3 = '나는 맛있는 밥을 뉴턴 선생님과 함께 먹었다.'
```


```python
# 형태소 분석기를 이용해 단어 묶음 리스트 생성
komoran = Komoran()
bow1 = komoran.nouns(sentence1)
bow2 = komoran.nouns(sentence2)
bow3 = komoran.nouns(sentence3)

print(bow1)
print(bow2)
print(bow3)
```

    ['6월', '뉴턴', '선생님', '제안', '트리니티', '입학']
    ['6월', '뉴턴', '선생님', '제안', '대학교', '입학']
    ['밥', '뉴턴', '선생', '님과 함께']
    


```python
# 단어 묶음 리스트를 하나로 합침
bow = bow1 + bow2 + bow3
print(bow)
```

    ['6월', '뉴턴', '선생님', '제안', '트리니티', '입학', '6월', '뉴턴', '선생님', '제안', '대학교', '입학', '밥', '뉴턴', '선생', '님과 함께']
    


```python
# 단어 묶음에서 중복을 제거해 단어 사전 구축
word_dics = []
for token in bow:
    if token not in word_dics:
        word_dics.append(token)

print(word_dics)
```

    ['6월', '뉴턴', '선생님', '제안', '트리니티', '입학', '대학교', '밥', '선생', '님과 함께']
    


```python
# 문장별 단어 문서 행렬 계산
freq_list1 = make_term_doc_mat(bow1, word_dics)
freq_list2 = make_term_doc_mat(bow2, word_dics)
freq_list3 = make_term_doc_mat(bow3, word_dics)

print(freq_list1)
print(freq_list2)
print(freq_list3)
```

    {'6월': 1, '뉴턴': 1, '선생님': 1, '제안': 1, '트리니티': 1, '입학': 1, '대학교': 0, '밥': 0, '선생': 0, '님과 함께': 0}
    {'6월': 1, '뉴턴': 1, '선생님': 1, '제안': 1, '트리니티': 0, '입학': 1, '대학교': 1, '밥': 0, '선생': 0, '님과 함께': 0}
    {'6월': 0, '뉴턴': 1, '선생님': 0, '제안': 0, '트리니티': 0, '입학': 0, '대학교': 0, '밥': 1, '선생': 1, '님과 함께': 1}
    


```python
# 문장 벡터 생성
doc1 = np.array(make_vector(freq_list1))
doc2 = np.array(make_vector(freq_list2))
doc3 = np.array(make_vector(freq_list3))

print(doc1)
print(doc2)
print(doc3)
```

    [1 1 1 1 1 1 0 0 0 0]
    [1 1 1 1 0 1 1 0 0 0]
    [0 1 0 0 0 0 0 1 1 1]
    


```python
# 코사인 유사도 계산
r1 = cos_sim(doc1, doc2)
r2 = cos_sim(doc3, doc1)

print(r1)
print(r2)
```

    0.8333333333333335
    0.20412414523193154
