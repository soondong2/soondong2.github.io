---
title: "[NLP] 텍스트 전처리(Text Preprocessing)2"
date: 2023-03-15

categories:
  - AI
  - Deep Learning
tags:
  - NLP
---

## 정수 인코딩(Integer Encoding)
- 컴퓨터는 텍스트보다 숫자를 더 잘 처리함
- 각 단어를 고유한 정수에 맵핑시키는 전처리 작업
- 보통은 단어 등장 빈도수를 기준으로 정렬한 뒤에 부여

- 단어를 빈도수 순으로 정렬한 단어 집합(vocabulary) 생성
- 빈도수가 높은 순서대로 차례로 낮은 숫자부터 정수를 부여


```python
from nltk.tokenize import sent_tokenize # 문장 토큰화
from nltk.tokenize import word_tokenize # 단어 토큰화
from nltk.corpus import stopwords # 불용어
```


```python
text = 'A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain.'
```


```python
# 문장 토큰화
sentences = sent_tokenize(text)
print(sentences)
```

    ['A barber is a person.', 'a barber is good person.', 'a barber is huge person.', 'he Knew A Secret!', 'The Secret He Kept is huge secret.', 'Huge secret.', 'His barber kept his word.', 'a barber kept his word.', 'His barber kept his secret.', 'But keeping and keeping such a huge secret to himself was driving the barber crazy.', 'the barber went up a huge mountain.']
    

- 정제 & 정규화 작업을 병행한 단어 토큰화
- 소문자화하여 단어의 개수를 통일
- 불영어와 단어 길이가 2이하인 경우 제외


```python
# 단어 토큰화
vocab = {} 
preprocessed_sentences = []
stop_words = stopwords.words('english')

for sentence in sentences:
    # 단어 토큰화 작업 단계
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence:
        # 모든 단어를 소문자화
        word = word.lower()
        # 불용어와 단어 길이가 2이하인 경우 제거
        if word not in stop_words and len(word) > 2:
            result.append(word)
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    preprocessed_sentences.append(result)

print(preprocessed_sentences)
```

    [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
    

### 1. Dictionary


```python
# 단어 빈도수
print(vocab)
```

    {'barber': 8, 'person': 3, 'good': 1, 'huge': 5, 'knew': 1, 'secret': 6, 'kept': 4, 'word': 2, 'keeping': 2, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1}
    

- 빈도수가 높은 순서대로 정렬


```python
vocab_sorted = sorted(vocab.items(), key=lambda x:x[1], reverse=True)
print(vocab_sorted)
```

    [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3), ('word', 2), ('keeping', 2), ('good', 1), ('knew', 1), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)]
    

- 빈도수가 낮은 단어는 제외하고 정수 인코딩
- 등장 빈도가 낮은 단어는 자연어 처리에서 의미를 가지지 않을 가능성이 높기 때문


```python
# 정수 인코딩
word_to_index = {}
i = 0

for word, frequence in vocab_sorted:
    if frequence > 1:
        i += 1
        word_to_index[word] = i

print(word_to_index)
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7}
    

- 단어를 모두 사용하기 보다는 빈도수가 가장 높은 n개의 단어만 사용하고 싶은 경우
- 위 단어들은 빈도수가 높은 순으로 낮은 정수가 부여되어져 있으므로 빈도수 상위 n개의 단어만 사용하고 싶다고하면 vocab에서 정수값이 1부터 n까지인 단어들만 사용


```python
vocab_size = 5

# 상위 5개만 추출(index6 부터는 제외)
word_frequence = [word for word, index in word_to_index.items() if index >= vocab_size + 1]

# 해당 단어에 대한 index 정보 삭제
for w in word_frequence:
    del word_to_index[w]

print(word_to_index)
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
    

- 단어 집합에 존재하지 않는 단어들이 생기는 상황을 `OOV` 문제라고 함
- 단어 집합에 없는 단어들은 `OOV`의 인덱스로 인코딩


```python
# OOV
word_to_index['OOV'] = len(word_to_index) + 1
print(word_to_index)
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'OOV': 6}
    

- 단어 토큰화가 된 상태로 저장된 각 단어를 정수 인코딩
- (예) ['barber', 'person'] -> [1, 5]


```python
preprocessed_sentences
```




    [['barber', 'person'],
     ['barber', 'good', 'person'],
     ['barber', 'huge', 'person'],
     ['knew', 'secret'],
     ['secret', 'kept', 'huge', 'secret'],
     ['huge', 'secret'],
     ['barber', 'kept', 'word'],
     ['barber', 'kept', 'word'],
     ['barber', 'kept', 'secret'],
     ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
     ['barber', 'went', 'huge', 'mountain']]




```python
# 정수 인코딩
encoded_sentences = []

for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            # 단어 집합에 있는 단어라면 해당 단어의 정수를 리턴
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            # 단어 집합에 없는 단어라면 'OOV'의 정수를 리턴
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)

print(encoded_sentences)
```

    [[1, 5], [1, 6, 5], [1, 3, 5], [6, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [6, 6, 3, 2, 6, 1, 6], [1, 6, 3, 6]]
    

### 2. Counter
- 위에서는 파이썬의 dictionary 자료형으로 정수 인코딩
- 이번에는 좀 더 쉽게 `Counter`, `FreqDist`, `enumerate`, `keras`의 `tokenizer` 사용


```python
from collections import Counter
```


```python
print(preprocessed_sentences)
```

    [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
    

- 단어 집합(vocabulary)을 만들기 위해 문장의 경계인 `,`를 제거하고 단어들을 하나의 리스트로 만듦


```python
all_words_list = sum(preprocessed_sentences, [])
print(all_words_list)
```

    ['barber', 'person', 'barber', 'good', 'person', 'barber', 'huge', 'person', 'knew', 'secret', 'secret', 'kept', 'huge', 'secret', 'huge', 'secret', 'barber', 'kept', 'word', 'barber', 'kept', 'word', 'barber', 'kept', 'secret', 'keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy', 'barber', 'went', 'huge', 'mountain']
    

- 파이썬의 `Counter()`의 입력으로 사용하면 중복을 제거하고 단어의 빈도수를 기록


```python
# 빈도수
vocab = Counter(all_words_list)
print(vocab)
```

    Counter({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1})
    

- `most_common()` : 상위 빈도수를 가진 주어진 수의 단어만을 리턴


```python
# 상위 5개의 단어만 단어 집합으로 저장
vocab_size = 5
vocab = vocab.most_common(vocab_size)
print(vocab)
```

    [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]
    

- 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스 부여


```python
word_to_index = {}
i = 0

for word, frequency in vocab:
    i += 1
    word_to_index[word] = i

print(word_to_index)
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
    

### 3. NLTK의 FreqDist
- `NLTK`에서는 빈도수 계산 도구인 `FreqDist()`를 지원


```python
from nltk import FreqDist
import numpy as np
```


```python
# np.hstack으로 문장 구분을 제거 후 빈도수 계싼
vocab = FreqDist(np.hstack(preprocessed_sentences))
```


```python
# barber 라는 단어의 빈도수 출력
print(vocab['barber'])
```

    8
    


```python
# 상위 5개의 단어만 단어 집합으로 저장
vocab_size = 5
vocab = vocab.most_common(vocab_size)
print(vocab)
```

    [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]
    


```python
# enumerate사용하여 인덱스 부여
word_to_index = {word[0] : index+1 for index, word in enumerate(vocab)}
print(word_to_index)
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
    

### 4. Keras의 Tokenizer


```python
from tensorflow.keras.preprocessing.text import Tokenizer
```


```python
# 위에서 전처리 과정을 거친 결과
print(preprocessed_sentences)
```

    [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
    


```python
tokenizer = Tokenizer()
```

- `fit_on_texts()` 안에 코퍼스를 입력하면 빈도수 기준으로 단어 집합 생성


```python
tokenizer.fit_on_texts(preprocessed_sentences)
```

- `tokenizer.word_index` : 각 단어에 인덱스가 어떻게 부여 되었는지 확인


```python
print(tokenizer.word_index)
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}
    

- `tokenizer.word_counts` : 각 단어의 빈도수


```python
print(tokenizer.word_counts)
```

    OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
    

- `tokenizer.texts_to_sequences()` : 코퍼스에 대해서 각 단어를 정해진 인덱스로 변환


```python
print(tokenizer.texts_to_sequences(preprocessed_sentences))
```

    [[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
    

- 앞서 빈도수가 높은 n개를 사용하기 위해 `vocab.most_common()`을 사용
- keras tokenizer에서는 `num_words=num` 옵션 사용


```python
# 상위 5개
vocab_size = 5
tokenizer = Tokenizer(num_words=vocab_size+1)
tokenizer.fit_on_texts(preprocessed_sentences)
```

- 실제 적용은 texts_to_sequences를 사용할 때 적용


```python
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(preprocessed_sentences))
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}
    OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
    [[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
    

- 모두 적용시키고 싶을 경우


```python
vocab_size = 5
words_frequency = [word for word, index in tokenizer.word_index.items() if index >= vocab_size + 1]

for word in words_frequency:
    del tokenizer.word_index[word]
    del tokenizer.word_counts[word]

print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(preprocessed_sentences))
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
    OrderedDict([('barber', 8), ('person', 3), ('huge', 5), ('secret', 6), ('kept', 4)])
    [[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
    

- keras는 단어 집합에 없는 단어인 `OOV`에 대해 정수로 바꾸는 과정에서 아예 단어를 제거한다는 특징
- OOV로 간주하고 보존하고 싶다면 Tokenizer의 인자 `oov_token`을 사용
- 기본적으로 OOV의 인덱스를 1로 함


```python
# 숫자 0과 OOV를 고려해서 단어 집합의 크기는 +2
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')
tokenizer.fit_on_texts(preprocessed_sentences)
```


```python
print('단어 OOV의 인덱스 : {}'.format(tokenizer.word_index['OOV']))
```

    단어 OOV의 인덱스 : 1
    


```python
# 코퍼스에 대해 정수 인코딩
print(tokenizer.texts_to_sequences(preprocessed_sentences))
```

    [[2, 6], [2, 1, 6], [2, 4, 6], [1, 3], [3, 5, 4, 3], [4, 3], [2, 5, 1], [2, 5, 1], [2, 5, 3], [1, 1, 4, 3, 1, 2, 1], [2, 1, 4, 1]]
    

## 패딩(Padding)
- 각 문장의 길이가 서로 다를 수 있음
- 기계는 하나의 행렬로 보고 한꺼번에 묶어서 처리
- 따라서 여러 문장의 길이를 임의로 동일하게 맞춰주는 작업이 필요


```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
```


```python
print(preprocessed_sentences)
```

    [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
    


```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)

print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(preprocessed_sentences))
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}
    OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
    [[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
    


```python
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
```


```python
# 문서 앞에 0으로 채우기
padded = pad_sequences(encoded)
padded
```




    array([[ 0,  0,  0,  0,  0,  1,  5],
           [ 0,  0,  0,  0,  1,  8,  5],
           [ 0,  0,  0,  0,  1,  3,  5],
           [ 0,  0,  0,  0,  0,  9,  2],
           [ 0,  0,  0,  2,  4,  3,  2],
           [ 0,  0,  0,  0,  0,  3,  2],
           [ 0,  0,  0,  0,  1,  4,  6],
           [ 0,  0,  0,  0,  1,  4,  6],
           [ 0,  0,  0,  0,  1,  4,  2],
           [ 7,  7,  3,  2, 10,  1, 11],
           [ 0,  0,  0,  1, 12,  3, 13]])




```python
# 문서 뒤에 0으로 채우기
padded = pad_sequences(encoded, padding='post')
padded
```




    array([[ 1,  5,  0,  0,  0,  0,  0],
           [ 1,  8,  5,  0,  0,  0,  0],
           [ 1,  3,  5,  0,  0,  0,  0],
           [ 9,  2,  0,  0,  0,  0,  0],
           [ 2,  4,  3,  2,  0,  0,  0],
           [ 3,  2,  0,  0,  0,  0,  0],
           [ 1,  4,  6,  0,  0,  0,  0],
           [ 1,  4,  6,  0,  0,  0,  0],
           [ 1,  4,  2,  0,  0,  0,  0],
           [ 7,  7,  3,  2, 10,  1, 11],
           [ 1, 12,  3, 13,  0,  0,  0]])



- `maxlen`의 인자로 정수를 주면, 해당 정수로 모든 문서의 길이를 동일하게 함


```python
# 길이가 5보다 짧은 문서들은 0으로 패딩
# 기존에 5보다 길었다면 데이터가 손실
padded = pad_sequences(encoded, padding='post', maxlen=5)
padded
```




    array([[ 1,  5,  0,  0,  0],
           [ 1,  8,  5,  0,  0],
           [ 1,  3,  5,  0,  0],
           [ 9,  2,  0,  0,  0],
           [ 2,  4,  3,  2,  0],
           [ 3,  2,  0,  0,  0],
           [ 1,  4,  6,  0,  0],
           [ 1,  4,  6,  0,  0],
           [ 1,  4,  2,  0,  0],
           [ 3,  2, 10,  1, 11],
           [ 1, 12,  3, 13,  0]])



- `truncating='post'`를 사용할 경우 뒤의 단어가 삭제


```python
padded = pad_sequences(encoded, padding='post', truncating='post', maxlen=5)
padded
```




    array([[ 1,  5,  0,  0,  0],
           [ 1,  8,  5,  0,  0],
           [ 1,  3,  5,  0,  0],
           [ 9,  2,  0,  0,  0],
           [ 2,  4,  3,  2,  0],
           [ 3,  2,  0,  0,  0],
           [ 1,  4,  6,  0,  0],
           [ 1,  4,  6,  0,  0],
           [ 1,  4,  2,  0,  0],
           [ 7,  7,  3,  2, 10],
           [ 1, 12,  3, 13,  0]])



## 원-핫 인코딩(One-Hot Encoding)
- 단어의 크기를 벡터의 차원으로 하고 표현하고 싶은 단어의 인덱스에 1의 값을 부여, 다른 인덱스에는 0을 부여

### 1. Keras를 활용한 One-Hot Encoding


```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
```


```python
text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
```


```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

print(tokenizer.word_index) # 단어 집합
```

    {'점심': 1, '먹으러': 2, '갈래': 3, '메뉴는': 4, '햄버거': 5, '최고야': 6}
    


```python
encoded = tokenizer.texts_to_sequences([text])[0]
print(encoded)
```

    [1, 2, 3, 4, 5, 6]
    


```python
# 원-핫 인코딩
one_hot = to_categorical(encoded)
print(one_hot)
```

    [[0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 1.]]
    

### 원-핫 인코딩의 한계
- 단어의 개수가 늘어날수록 벡터를 저장하기 위해 필요한 공간이 늘어남
- 원-핫 벡터는 단어의 유사도를 표현하지 못함
- 이러한 단점을 해결하기 위해 `LSA(잠재 의미 분석)`, `HAL`, `NNLM`, `RNNLM`, `Word2Vec`, `FastText`, `GloVe` 등을 사용

## 데이터 분리(Splitting Data)
- 모델을 학습시키고 평가하기 위해 데이터를 분리하는 작업 필요


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
```

### 1. zip


```python
X, y = zip(['a', 1], ['b', 2], ['c', 3])

print('X 데이터 :', X)
print('y 데이터 :', y)
```

    X 데이터 : ('a', 'b', 'c')
    y 데이터 : (1, 2, 3)
    


```python
sequences = [['a', 1], ['b', 2], ['c', 3]]
X, y = zip(*sequences)

print('X 데이터 :',X)
print('y 데이터 :',y)
```

    X 데이터 : ('a', 'b', 'c')
    y 데이터 : (1, 2, 3)
    

### 2. Data Frame


```python
values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>메일 본문</th>
      <th>스팸 메일 유무</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>당신에게 드리는 마지막 혜택!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내일 뵐 수 있을지 확인 부탁드...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>도연씨. 잘 지내시죠? 오랜만입...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(광고) AI로 주가를 예측할 수 있다!</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df['메일 본문']
y = df['스팸 메일 유무']

print('X 데이터 :',X.to_list())
print('y 데이터 :',y.to_list())
```

    X 데이터 : ['당신에게 드리는 마지막 혜택!', '내일 뵐 수 있을지 확인 부탁드...', '도연씨. 잘 지내시죠? 오랜만입...', '(광고) AI로 주가를 예측할 수 있다!']
    y 데이터 : [1, 0, 0, 1]
    

### 2. scikit-learn


```python
# 임의로 X와 y 데이터를 생성
X, y = np.arange(10).reshape((5, 2)), range(5)

print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))
```

    X 전체 데이터 :
    [[0 1]
     [2 3]
     [4 5]
     [6 7]
     [8 9]]
    y 전체 데이터 :
    [0, 1, 2, 3, 4]
    


```python
# 7:3의 비율로 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
```


```python
print('X 훈련 데이터 :')
print(X_train)
print('X 테스트 데이터 :')
print(X_test)
```

    X 훈련 데이터 :
    [[2 3]
     [4 5]
     [6 7]]
    X 테스트 데이터 :
    [[8 9]
     [0 1]]
    


```python
print('y 훈련 데이터 :')
print(y_train)
print('y 테스트 데이터 :')
print(y_test)
```

    y 훈련 데이터 :
    [1, 2, 3]
    y 테스트 데이터 :
    [4, 0]
