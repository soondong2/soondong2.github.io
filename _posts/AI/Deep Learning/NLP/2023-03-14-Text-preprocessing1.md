---
title: "[NLP] 텍스트 전처리(Text Preprocessing)1"
date: 2023-03-14

categories:
  - AI
  - Deep Learning
tags:
  - NLP
---

# 자연어 처리(NLP)
-  자연어란 우리가 일상 생활에서 사용하는 언어
- 자연어 처리란 자연어의 의미를 분석하여 컴퓨터가 처리할 수 있도록 하는 일
- 활용 분야 : 음성 인식, 내용 요약, 번역, 감성 분석, 텍스트 분류, 질의 응답, 챗봇

# 텍스트 전처리(Text Preprocessing)
- 풀고자 하는 문제의 용도에 맞게 텍스트를 사전에 처리하는 작업
- 텍스트 전처리를 제대로 하지 않으면 자연어 처리 기법들이 제대로 동작하지 않음

## 토큰화(Tokenization)
- 자연어 처리에서 코퍼스 데이터가 필요에 맞게 전처리 되지 않은 상태라면, 해당 데이터를 사용하고자 하는 용도에 맞게 `토큰화(tokenization)`, `정제(cleaning)`, `정규화(normalization)` 해야 한다.
- 주어진 코퍼스(corpus)에서 토큰(token)이라 불리는 단위로 나누는 작업을 토큰화라고 한다.
- 보통 의미 있는 단위로 토큰을 정의한다.
- `NLTK`, '`KoNLPy` 패키지 사용

### 1. 단어 토큰화(Word Tokenization)
- 토큰의 기준을 단어로 하는 경우
- `NLTK`는 영어 코퍼스를 토큰화하기 위한 도구들을 제공한다.
- `word_tokenize`, `WordPunctTokenizer`

#### 토큰화 중 생기는 선택의 순간
- 아래의 문장에서 `Don't`와 `Jone's`는 어떻게 토큰화할 수 있을까?
- 다양한 선택지의 예시는 다음과 같다.
- Don't, Don t, Dont, Do n't 등


```python
corpus = 'Don\'t be fooled by the dark sounding name, Mr. Jone\'s Orphanage is as cheery as cheery goes for a pastry shop.'
print(corpus)
```

    Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.
    


```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
```

- `word_tokenize`는 `Don't`를 Do, n't로 분리하였다.
- 반면 Jone's는 Jone, 's로 분리하였다.


```python
# word_tokenize 사용
print(word_tokenize(corpus))
```

    ['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
    

- `WordPunctTokenizer`는 Don't를 Don, ', t로 분리하였다.


```python
# WordPunctTokenizer 사용
print(WordPunctTokenizer().tokenize(corpus))
```

    ['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
    

- 케라스의 `text_to_word_sequence`는 모든 알파벳을 소문자로 바꾼다.
- 마침표, 컴마, 느낌표 등의 구두점을 제거한다.
- 그러나 아포스트로피는 보존된다(`'`)


```python
print(text_to_word_sequence(corpus))
```

    ["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
    

#### 토큰화 고려 사항
- 구두점이나 특수 문자를 단순 제외해서는 안 된다.
- 줄임말과 단어 내에 띄어쓰기가 있는 경우

### 2. 문장 토큰화(Sentence Tokenization)
- 문장 단위로 구분하는 작업
- 문장 분류 라고도 함
- `NLTK`에서 영어 문장 토큰화를 수행하는 `senttokenize`를 사용
- `KSS`는 한국어 문장 토큰화를 수행


```python
from nltk.tokenize import sent_tokenize

text = 'His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near'

print(sent_tokenize(text))
```

    ['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near']
    


```python
text = 'I am actively looking for Ph.D. students. and you are a Ph.D student.'

print(sent_tokenize(text))
```

    ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
    


```python
# 한국어
import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'

print(kss.split_sentences(text))
```

    ['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다.', '이제 해보면 알걸요?']
    

#### 한국어 토큰화의 어려움
- 영어는 `New Tork` 같은 합성어나 `he's` 같은 줄임말에 대한 예외처리만 한다면 띄어쓰기(whitespace)를 기준으로 하는 토큰화를 수행해도 단어 토큰화가 잘 작동한다. 대부분의 경우에서 단어 단위로 띄어쓰기가 이루어지기 때문이다.
- 한국어는 영어와 달리 띄어쓰기만으로는 토큰화를 하기에 부족하다.
- 한국어의 경우 띄어쓰기 단위가 되는 단위를 `어절`이라고 하는데, 어절 토큰화는 한국어 NLP에서 지양되고 있다. 어절 토큰화와 단어 토큰화는 같지 않기 때문이다.
- 근본적인 이유는 한국어가 영어와는 다른 형태를 가지는 언어인 교착어라는 점에서 기인합니다. `교착어`란 `조사`, `어미` 등을 붙여서 말을 만드는 언어를 말합니다.

#### 형태소
- `형태소`란 뜻을 가진 가장 작은 말의 단위
- `자립 형태소` : 접사, 어미, 조사와 상관 없이 자립하여 사용할 수 있는 형태소이다. 명사, 대명사, 관형사, 부사, 감탄사 등
- `의존 형태소` : 다른 형태소와 결합하여 사용되는 형태소이다. 접사, 어미, 조사, 어간 등
- (예) 에디가 책을 읽었다.
- 자립 형태소 : 에디, 책
- 의존 형태소 : -가, -을, 읽-, -었, -다

### 3. 품사 태깅(Part-of-speech tagging)
- 단어는 표기는 같지만 품사에 따라 의미가 달라지기도 함
- (예) 영어 단어 `fly`는 동사로는 날다, 명사로는 파리
- (예) 한국어 단어 `못`은 물건, 동작 동사를 할 수 없다
- 해당 단어가 어떤 품사로 쓰였는지 보여주는 것이 주요 지표가 될 수도 있다.
- `NLTK`의 `pos_tag`을 통해 품사 태깅
- `KoNLPy`의 `Okt`, `Komoran`, `Hananum`, `Kkma`, `Mecab`을 통해 품사 태깅

<br>

|품사|내용|
|:---:|:---:|
|PRP|인칭대명사|
|VBP|동사|
|RB|부사|
|VBG|현재부사|
|IN|전치사|
|NNP|고유 명사|
|NNS|복수형 명사|
|CC|접속사|
|DT|관사|


```python
# nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = 'I am actively looking for Ph.D. students. and you are a Ph.D. student.'

print('단어 토큰화 : ', word_tokenize(text))
print('품사 태킹 : ', pos_tag(word_tokenize(text)))
```

    단어 토큰화 :  ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
    품사 태킹 :  [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
    

- `morphs` : 형태소 추출
- `pos` : 품사 태깅
- `nouns` : 명사 추출


```python
# KoNLPy - Okt
from konlpy.tag import Kkma, Okt

okt = Okt()

text = '열심히 코딩한 당신, 연휴에는 여행을 가봐요'

print('Okt 형태소 분석 : ', okt.morphs(text))
print('Okt 품사 태깅 : ', okt.pos(text))
print('Okt 명사 추출: ', okt.nouns(text))
```

    Okt 형태소 분석 :  ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
    Okt 품사 태깅 :  [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
    Okt 명사 추출:  ['코딩', '당신', '연휴', '여행']
    


```python
# KoNLPy - Kkma
from konlpy.tag import Kkma, Okt

kkma = Kkma()

text = '열심히 코딩한 당신, 연휴에는 여행을 가봐요'

print('Kkma 형태소 분석 : ', kkma.morphs(text))
print('Kkma 품사 태깅 : ', kkma.pos(text))
print('Kkma 명사 추출: ', kkma.nouns(text))
```

    Kkma 형태소 분석 :  ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
    Kkma 품사 태깅 :  [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
    Kkma 명사 추출:  ['코딩', '당신', '연휴', '여행']
    

- 각 형태소 분석기는 성능과 결과가 다르게 나오기 때문에, 형태소 분석기의 선택은 사용하고자 하는 필요 용도에 어떤 형태소 분석기가 가장 적절한지를 판단하고 사용하면 된다.

## 정제(Cleaning) & 정규화(Normalization)
- 코퍼스에서 용도에 맞게 토큰을 분류하는 작업을 `토큰화`
- 토큰화 작업 전, 후에는 텍스트 데이터를 용도에 맞게 정제 및 정규화 해야 함
- `정제` : 갖고 있는 코퍼스로부터 노이즈 데이터를 제거
- `정규화` : 표현 방법이 다른 단어들을 통합시켜 같은 단어로 만듦

### 1. 규칙에 기반한 표기가 다른 단어들의 통합
- USA와 US는 같은 의미를 가지므로 하나의 단어로 정규화
- 표기가 다른 단어들을 통합하는 방법인 `어간 추출(stemming)`과 `표제어 추출(lemmatizaiton)`

### 2. 대소문자 통합
### 3. 불필요한 단어 제거
- 등장 빈도가 적은 단어
- 길이가 짧은 단어


```python
# 길이가 1 ~ 2인 단어들은 정규 표현식을 이용하여 삭제
import re

text = 'I was wondering if anyone out there could enlighten me on this car.'"Anaconda Navigator (anaconda3).lnk"
r = re.compile(r'\W*\b\w{1,2}\b')
print(r.sub('', text))
```

     was wondering anyone out there could enlighten this car.Anaconda Navigator (anaconda3).lnk
    

### 4. 정규표현식

## 어간 추출(Stemming) & 표제어 추출(Lemmatization)
- 하나의 단어로 일반화 시켜서 문서 내 단어 수를 줄이는 것

### 1. 표제어 추출(Lemmatization)
- 단어들이 다른 형태를 가지더라도 뿌리 단어를 찾아가서 단어의 개수를 줄일 수 있는지 판단
- (예) am, are, is는 서로 다른 스펠링이지만, 이 단어들의 표제어는 be
- 어간 추출과는 달리 단어의 형태가 적절히 보존되는 특징
- `NLTK`의 `WordNetLemmatizer`를 사용


```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('표제어 추출 전 : ', words)
print('표제어 추출 후 : ', [lemmatizer.lemmatize(word) for word in words])
```

    표제어 추출 전 :  ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
    표제어 추출 후 :  ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
    

- dies를 dy로, has를 ha로 의미를 알 수 없는 적절하지 못한 단어 출력됨
- 이는 표제어 추출기가 본래 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있음


```python
print(lemmatizer.lemmatize('dies', 'v')) # 동사
print(lemmatizer.lemmatize('watched', 'v')) # 동사
print(lemmatizer.lemmatize('has', 'v')) # 동사
```

    die
    watch
    have
    

### 2. 어간 추출(Stemming)
- 어간을 추출하는 작업
- 포터 알고리즘의 어간 추출 규칙은 다음과 같다.
- ALIZE -> AL
- ANCE -> 제거
- ICAL -> IC


```python
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

sentence = 'This was not the map we found in Billy Bones\'s chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes.'

print('어간 추출 전 : ', word_tokenize(sentence))
print('어간 추출 후 : ', [stemmer.stem(word) for word in word_tokenize(sentence)])
```

    어간 추출 전 :  ['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']
    어간 추출 후 :  ['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']
    

## 불용어(Stopword)
- 유의미한 토큰만을 선별하기 위해 의미가 없는 단어 토큰을 제거
- NLTK에서는 100여개 이상의 영어 단어들을 불용어 패키지 내에서 미리 정의


```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt
```

### 1. NLTK로 영어 불용어 제거


```python
# 불용어 목록
print(stopwords.words('english'))
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    


```python
text = 'Family is not an important thing. It\'s everything.'

# 불용어
stop_words = stopwords.words('english')

result = []
for word in word_tokenize(text):
    if word not in stop_words:
        result.append(word)

print('불용어 제거 전 : ', word_tokenize(text))
print('불용어 제거 후 : ', result)
```

    불용어 제거 전 :  ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
    불용어 제거 후 :  ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
    

### 2. 한국어 불용어 제거
- 불용어를 직접 정의하고 제거
- 한국어 불용어 리스트 : https://www.ranks.nl/stopwords/korean


```python
from konlpy.tag import Okt

text = '고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지.'

# 불용어 설정
stop_words = '를 아무렇게나 구 우려 고 안 돼 같은 게 구울 때 는'
stop_words = stop_words.split(' ') # 공백으로 구분

result = [word for word in okt.morphs(text) if not word in stop_words]

print('불용어 제거 전 : ', okt.morphs(text))
print('불용어 제거 후 : ', result)
```

    불용어 제거 전 :  ['고기', '를', '아무렇게나', '구', '우려', '고', '하면', '안', '돼', '.', '고기', '라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살', '을', '구울', '때', '는', '중요한', '게', '있지', '.']
    불용어 제거 후 :  ['고기', '하면', '.', '고기', '라고', '다', '아니거든', '.', '예컨대', '삼겹살', '을', '중요한', '있지', '.']
    
