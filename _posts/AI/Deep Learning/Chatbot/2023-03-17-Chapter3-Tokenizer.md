---
title: "[Chatbot] Chapter3 토크나이저"
date: 2023-03-17

categories:
  - AI
  - Deep Learning
tags:
  - NLP
  - Chatbot
---

## 토크나이저
- 가장 기본이 되는 단어들을 `토큰(token)`이라고 함
- 토큰의 단위는 토크나이징 방법에 따라 달라질 수 있지만 일반적으로 일정한 의미가 있는 가장 작은 정보 단위로 결정
- 주어진 문장에서 토큰 단위로 정보를 나누는 작업을 토크나이징이라고 함
- 토크나이징은 문장 형태의 데이터를 처리하기 위해 제일 처음 수행해야 하는 기본적인 작업
- 주로 텍스트 전처리 과정에서 사용
- 토크나이징을 어떻게 하느냐에 따라 성능 차이

## KoNLPy
- 한국어 자연어 처리 토크아니징을 지원하는 파이썬 라이브러리

### 1. Kkma

- Kkma 모듈의 함수 설명

|함수|설명|
|:---:|:---:|
|morphs|문장을 형태소 단위로 토크아니징|
|nouns|문장에서 품사가 명사인 토큰만 추출|
|pos|형태소를 추출한 뒤 품사 태깅|
|sentences|여러 문장들을 분리해주는 역할|

- Kkma 품사 태그

|품사|설명|
|:---:|:---:|
|NNG|일반 명사|
|JKS|주격 조사|
|JKM|부사격 조사|
|VV|동사|
|EFN|평서형 종결 어미|
|SF|마침표, 물음표, 느낌표|


```python
from konlpy.tag import Kkma

kkma = Kkma()
text = '아버지가 방에 들어갑니다.'

# 형태소 추출
print(kkma.morphs(text))

# 형태소와 품사 태크 추출
print(kkma.pos(text))

# 명사만 추출
print(kkma.nouns(text))

# 문장 분리
sentences = '오늘 날씨는 어때요? 내일은 덥다던데.'
print(kkma.sentences(sentences))
```

    ['아버지', '가', '방', '에', '들어가', 'ㅂ니다', '.']
    [('아버지', 'NNG'), ('가', 'JKS'), ('방', 'NNG'), ('에', 'JKM'), ('들어가', 'VV'), ('ㅂ니다', 'EFN'), ('.', 'SF')]
    ['아버지', '방']
    ['오늘 날씨는 어 때요?', '내일은 덥다 던데.']
    

### 2. Okt

- Okt 모듈의 함수 설명

|함수|설명|
|:---:|:---:|
|morphs|문장을 형태소 단위로 토크아니징|
|nouns|문장에서 품사가 명사인 토큰만 추출|
|pos|형태소를 추출한 뒤 품사 태깅|
|normalize|문장을 정규화|
|phrases|문장에서 어구 추출|

- Okt 품사 태그 표

|품사|설명|
|:---:|:---:|
|Noun|명사|
|Verb|동사|
|Adjective|형용사|
|Josa|조사|
|Punctuation|구두점|


```python
from konlpy.tag import Okt

okt = Okt()
text = '아버지가 방에 들어갑니다.'

# 형태소 추출
print(okt.morphs(text))

# 형태소와 품사 태그 추출
print(okt.pos(text))

# 명사만 추출
print(okt.nouns(text))

# 정규화, 어구 추출
text = '오늘 날씨가 좋아욬ㅋㅋ'

print(okt.normalize(text))
print(okt.phrases(text))
```

    ['아버지', '가', '방', '에', '들어갑니다', '.']
    [('아버지', 'Noun'), ('가', 'Josa'), ('방', 'Noun'), ('에', 'Josa'), ('들어갑니다', 'Verb'), ('.', 'Punctuation')]
    ['아버지', '방']
    오늘 날씨가 좋아요ㅋㅋ
    ['오늘', '오늘 날씨', '좋아욬', '날씨']
    

### 3. Komoran

- Komoran 모듈의 함수 설명

|함수|설명|
|:---:|:---:|
|morphs|문장을 형태소 단위로 토크아니징|
|nouns|문장에서 품사가 명사인 토큰만 추출|
|pos|형태소를 추출한 뒤 품사 태깅|

- Komoran 품사 태그

|품사|설명|
|:---:|:---:|
|NNG|일반 명사|
|JKS|주격 조사|
|JKB|부사격 조사|
|VV|동사|
|EF|종결 어미|
|SF|마침표, 물음표, 느낌표|


```python
from konlpy.tag import Komoran

komoran = Komoran()
text = '아버지가 방에 들어갑니다.'

# 형태소 추출
print(komoran.morphs(text))

# 형태소와 품사 태그 추출
print(komoran.pos(text))

# 명사만 추출
print(komoran.nouns(text))
```

    ['아버지', '가', '방', '에', '들어가', 'ㅂ니다', '.']
    [('아버지', 'NNG'), ('가', 'JKS'), ('방', 'NNG'), ('에', 'JKB'), ('들어가', 'VV'), ('ㅂ니다', 'EF'), ('.', 'SF')]
    ['아버지', '방']
    

### 4. 사용자 사전 구축
- 형태소 분석기에서 인식하지 못하는 단어들을 직접 추가하는 방법
- Komoran이 다른 형태소 분석기에 비해 사전을 관리하는 방법이 편리하고 성능과 속도가 괜찮은 편이므로 Komoran 기준으로 챗봇 만들 예정
<br>

- ('엔', 'NNB'), ('엘', 'NNP'), ('피', 'NNG')
- 엔엘피라는 단어를 엔, 엘, 피라는 문자로 분리해 명사로 인식함
- 챗봇의 경우 오류를 낼 확률이 높음
- 이를 해결하기 위해 Komoran의 사용자 사전에 '엔엘피'라는 신규 단어 등록


```python
from konlpy.tag import Komoran

komoran = Komoran()
text = '우리 챗봇은 엔엘피를 좋아해.'

print(komoran.pos(text))
```

    [('우리', 'NP'), ('챗봇은', 'NA'), ('엔', 'NNB'), ('엘', 'NNP'), ('피', 'NNG'), ('를', 'JKO'), ('좋아하', 'VV'), ('아', 'EF'), ('.', 'SF')]
    


```python
from konlpy.tag import Komoran

komoran = Komoran(userdic='C:/Users/USER/Desktop/userdic.txt')
text = '우리 챗봇은 엔엘피를 좋아해.'

print(komoran.pos(text))
```

    [('우리', 'NP'), ('챗봇은', 'NA'), ('엔엘피', 'NNG'), ('를', 'JKO'), ('좋아하', 'VV'), ('아', 'EF'), ('.', 'SF')]
    

- `userdic='C:/Users/USER/Desktop/userdic.txt'`
- 메모장을 활용해 다음과 같은 txt 파일을 생성하여 `userdic`에 설정
- 단어와 품사 사이 간격은 반드시 `탭(Tab)`으로 구분할 것

![image](https://user-images.githubusercontent.com/100760303/225826695-3457e617-0702-4676-ab19-8585198343f1.png)
