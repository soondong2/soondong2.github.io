---
title: "해시 테이블(Hash table)"
date: 2023-03-06

categories:
  - Programming
  - Algorithm
tags:
  - Algorithm  
---

## 해시 테이블(Hash Table)
- 데이터를 관리하고 유지하는 자료구조이다.
- 리소스를 포기하고 속도를 취한 자료구조이다.
- `O(1)`의 시간 복잡도를 가진다. → 절대적이진 않지만 대부분 그렇다.
- `Key`와 `Value`를 활용한다.
- `해시 함수(Hash Function)`를 사용하여 Key를 Index 숫자로 변환 해준다.

<br>

| Key |
| --- |
| Key1 |
| Key2 |

<br>

| Index | Value |
| --- | --- |
| 0 | Value1 |
| 1 | Value2 |
    
## 충돌 대처법
### 체이닝
해당 Index에 Value가 있으면 체인으로 연결한다(List)
        
| Index | Value |
| --- | --- |
| 0 | Value1 → Value 2 → Value3 |

### 선형탐색
먼저 만들어 놓은 버켓을 먼저 소모한다.
해시 테이블이 꽉 차면 테이블 리사이징이 필요하다.
        
    
[Hash Table vs Array]
배열(Array)는 하나하나 선형탐색하므로 O(n)의 시간 복잡도를 가진다.
    
#### Array
```python
menu = [
  {name: : 'coffee', price : 0},
  {name : 'burger', price : 15},
    	...
  {name : 'pizza', price : 10}
    ]
```
    
#### Hash Table
    
```python
menu = {
  coffee : 10,
  burger : 15,
  pizza : 10
 }
```
    
## 활용 예시
- () 안에 변수명, 숫자를 입력할 수 있다.
- print()를 하면 해시값이 출력된다.
    
```python
    h = hash()
    print(h)
```
    
```python
    # 해시 테이블 정의
    hash_map = {}
    
    # key : value
    hash_map[key] = value
    
    # 리스트로 value를 넣어줄 경우
    # hash_map = {key : [value1, value2]}
    hash_map.append([value1, value2])
    hash_map[key] = [[value1, value2]]
    
    # key : value로 이루어진 dictionary 정렬 방법
    sorted(hash_map.itemps(), key=lambda x:x[0], reverse=True) # key로 내림차순 정렬
    sorted(hash_map.itemps(), key=lambda x:x[1], reverse=False) # value로 오름차순 정렬
```

### 프로그래머스 Level3 베스트 앨범
```python
def solution(genres, plays):
    answer = []
    hash_map1 = {}
    # {장르1 : [재생 횟수, index], 장르2 : [재생 횟수, index]} 해시 테이블 만들기
    for i, g in enumerate(genres):
        if g in hash_map1:
            hash_map1[g].append([plays[i], i])
        else:
            hash_map1[g] = [[plays[i], i]]
    
    # 장르별 총 재생 횟수 계산
    hash_map2 = {}
    for k in hash_map1.keys():
        total = 0
        for i in range(len(hash_map1[k])):
            total += hash_map1[k][i][0]
        hash_map2[k] = total

    # 장르별 총 재생 횟수가 큰 순으로 정렬
    genre_rank = sorted(hash_map2.items(), key=lambda x:x[1], reverse=True)
    print(genre_rank)
    # 장르 내에서 각각의 재생 횟수가 큰 순으로 정렬
    for g in genre_rank:
        sorted_rank = sorted(hash_map1[g[0]], key=lambda x:x[0], reverse=True)[:2]
        # 고유 번호
        for i in sorted_rank:
            answer.append(i[1])
    return answer
```

### 프로그래머스 Level2 전화번호 목록
```python
def solution(phone_book):
    # 1. Hash map을 만든다
    hash_map = {}
    for phone_number in phone_book:
        hash_map[phone_number] = 1
    
    # 2. 접두어가 Hash map에 존재하는지 찾는다
    for phone_number in phone_book:
        jubdoo = ""
        for number in phone_number:
            jubdoo += number
            # 3. 접두어를 찾아야 한다 (기존 번호와 같은 경우 제외)
            if jubdoo in hash_map and jubdoo != phone_number:
                return False
    return True
```
