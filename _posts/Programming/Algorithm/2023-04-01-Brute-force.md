---
title: "완전 탐색(Brute Force)"
date: 2023-04-01

categories:
  - Programming
  - Algorithm
tags:
  - Algorithm  
---

## 완전 탐색
`완전탐색`은 가능한 경우의 수를 모두 조사해서 정답을 찾는 방법으로, 무식하게 가능한 것을 다 해보겠다는 의미로 `Brute Force`라고도 한다.

## 완전 탐색 종류
### 1. 브루트포스(Brute Force)
브루트포스 기법은 반복/조건문을 통해 가능한 모든 방법을 단순히 찾는 경우를 말한다.

### 2. 백트래킹(Backtracking)
백트래킹은 현재 상태에서 가능한 후보군으로 가지를 치며 탐색하는 알고리즘이다. `분할 정복`을 이용한 기법으로, `재귀함수`를 이용하고 해를 찾아가는 도중 해가 될 것 같지 않은 경로가 있다면 더 이상 가지 않고 되돌아간다.

### 3. 순열(Permutation)
순열은 임의의 수열이 주어졌을 때 그것을 다른 순서로 연산하는 방법이다.

```python
from itertools import permutations
permutations(arr, n) 
```

### 4. 재귀함수
재귀함수를 통해서 문제를 만족하는 경우들을 만들어가는 방식이다.

### 5. DFS/BFS
난이도가 있는 문제로 `완전탐색 + DFS/BFS` 문제가 나온다. 예를 들어, 단순히 길을 찾는 문제라면 DFS/BFS만 이용해도 충분하지만, 주어진 도로에 장애물을 설치하거나 목적지를 추가하는 등의 추가적인 작업이 필용한 경우에 이를 완전 탐색으로 해결하고 나서 DFS/BFS를 이용해야 한다.

## 활용 예시
### 프로그래머스 Level1 모의고사
```python
def solution(answers):
    answer = []
    one = [1, 2, 3, 4, 5]
    two = [2, 1, 2, 3, 2, 4, 2, 5]
    three = [3, 3, 1, 1, 2, 2, 4, 4, 5, 5]
    
    score = [0, 0, 0]
    for i, a in enumerate(answers):
        if one[i % len(one)] == a:
            score[0] += 1
        if two[i % len(two)] == a:
            score[1] += 1
        if three[i % len(three)] == a:
            score[2] += 1
    
    for i in range(len(score)):
        if max(score) == score[i]:
            answer.append(i + 1)
            
    return answer
```
