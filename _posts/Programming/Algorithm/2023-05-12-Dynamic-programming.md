---
title: "다이나믹 프로그래밍(Dynamic Programming)"
date: 2023-05-12

categories:
  - Programming
  - Algorithm
tags:
  - Algorithm  
---

## DP란?
메모리 공간을 약간 더 사용하면 연산 속도를 비약적으로 증가시킬 수 있는 방법이 대표적으로 다이나믹 프로그래밍이다. `동적 계획법`이라고도 부른다. 정리하자면, 큰 문제를 작게 나누고, 같은 문제라면 한 번씩만 풀어 문제를 효율적으로 해결하는 알고리즘 기법이다. 또한 일반적으로 반복문을 활용한 다이나믹 프로그래밍 성능이 더 좋다.

## 기존 알고리즘으로 해결하기 어려운 문제들
### 피보나치 수열
f(n)의 n이 커지면 커질수록 수행 시간이 기하급수적으로 늘어난다. 이러한 문제를 DP로 해결할 수 있다.

## DP를 활용하기 위한 조건
다만, 항상 DP를 사용할 수는 없으며 다음 조건을 만족할 때 사용할 수 있다.

1. 큰 문제를 작은 문제로 나눌 수 있다.
2. 작은 문제에서 구한 정답은 그것을 포함하는 큰 문제에서도 동일하다.

## 메모이제이션이란?
다이나믹 프로그래밍을 구현하는 방법 중 한 종류로, 한 번 구한 결과를 메모리 공간에 메모해두고 같은 식을 다시 호출하면 메모한 결과를 그대로 가져오는 기법을 의미한다. 메모이제이션은 값을 저장하는 방법이므로 `캐싱`이라고도 한다.

## DP 구현 방법
### 탑다운(Top-Down) 방식
큰 문제를 해결하기 위해 작은 문제를 호출하는 방법을 말한다.

```python
# 피보나치 수열 (재귀적)

# 한 번 계산된 결과를 메모이제이션하기 위한 리스트 초기화
d = [0] * 100

# 피보나치 함수를 재귀함수로 구현(탑 다운 다이나믹 프로그래밍)
def fibo(x):
    # 종료 조건
    if x == 1 or x == 2:
        return 1
    # 이미 계산한 적 있는 문제라면 그대로 반환
    if d[x] != 0:
        return d[x]
    # 아직 계산하지 않은 문제라면 점화식에 따라서 피보나치 결과 반환
    d[x] = fibo(x - 1) + fibo(x - 2)
    return d[x]
```

### 보텀업(Bottom-Up) 방식
작은 문제부터 차근차근 답을 도출하는 방법을 말한다. 주로 많이 사용하는 방법이다.

```python
# 피보나치 수열 (반복문)

# 한 번 계산된 결과를 메모이제이션하기 위한 리스트 초기화
d = [0] * 100

# 첫 번째, 두 번째 피보나치 수는 1
d[1] = 1
d[2] = 1
n = 99

# 피보나치 함수를 반복문으로 구현(보텀업 다이나믹 프로그래밍)
for i in range(3, n + 1):
    d[i] = d[i - 1] + d[i - 2]

print(d[n])
```