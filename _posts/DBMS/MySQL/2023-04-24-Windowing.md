---
title: "[MySQL] WINDOWING절"
date: 2023-04-24

categories:
  - DBMS
  - MySQL
tags:
  - Database
  - MySQL
  - PostgreSQL
---

## Window Function
```sql
SELECT
  WINDOW_FUNCTION(ARGUMENTS)
  OVER(
    [PARTITION BY 컬럼]
    [ORDER BY 절]
    [WINDOWING 절]
  )
 FROM 테이블명;
```

![image](https://user-images.githubusercontent.com/100760303/234008308-835d297c-7de0-466e-a39f-2f02a1fab246.png)

## WINDOWING절
- WINDOWING절은 함수의 대상이 되는 범위를 지정해주는 역할을 한다.
- 범위를 지정하는 데에는 `ROWS`와 `RANGE` 두 가지 방식이 있다.

### ROWS vs RANGE
- `ROWS` : 행의 수를 선택할 때 사용
- `RANGE` : 값의 범위를 선택할 때 사용

![image](https://user-images.githubusercontent.com/100760303/234011344-b54dad54-8e69-4665-a2d4-97e845a3ee4e.png)

- `ROWS UNBOUNDED PRECEDING`, `ROWS UNBOUNDED FOLLOWING` 과 같이 시작 범위나 종료 범위만 지정할 수 있다.
- `ROWS BETWEEN 1 PRECEDING AND UNBOUNDED FOLLOWING`, `RANGE BETWEEN 300 PRECEDING AND 300 FOLLOWING` 과 같이 시작 범위와 종료 범위를 함께 지정할 수 있다.
- BETWEEN을 쓰지 않으면 자동으로 `CURRENT ROW`를 기준으로 계산한다.

### 예시

```sql
-- 맨 위 행에서부터 현재 행까지만 계산
SELECT
  JOB,
  SUM(SALARY) OVER(PARTITION JOB ORDER BY SALARY DESC ROWS UNBOUNDED PRECEDING) 
```

```sql
-- 현재 행부터 맨 마지막 행까지 계산
SELECT
  JOB,
  SUM(SALARY) OVER(PARTITION JOB ORDER BY SALARY DESC ROWS UNBOUNDED FOLLOWING) 
```

```sql
-- 현재 행부터 다음 행까지 계산 (ROWS 1 FOLLOWING과 같다)
SELECT
  JOB,
  SUM(SALARY) OVER(PARTITION JOB ORDER BY SALARY DESC ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING) 
```

- RANGE는 행이 가지고 있는 값을 이용해 지정을 해준다.

```sql
-- 현재 행의 SALARY를 기준으로 -100 ~ +300 범위 내에 있는 행의 합을 계산
SELECT
  JOB,
  SUM(SALARY) OVER(PARTITION JOB ORDER BY SALARY DESC RANGE BETWEEN 100 PRECEDING AND 300 FOLLOWING) 
```
