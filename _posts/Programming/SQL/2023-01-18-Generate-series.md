---
title: generate_series
date: 2023-01-18

categories:
  - Programming
  - SQL
tags:
    - SQL
---

## generate_series
사용법은 다음과 같다.
```sql
SELECT generate_series AS 별칭
FROM generate_series(start, stop, step)
```

<br>

만약 다음과 같은 쿼리를 작성했을 경우의 결과를 알아보자.
```sql
SELECT generate_series AS period
FROM generate_series(0, 10, 1)
```
|period|
|:---:|
|0|
|1|
|2|
|3|
|4|
|5|
|6|
|7|
|8|
|9|
|10|

<br>

참고로 MySQL에서는 사용 불가능한 것 같다.
