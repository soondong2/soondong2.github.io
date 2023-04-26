---
title: "[PostgreSQL] 날짜 데이터 함수(DATE_PART, DATE_TRUNC)"
date: 2023-04-26

categories:
  - DBMS
  - PostgreSQL
tags:
  - Database
  - PostgreSQL
---

## DATE_PART()
- `field` : year, month, day 와 같은 **날짜/시간 형태의 문자열**
- `soucre` : 실제 날짜/시간 값이다.

```sql
-- DATE_PART('field', source)
SELECT DATE_PART('year', '2023-04-26'); -> 2023
SELECT DATE_PART('month', '2023-04-26'); -> 4
SELECT DATE_PART('day', '2023-04-26'); -> 26
```

![image](https://user-images.githubusercontent.com/100760303/234484079-60cfa7df-f4a4-4310-b2f9-a32c3d15f0f7.png)

## DATE_TRUNC()
```sql
-- DATE_TRUNC(text, timestamp)
SELECT DATE_TRUNC('month', '2019-04-01 12:12:12::timestamp') -> 2019-04-01 00:00:00
SELECT DATE_TRUNC('month', '2019-04-05 12:12:12::timestamp') -> 2019-04-01 00:00:00
SELECT DATE_TRUNC('year', '2019-04-01 12:12:12::timestamp') -> 2019-01-01 00:00:00
```
