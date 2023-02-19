---
title: 정규 표현식을 활용한 패턴 매칭과 대체
date: 2023-02-11

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## 정규 표현식을 활용한 패턴 매칭과 대체
PostgreSQL은 정규식 일치의 경우 `~`를 사용하며 정규식 형식은 `POSIX` 정규식 표준을 따른다. `~`는 **similar to**를 의미한다.

<br>

## 정규표현식
- 0~9 숫자 중 하나 이상의 숫자를 포함한 경우
- `light`, `light`, `lights`, `light,` `lights,` 문자가 포함된 경우
```sql
SELECT LEFT(description, 50)
FROM ufo
WHERE LEFT(description, 50) ~ '[0-9]+ light[s ,.]'
```
![](https://velog.velcdn.com/images/ddoddo/post/61c763a9-6ad1-445c-9ce3-a7c82d3bf2d4/image.png)

<br>

## REGEXP_MATCHES()
`REGEXP_MATCHES()` 함수는 정규 표현을 문자열과 대조하여 일치하는 하위 문자열을 반환한다.
```sql
SELECT (REGEXP_MATCHES(description, '[0-9]+ light[s ,.]'))[1], COUNT(*)
FROM ufo
WHERE description ~ '[0-9]+ light[s ,.]'
GROUP BY 1
ORDER BY 2 DESC;
```
![](https://velog.velcdn.com/images/ddoddo/post/33e58e7b-144c-4500-961b-fe7a5e8c33cb/image.png)

<br>

## REGEXP_MATCHES() 예시
REGEXP_MATCHES() 함수를 정확하게 이해하기 위해 다음과 같은 예시를 활용하겠다.

|x|
|:---:|
|100|
|2|
|a한글12|
|999test|
|a11b22c33|

<br>

- 예시 1번
```sql
-- 배열을 반환한다.
SELECT
	x,
    REGEXP_MATCHES(x, '[0-9]+')
FROM test
WHERE x ~ '[0-9]+';
```
|x|regexp_matches|
|:---:|:---:|
|100|{100}|
|2|{2}|
|a한글12|{12}|
|999test|{999}|
|a11b22c33|{11}|

<br>

- 예시 2번
```sql
-- 배열을 반환한다.
SELECT
	x,
    (REGEXP_MATCHES(x, '[0-9]+'))[1]
FROM test
WHERE x ~ '[0-9]+';
```
|x|regexp_matches|
|:---:|:---:|
|100|100|
|2|2|
|a한글12|12|
|999test|999|
|a11b22c33|11|
