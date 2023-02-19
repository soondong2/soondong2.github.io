---
title: LIKE, ILIKE
date: 2023-02-07

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## 와일드카드 매칭 LIKE vs ILIKE
`LIKE`는 대소문자를 구분하며, `ILIKE`는 대소문자를 구분하지 않는다.

<br>

## LIKE
```sql
SELECT COUNT(*)
FROM ufo
WHERE description LIKE '%wife%';
```
![](https://velog.velcdn.com/images/ddoddo/post/fde37ad7-2e1d-4c30-8854-c299aabad404/image.png)

<br>

```sql
SELECT COUNT(*)
FROM ufo
WHERE LOWER(description) LIKE '%wife%';
```
![](https://velog.velcdn.com/images/ddoddo/post/63292785-1d96-4cdc-aaa9-6488f44838ba/image.png)

<br>

첫 번째는 6231개, 두 번째는 6439개로 다른 개수가 나타났다. 첫 번째의 경우 **Wife와 wife가 서로 다른 글자로 인식되기 때문이다.** 대소문자 상관 없이 Wife나 wife 글자가 들어간 description의 행의 개수는 총 6439개이다.

<br>

### ILIKE
```sql
SELECT COUNT(*)
FROM ufo
WHERE description ILIKE '%wife%';
```
```sql
SELECT COUNT(*)
FROM ufo
WHERE LOWER(description) ILIKE '%wife%';
```
![](https://velog.velcdn.com/images/ddoddo/post/102dda22-f70b-4a38-8a4f-a0c1212d0cc4/image.png)

<br>

ILIKE는 대소문자를 구분하지 않기 때문에 두 경우 모두 같은 개수가 나타난다.

<br>

## wife 또는 husband가 들어간 행의 수
```sql
SELECT COUNT(*)
FROM ufo
WHERE LOWER(description) LIKE '%wife%'
	OR LOWER(description) LIKE '%husband%';
```
![](https://velog.velcdn.com/images/ddoddo/post/95b1c689-ec55-434b-99f5-91e02e092c7d/image.png)

<br>

## AND, OR 우선순위
```sql
SELECT COUNT(*)
FROM ufo
WHERE LOWER(description) LIKE '%wife%'
	OR LOWER(description) LIKE '%husband%'
	AND LOWER(description) LIKE '%mother%';
```
![](https://velog.velcdn.com/images/ddoddo/post/f1df60c8-2479-4fcf-98b7-06fd9633d795/image.png)

<br>

```sql
SELECT description
FROM ufo
WHERE (LOWER(description) LIKE '%wife%'
	OR LOWER(description) LIKE '%husband%')
	AND LOWER(description) LIKE '%mother%';
```
![](https://velog.velcdn.com/images/ddoddo/post/55741862-279b-47e8-8530-25ac10b517b7/image.png)

<br>
AND가 OR보다 우선순위가 높다. -> ** AND > OR** <br>
OR를 우선순위로 적용하고 싶다면 `괄호()`를 통해 우선순위를 지정해주어야 한다.

<br>

## driving, walking, running, swimming
UFO를 목격한 설명이 담긴 description 컬럼에 driving, walking, running, swimming 단어가 담겨 있는 경우의 개수를 센다.
```sql
SELECT
	(CASE
	 	WHEN LOWER(description) LIKE '%driving%' THEN 'driving'
	 	WHEN LOWER(description) LIKE '%walking%' THEN 'walking'
	 	WHEN LOWER(description) LIKE '%%running' THEN 'running'
	 	WHEN LOWER(description) LIKE '%swimming%' THEN 'swimming'
	 	ELSE 'none'
	END) AS activity,
	COUNT(*)
FROM ufo
GROUP BY 1
ORDER BY 2 DESC;
```
![](https://velog.velcdn.com/images/ddoddo/post/5d54ace2-9ff9-4957-9e0b-14e595d2fc7e/image.png)

<br>

## SELECT절에 와일드카드 매칭 사용
- 보통 LIKE나 ILIKE를 사용할 때 `WHERE` 절에 사용하는 경우가 많다.
- 그러나 `SELECT` 절에 사용할 경우 `True or False`의 값을 반환해준다.
```sql
SELECT
	description ILIKE '%south%' AS south,
	description ILIKE '%north%' AS north,
	description ILIKE '%east%' AS east,
	description ILIKE '%west%' AS west,
	COUNT(*)
FROM ufo
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2, 3, 4;
```
![](https://velog.velcdn.com/images/ddoddo/post/45fe7e7f-5d9d-478e-9d75-6c26c432c677/image.png)

<br>

## south, north, east, west
UFO를 목격한 설명이 담긴 description 컬럼에 south, north, east, west 단어가 들어가는 경우의 개수를 센다.
```sql
SELECT
	COUNT(CASE WHEN description ILIKE '%south%' THEN 1 END) AS south,
	COUNT(CASE WHEN description ILIKE '%north%' THEN 1 END) AS north,
	COUNT(CASE WHEN description ILIKE '%east%' THEN 1 END) AS east,
	COUNT(CASE WHEN description ILIKE '%west%' THEN 1 END) AS west
FROM ufo;
```
![](https://velog.velcdn.com/images/ddoddo/post/87c0708c-4718-46cf-8756-eacd82d92a94/image.png)
