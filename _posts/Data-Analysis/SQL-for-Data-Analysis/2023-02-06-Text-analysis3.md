---
title: 텍스트 변환
date: 2023-02-06

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## Shape
- Shape만을 추출해낸다.
- 문자의 첫 글자를 대문자로 변환한다.
```sql
SELECT DISTINCT shape, INITCAP(shape) AS shape_clean
FROM (
	SELECT SPLIT_PART(SPLIT_PART(sighting_report, 'Duration', 1), 'Shape: ', 2) AS shape
	FROM ufo
) AS A;
```
![](https://velog.velcdn.com/images/ddoddo/post/64230feb-6e23-48a3-8ae5-49c85950909b/image.png)

<br>

## Duration
- Duration을 추출한다.
- 앞 뒤 공백을 제거한다.
```sql
SELECT duration, TRIM(duration) AS duration_clean
FROM (
	SELECT SPLIT_PART(sighting_report, 'Duration:', 2) AS duration
	FROM ufo
) AS A;
```
![](https://velog.velcdn.com/images/ddoddo/post/46f0b4b3-bf93-4214-9b7f-e67a44b89154/image.png)

<br>

## Occurred, Reported, Posted
```sql
SELECT
	SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, '(Entered', 1), 'Occurred : ', 2), 'Reported', 1) AS occurred,
	SPLIT_PART(SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, 'Post', 1), 'Reported: ', 2), ' AM', 1), ' PM', 1) AS reported,
	SPLIT_PART(SPLIT_PART(sighting_report, 'Location', 1), 'Posted: ', 2) AS posted
FROM ufo;
```
![](https://velog.velcdn.com/images/ddoddo/post/8ecb7c82-916b-4035-b7f4-b08f771ec5a2/image.png)

<br>

## 데이터 타입 변환
- 위 쿼리를 서브쿼리로 사용한다.
- 각 컬럼의 문자열 길이가 8개 이상인 것을 조건으로 사용한다.
- occurred를 오름차순으로 정렬한다.

<br>

[날짜 타입 변환]
- SET DATESTYLE = mdy;를 통해 현재 데이터 타입을 명시한다.
- ::TIMESTAMP 또는 ::DATE를 통해 변환할 데이터 타입을 설정한다.
```sql
-- 타입 변환 에러가 발생하지 않도록, 현재 저장된 문자열이 월/일/연도(month/day/year)의 포맷으로 저장되어 있음을 명시 
SET DATESTYLE = mdy;
SELECT
	occurred::timestamp,
	reported::timestamp,
	posted::date
FROM (
	SELECT
		SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, '(Entered', 1), 'Occurred : ', 2), 'Reported', 1) AS occurred,
		SPLIT_PART(SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, 'Post', 1), 'Reported: ', 2), ' AM', 1), ' PM', 1) AS reported,
		SPLIT_PART(SPLIT_PART(sighting_report, 'Location', 1), 'Posted: ', 2) AS posted
	FROM ufo
) AS A
WHERE LENGTH(occurred) >= 8
	AND LENGTH(reported) >= 8
	AND LENGTH(posted) >= 8
ORDER BY 1;
```
![](https://velog.velcdn.com/images/ddoddo/post/dc1e174b-b218-408f-84d9-d205596cfeab/image.png)

<br>

## CASE 구문 사용하여 데이터 타입 변환
- 위 쿼리를 CASE 구문을 사용하여 같은 결과를 나타낼 수 있다.
- occurred 컬럼에는 NULL이 존재하므로 조건을 추가해준다.
- 나머지는 동일하게 문자열의 길이가 8보다 작으면 NULL로 표시한다.
- 8자리를 기준으로 설정한 이유는 `6/2/2023` 일 경우 LENGTH가 8이기 때문에 저 형식을 기준으로 한 것 같다.
```sql
-- 타입 변환 에러가 발생하지 않도록, 현재 저장된 문자열이 월/일/연도(month/day/year)의 포맷으로 저장되어 있음을 명시 
SELECT
	(CASE
		WHEN occurred = '' THEN NULL
		WHEN LENGTH(occurred) < 8 THEN NULL
	 	ELSE occurred::TIMESTAMP
	END) AS occurred,
	(CASE
	 	WHEN LENGTH(reported) < 8 THEN NULL
	 	ELSE reported::TIMESTAMP
	END) AS reported,
	(CASE
	 	WHEN LENGTH(posted) < 8 THEN NULL
	 	ELSE posted::TIMESTAMP
	END) AS posted
FROM (
	SELECT
		SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, '(Entered', 1), 'Occurred : ', 2), 'Reported', 1) AS occurred,
		SPLIT_PART(SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, 'Post', 1), 'Reported: ', 2), ' AM', 1), ' PM', 1) AS reported,
		SPLIT_PART(SPLIT_PART(sighting_report, 'Location', 1), 'Posted: ', 2) AS posted
	FROM ufo
) AS A
ORDER BY 1;
```
![](https://velog.velcdn.com/images/ddoddo/post/435647b6-552e-4a4b-8afc-7fc030ceddae/image.png)

<br>

## Location
```sql
SELECT
	location,
	REPLACE(REPLACE(location, 'close to', 'near'), 'outside of', 'near') AS location_clean
FROM (
	SELECT SPLIT_PART(SPLIT_PART(sighting_report, 'Shape', 1), 'Location: ', 2) AS location
	FROM ufo
) AS A
ORDER BY 1;
```
![](https://velog.velcdn.com/images/ddoddo/post/beaa2fa0-609d-40b9-b49a-fd58d0133561/image.png)

<br>

## Occurred, Entered, Reported, Posted, Shape, Duration 텍스트 변환
- SPLIT_PART() 함수를 통해 각각의 컬럼에 해당하는 텍스트를 추출한다.
- 위 내용을 서브쿼리로 사용한다.
- 문자열의 길이가 8개 미만일 경우 NULL, 아닐 경우 날짜 데이터 변환한다.
- location은 close to와 outside of 문자를 near로 변경한다.
- shape는 첫 문자를 대문자로 변환한다.
- duration은 양쪽의 공백을 제거한다.
```sql
SET datestyle = 'mdy';
SELECT
	(CASE
	 	WHEN occurred = '' THEN NULL
	 	WHEN LENGTH(occurred) < 8 THEN NULL
	 ELSE occurred::TIMESTAMP
	END) AS occurred,
	entered_as,
	(CASE
	 	WHEN LENGTH(reported) < 8 THEN NULL
	 	ELSE reported::TIMESTAMP
	END) AS reported,
	(CASE
	 	WHEN posted = '' THEN NULL
	 	ELSE posted::DATE
	END) AS posted,
	REPLACE(REPLACE(location, 'close to', 'near'), 'outside of', 'near') AS location,
	INITCAP(shape) AS shape,
	TRIM(duration) AS duration
FROM (
SELECT
	SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, '(Entered', 1), 'Occurred : ', 2), 'Reported', 1) AS occurred,
	SPLIT_PART(SPLIT_PART(sighting_report, ')', 1), 'Entered as : ', 2) AS entered_as,
	SPLIT_PART(SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, 'Post', 1), 'Reported: ', 2), ' AM', 1), ' PM', 1) AS reported,
	SPLIT_PART(SPLIT_PART(sighting_report, 'Location', 1), 'Posted: ', 2) AS posted,
	SPLIT_PART(SPLIT_PART(sighting_report, 'Shape', 1), 'Location: ', 2) AS location,
	SPLIT_PART(SPLIT_PART(sighting_report, 'Duration', 1), 'Shape: ', 2) AS shape,
	SPLIT_PART(sighting_report, 'Duration:', 2) AS duration
FROM ufo
) AS A
ORDER BY 1;
```
![](https://velog.velcdn.com/images/ddoddo/post/a6708390-5a43-4487-8968-1130351083dd/image.png)
