---
title: 텍스트 파싱
date: 2023-02-01

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## Occurred
```sql
SELECT LEFT(sighting_report, 8) AS left_digits, COUNT(*)
FROM ufo
GROUP BY 1;
```
![](https://velog.velcdn.com/images/ddoddo/post/4013444c-a31d-47ba-84b4-6fa1086e4da2/image.png)

<br>

```sql
SELECT RIGHT(LEFT(sighting_report, 25), 14) AS occurred
FROM ufo
```
![](https://velog.velcdn.com/images/ddoddo/post/dc60f64e-fc80-45b5-a290-5295dc14eb06/image.png)
 
 <br>
 
## split_part
```sql
SELECT SPLIT_PART(sighting_report, 'Occurred : ', 2) AS split_1
FROM ufo;
```
![](https://velog.velcdn.com/images/ddoddo/post/a5cefd7f-2a1b-4f52-b37f-21a3435baf24/image.png)

<br>

```sql
SELECT SPLIT_PART(sighting_report, ' (Entered', 1) AS split_2
FROM ufo;
```
![](https://velog.velcdn.com/images/ddoddo/post/3b61e795-0c14-4200-b314-4be859ca10a3/image.png)

<br>

` (Entered`를 기준으로 자른 첫 번째 문자열을 또 다시 `Occurred : `를 기준으로 잘라서 두 번째 문자열을 가져온 것이다.
```sql
SELECT SPLIT_PART(SPLIT_PART(sighting_report, ' (Entered', 1), 'Occurred : ', 2) AS occurred
FROM ufo;
```
![](https://velog.velcdn.com/images/ddoddo/post/502fdcd2-83a3-49d2-860d-768d71cf039b/image.png)

<br>

아래 사진처럼 다른 형식과는 다르게 Reported가 붙은 긴 문자열이 보인다. <br>

![](https://velog.velcdn.com/images/ddoddo/post/423b21f4-5b0f-428f-a61f-b2d1678667aa/image.png)

<br>

14번째 행의 문자열도 다른 형식과 마찬가지로 처리 되었다.
```sql
SELECT SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, ' (Entered', 1), 'Occurred : ', 2), 'Reported', 1) AS occurred
FROM ufo;
```
![](https://velog.velcdn.com/images/ddoddo/post/2728ca31-0249-470a-802b-dd5878e55659/image.png)

<br>

## occurred, entered, reported, posted, location, shape, duration
```sql
SELECT
	SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, ' (Entered', 1), 'Occurred : ', 2), 'Reported', 1) AS occurred,
	SPLIT_PART(SPLIT_PART(sighting_report, ')', 1), 'Entered as : ', 2) AS entered_as,
	SPLIT_PART(SPLIT_PART(SPLIT_PART(SPLIT_PART(sighting_report, 'Post', 1), 'Reported: ', 2), 'AM', 1), 'PM', 1) AS reported,
	SPLIT_PART(SPLIT_PART(sighting_report, 'Location', 1), 'Posted: ', 2) AS posted,
	SPLIT_PART(SPLIT_PART(sighting_report, 'Shape', 1), 'Location: ', 2) AS location,
	SPLIT_PART(SPLIT_PART(sighting_report, 'Duration', 1), 'Shape: ', 2) AS shape,
	SPLIT_PART(sighting_report, 'Duration:', 2) AS duration
FROM ufo;
```
![](https://velog.velcdn.com/images/ddoddo/post/722f3188-fe35-42af-b89f-ec90d9d7f724/image.png)

