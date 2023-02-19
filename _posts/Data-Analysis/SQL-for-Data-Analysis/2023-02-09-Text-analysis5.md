---
title: IN, NOT IN
date: 2023-02-09

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## Red, Orange, Yellow, Green, Blue, Purple, White
- UFO를 목격한 설명이 담긴 description 컬럼에서 각 문장의 첫 단어가 색상으로 시작하는 값들만을 조회한다.
```sql
SELECT *
FROM (
	SELECT
		SPLIT_PART(description, ' ', 1) AS first_word,
		description
	FROM ufo
) AS A
WHERE first_word = 'Red'
	OR first_word = 'Orange'
	OR first_word = 'Yellow'
	OR first_word = 'Green'
	OR first_word = 'Blue'
	OR first_word = 'Purple'
	OR first_word = 'White';
```
![](https://velog.velcdn.com/images/ddoddo/post/e47c41ba-9cb6-44bd-a058-f086cc563260/image.png)

<br>

## IN 사용
```sql
SELECT *
FROM (
	SELECT
		SPLIT_PART(description, ' ', 1) AS first_word,
		description
	FROM ufo
) AS A
WHERE first_word IN ('Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Purple', 'White');
```
![](https://velog.velcdn.com/images/ddoddo/post/6509b1dc-ad90-4179-9d85-c934af86f003/image.png)

<br>

## Color, Shape, Motion, Other
- 문장의 첫 글자가 어느 종류에 속하는지를 구별한 후 개수를 센다.
```sql
SELECT
	(CASE
	 	WHEN LOWER(first_word) IN ('red', 'orange', 'yellow', 'green', 'blue', 'purple', 'white') THEN 'Color'
	 	WHEN LOWER(first_word) IN ('round','circular','oval','cigar') THEN 'Shape'
	 	WHEN first_word ILIKE 'triang%' THEN 'Shape'
	 	WHEN first_word ILIKE 'flash%' THEN 'Motion'
	 	WHEN first_word ILIKE 'hover%' THEN 'Motion'
	 	WHEN first_word ILIKE 'pulsat%' THEN 'Motion'
	 	ELSE 'Other'
	END) AS first_word_type,
	COUNT(*)
FROM (
	SELECT
		SPLIT_PART(description, ' ', 1) AS first_word,
		description
	FROM ufo
) AS A
GROUP BY 1
ORDER BY 2 DESC;
```
![](https://velog.velcdn.com/images/ddoddo/post/a05d2918-6766-43d3-be22-8e3fb8ae1b38/image.png)

