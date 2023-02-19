---
title: 시간 윈도우 롤링 - 희소 데이터와 시간 윈도우 롤링
date: 2023-01-11

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## 희소 데이터와 시간 윈도우 롤링
date_dim 데이터를 살펴보자.
```sql
SELECT * FROM date_dim
```
![](https://velog.velcdn.com/images/ddoddo/post/8b2f9e4e-b9f5-4542-ba4a-d944b44a669c/image.png)

<br>

date_dim 테이블의 날짜 컬럼인 date 범위는 다음과 같다.
```sql
SELECT MIN(date), MAX(date)
FROM date_dim
```
![](https://velog.velcdn.com/images/ddoddo/post/1a8fe1b4-4e6e-4fbd-9a2f-6c2615d4b18c/image.png)

<br>

retail_sales 테이블에서 Women store와 1월 7월에 해당하는 날짜와 매출을 조회한다.
```sql
SELECT sales_month, sales
FROM retail_sales
WHERE kind_of_business = 'Women''s clothing stores'
	AND DATE_PART('month', sales_month) IN (1, 7)
```
![](https://velog.velcdn.com/images/ddoddo/post/963fa4f2-b4de-49ed-9842-87184a251c26/image.png)

<br>

- date_dim 테이블과 위에서 조회한 테이블을 JOIN 한다.
- JOIN 기준은 date를 포함하여 이전 11달까지의 날짜로 한다.
- date와 first_day_of_month가 같은 경우와 date 컬럼도 sales_month의 컬럼에 맞춰 1993-01-01부터 2020-12-01까지의 날짜만 조회한다.
```sql
SELECT A.date, B.sales_month, B.sales
FROM date_dim AS A
	JOIN (SELECT sales_month, sales
		  FROM retail_sales
		  WHERE kind_of_business = 'Women''s clothing stores'
		  AND DATE_PART('month', sales_month) IN (1, 7)
) AS B ON B.sales_month BETWEEN A.date - INTERVAL '11 month' AND A.date
WHERE A.date = A.first_day_of_month
	AND A.date BETWEEN '1993-01-01' AND '2020-12-01'
ORDER BY A.date ASC, B.sales_month ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/687a5c4d-ba69-451d-89ee-6ab2d720c081/image.png)

<br>

- 위 쿼리에서 date별로 매출 평균을 계산한다.
- COUNT() 함수를 통해 record 개수를 센다.
1월과 7월만을 포함했기 때문에 앞서 했던 12개가 아닌 2개가 존재한다.
```sql
SELECT
	A.date,
	ROUND(AVG(B.sales), 2) AS moving_avg,
	COUNT(B.sales) AS records
FROM date_dim AS A
	JOIN (SELECT sales_month, sales
		  FROM retail_sales
		  WHERE kind_of_business = 'Women''s clothing stores'
		  AND DATE_PART('month', sales_month) IN (1, 7)
) AS B ON B.sales_month BETWEEN A.date - INTERVAL '11 month' AND A.date
WHERE A.date = A.first_day_of_month
	AND A.date BETWEEN '1993-01-01' AND '2020-12-01'
GROUP BY A.date
ORDER BY A.date ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/f00aa380-8263-4340-b470-41a2b833b6bd/image.png)

<br>

1993-01-01부터 2020-12-01까지의 고유한 날짜 테이블을 조회한다.
```sql
SELECT DISTINCT(sales_month) 
FROM retail_sales
WHERE sales_month BETWEEN '1993-01-01' AND '2020-12-01'
ORDER BY sales_month;
```
![](https://velog.velcdn.com/images/ddoddo/post/82d9fecb-8d53-4785-82e4-f7474bf16aa9/image.png)

<br>

- 위에서 조회한 테이블과 retail_sales 테이블을 JOIN한다.
- B 테이블의 날짜는 A 테이블의 날짜를 포함하여 이전 11개월까지의 날짜를 기준인 것과, 비즈니스 종류를 Women store인 경우만을 JOIN한다.
- A 테이블의 날짜를 그루핑하며 각 날짜별 매출 합계를 계산한다.
```sql
SELECT A.sales_month, ROUND(AVG(B.sales), 2) AS moving_avg
FROM(
	SELECT DISTINCT(sales_month) 
	FROM retail_sales
	WHERE sales_month BETWEEN '1993-01-01' AND '2020-12-01'
) AS A JOIN retail_sales AS B
	ON B.sales_month BETWEEN A.sales_month - INTERVAL '11 month' AND A.sales_month
		AND B.kind_of_business = 'Women''s clothing stores'
GROUP BY A.sales_month
ORDER BY A.sales_month ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/05035c49-f131-407a-a257-5cabfb897bbd/image.png)
