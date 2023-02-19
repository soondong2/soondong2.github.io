---
title: 트랜드 분석 - 요소 비교
date: 2023-01-04

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

연도와 비즈니스 종류에 따라 그루핑한 후 서점, 굿즈, 취미 및 게임 상점에 해당하는 값들의 연도, 종류, 매출의 합계를 살펴본다.
```sql
SELECT
	DATE_PART('year', sales_month) AS sales_year,
	kind_of_business,
	SUM(sales) AS sales
FROM retail_sales
WHERE kind_of_business IN ('Book stores', 'Sporting goods stores', 'Hobby, toy, and game stores')
GROUP BY DATE_PART('year', sales_month), kind_of_business
ORDER BY DATE_PART('year', sales_month) ASC, kind_of_business ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/e0db549f-4a64-450f-a14d-ea004580244b/image.png)


<br>

남성과 여성의 옷가게의 매출을 살펴본다.
```sql
SELECT sales_month, kind_of_business, sales
FROM retail_sales
WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
ORDER BY sales_month ASC, kind_of_business ASC;
```

연도별, 남성과 여성 옷 가게별로 매출 합계를 살펴본다.
```sql
SELECT 
	DATE_PART('year', sales_month) AS sales_year,
	kind_of_business,
	SUM(sales) AS sales
FROM retail_sales
WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
GROUP BY DATE_PART('year', sales_month), kind_of_business
ORDER BY DATE_PART('year', sales_month) ASC, kind_of_business ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/9a8008a4-6e25-4c79-94b4-5c8c8993029c/image.png)


<br>


연도별로 여성 옷가게일 경우의 매출합계와 남성 옷가게일 경우의 매출 합계를 계산한다. 아래의 쿼리에서 SELECT절에 SUM(sales)를 추가해줄 경우 여성과 남성의 매출을 합한 값이 나타난다.
```sql
SELECT
	DATE_PART('year', sales_month) AS sales_year,
	SUM(CASE WHEN kind_of_business='Women''s clothing stores' THEN sales END) AS women,
	SUM(CASE WHEN kind_of_business='Men''s clothing stores' THEN sales END) AS Men
FROM retail_sales
WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
GROUP BY DATE_PART('year', sales_month)
ORDER BY DATE_PART('year', sales_month) ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/50b2cf5d-77a7-441c-afa1-04ce56086ea2/image.png)


연도별 **여성 - 남성** 의류 매출 차이와 **남성 - 여성** 의류 매출 차이에 대해 살펴본다.
```sql
SELECT
	sales_year,
	(women - men) AS women_minus_men,
	(men - women) AS men_minus_men
FROM
	(SELECT
		DATE_PART('year', sales_month) AS sales_year,
		SUM(CASE WHEN kind_of_business='Women''s clothing stores' THEN sales END) AS women,
		SUM(CASE WHEN kind_of_business='Men''s clothing stores' THEN sales END) AS men
	FROM retail_sales
	WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
		AND sales_month <= '2019-12-01'
	GROUP BY DATE_PART('year', sales_month)
	) AS sub
ORDER BY sales_year ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/1c0c535c-147e-4d8d-a50e-7a05400e4f90/image.png)

위 결과에서 여성 의류 매출의 합계가 더 높은 걸 확인할 수 있다.
따라서 연도별로 여성 의류 매출에서 남성 의류 매출을 뺀 값을 보여주는 쿼리를 작성한다.
```sql
SELECT
	DATE_PART('year', sales_month) AS sales_year,
	SUM(CASE WHEN kind_of_business='Women''s clothing stores' THEN sales END)
	-
	SUM(CASE WHEN kind_of_business='Men''s clothing stores' THEN sales END) AS women_minus_men
FROM retail_sales
WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
	AND sales_month <= '2019-12-01'
GROUP BY DATE_PART('year', sales_month)
ORDER BY DATE_PART('year', sales_month) ASC
```
![](https://velog.velcdn.com/images/ddoddo/post/ce787c1f-17d7-4393-90ee-4f45ab9a595e/image.png)
