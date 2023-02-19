---
title: 시간 윈도우 롤링 - 누적값 계산
date: 2023-01-11

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## 누적값 계산
- 누적값을 계산하는 쿼리이다.
- 연도별로 누적 매출 합계를 구한다.
```sql
-- JOIN 이용
SELECT
	A.sales_month,
	A.sales,
	SUM(B.sales) AS sales_ytd
FROM retail_sales AS A JOIN retail_sales AS B
	ON DATE_PART('year', A.sales_month) = DATE_PART('year', B.sales_month)
		AND B.sales_month <= A.sales_month
		AND B.kind_of_business = 'Women''s clothing stores'
WHERE A.kind_of_business = 'Women''s clothing stores'
GROUP BY A.sales_month, A.sales
```

<br>

- 위 쿼리보다 더 간단하게 같은 결과를 도출해낼 수 있다.
```sql
SELECT
	sales_month,
	sales,
	SUM(sales) OVER(PARTITION BY DATE_PART('year', sales_month) ORDER BY sales_month) AS sales_ytd
FROM retail_sales
WHERE kind_of_business = 'Women''s clothing stores'
```
![](https://velog.velcdn.com/images/ddoddo/post/c32750be-898e-4e85-bcf9-9d450601dea4/image.png)

