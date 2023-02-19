---
title: 시간 윈도우 롤링 - 롤링 계산
date: 2023-01-09

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## 쿼리 뜯어보기
해당 쿼리 또한 헷갈려서 과정을 하나씩 이해해보려고 했다.
<br>

- retail_sales 테이블을 `SELF JOIN`한다.
- JOIN 조건은 kind_of_business가 같아야 한다. (다른 컬럼들의 값은 일치하지 않는다는 것에 주의)
- `B.sales_month BETWEEN A.sales_month - interval '11 month' AND A.sales_month`
B.sales_month가 A.sales_month - 11개월에서 A.sales_month까지의 날짜를 의미한다.
- 이후 조건들은 간단하므로 설명을 생략한다.

<br>

`A.sales_month - interval '11 month'`은 A.sales_month가 `2019-12-01`이라면 `2019-01-01`에서부터 `2019-12-01`까지를 의미한다. ON 외에도 `BETWEEN A AND B`을 사용하여 JOIN 할 수도 있다.
```sql
SELECT
	A.sales_month,
	A.sales,
	B.sales_month AS rolling_sales_month,
	B.sales AS rolling_sales
FROM retail_sales AS A JOIN retail_sales AS B
	ON A.kind_of_business = B.kind_of_business
		AND B.sales_month BETWEEN A.sales_month - INTERVAL '11 month' AND A.sales_month
		AND B.kind_of_business = 'Women''s clothing stores'
WHERE A.kind_of_business = 'Women''s clothing stores'
	AND A.sales_month = '2019-12-01';
```
![](https://velog.velcdn.com/images/ddoddo/post/dee74bcc-d042-41bd-b81f-608b8f22765c/image.png)

<br>

```sql
SELECT
	A.sales_month,
	A.sales,
	ROUND(AVG(B.sales), 2) AS moving_avg,
	COUNT(B.sales) AS records_count
FROM retail_sales AS A JOIN retail_sales AS B
	ON A.kind_of_business = B.kind_of_business
		AND B.sales_month BETWEEN A.sales_month - INTERVAL '11 month' AND A.sales_month
		AND B.kind_of_business = 'Women''s clothing stores'
WHERE A.kind_of_business = 'Women''s clothing stores'
	AND A.sales_month >= '1993-01-01'
GROUP BY A.sales_month, A.sales
ORDER BY A.sales_month ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/5de12f6c-575c-4f7d-b667-dfd96c68ef39/image.png)

<br>

위에서 복잡하게 계산한 과정들을 간단하게 구해볼 수 있는 쿼리이다. 
- `AVG(sales) OVER(ORDER BY sales_month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)`
현재 행을 포함하여 이전 11개까지의 행의 **합계**를 구한다.
- `COUNT(sales) OVER(ORDER BY sales_month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)`
현재 행을 포함하여 이전 11개까지의 행의 **개수**를 구한다.
```sql
SELECT
	sales_month,
	ROUND(AVG(sales) OVER(ORDER BY sales_month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW), 2) AS moving_avg,
	COUNT(sales) OVER(ORDER BY sales_month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS records_count
FROM retail_sales
WHERE kind_of_business = 'Women''s clothing stores';
```
![](https://velog.velcdn.com/images/ddoddo/post/9587b28f-59a8-4edb-bdf5-ac2ac3ba19fd/image.png)

<br>

1992-01-01부터 1992-11-01까지는 records_count가 12가 될 수 없으며, 1992-12-01부터 이후 모든 날짜의 records_count가 12로 동일하다.<br>
retail_sales 테이블에는 1992-01-01부터 2020-12-01까지의 데이터만 담겨있기 때문이다. 

```sql
SELECT MIN(sales_month), MAX(sales_month)
FROM retail_sales;
```
![](https://velog.velcdn.com/images/ddoddo/post/03fbf427-2e6e-4938-b1ce-0eb663b3c9d4/image.png)

<br>

---
## PostgreSQL vs MySQL
```sql
SELECT *
FROM retail_sales AS A JOIN retail_sales AS B
	ON A.kind_of_business = B.kind_of_business
		AND B.sales_month BETWEEN A.sales_month - INTERVAL '11 month' AND A.sales_month
```
위에서 진행했던 쿼리 중 JOIN 조건에 `BETWEEN A AND B`를 사용하여 조인하던 구문이 있었다.<br>
그 중 **BETWEEN A.sales_month - INTERVAL '11 month' AND A.sales_month** 부분에서 `11 month`에 주의한다.<br>

- PostgreSQL에서는 `따옴표(')`를 사용해야만 오류가 발생하지 않고 쿼리가 실행된다. 혹은 숫자만 따옴표로 감싸주어야 한다.
```sql
-- 오류 발생
11 month

-- 오류 발생하지 않음
'11 month'
'11' month
```

- MySQL에서는 `따옴표(')`를 사용하지 않아야만 오류가 발생하지 않는다. 혹은 숫자만 따옴표로 감싸주어야 한다.
```sql
-- 오류 발생
'11 month' 

-- 오류 발생하지 않음
11 month
'11' month
```

<br>

또한 마지막 쿼리 부분에서도 차이가 있었다.
```sql
SELECT
	sales_month,
	ROUND(AVG(sales) OVER(ORDER BY sales_month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW), 2) AS moving_avg,
	COUNT(sales) OVER(ORDER BY sales_month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS records_count
FROM retail_sales
WHERE kind_of_business = 'Women''s clothing stores';
```
**COUNT(sales) OVER(ORDER BY sales_month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)** 부분에서 숫자를 사용할 때 차이가 있다.

- PostgreSQL
```sql
-- 오류 발생하지 않음
ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) 
ROWS BETWEEN '11' PRECEDING AND CURRENT ROW)
```
- MySQL
```sql
-- 오류 발생
ROWS BETWEEN '11' PRECEDING AND CURRENT ROW)

-- 오류 발생하지 않음
ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)
```
