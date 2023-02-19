---
title: 트렌드 분석 - 전체 대비 비율
date: 2023-01-05

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## 전체 대비 비율
### 날짜별 매출 비율
날짜별 매점별 매출 비율에 대해서 알아보자.

중간 과정이 살짝쿵 헷갈려져서 쿼리를 뜯어보았다. 아래는 내가 이해해가는 과정이다. 틀린 부분이 있다면 피드백을 남겨주세요!

<서브쿼리>
- 동일한 retail_sales 테이블을 `SELF JOIN` 한다.
- `JOIN` 기준은 `sales_month`(날짜)가 같은 기준으로 한다.
- `JOIN` 조건에 AND로 B 테이블의 `kind_of_business`가 Men, Women store만을 포함하도록 설정한다.

<br>

주의 ❗❗ <br>
1992-01-01 이라는 날짜가 일치하는 것과 kind_of_business가 Men과 Women을 포함하는 것을 기준으로 JOIN 하였으므로 sales_month를 제외한 컬럼들은 A 테이블과 동일한 상태로 정렬되어 있지는 않다.

- `WHERE`절에 A 테이블도 마찬가지로 `kind_of_business`가 Men, Women store만을 포함하도록 설정한다.
- sales_month, kind_of_business, sales 순으로 `GROUP BY` 한다.
- `SELECT` 절에 A 테이블의 sales_month, kind_of_business, sales와 B 테이블의 sales의 합계를 구한다. `SUM(B.sales)`는 날짜별 store의 `총 매출`을 뜻한다.

<br>

주의 ❗❗
```sql
-- 해당 쿼리는 696개의 row를 갖는다.
SELECT * 
FROM retail_sales
WHERE kind_of_business = 'Men''s clothing stores'
	OR kind_of_business = 'Women''s clothing stores'
```
위 쿼리는 **696개**의 row를 갖지만, JOIN 이후 2배인 **1392개**의 row를 갖게 된다. 따라서 sales까지 GROUP BY를 해준 후 `SUM(B.sales)` 해주어야 중복이 제거된 매출의 합계를 구할 수 있다. 이는 날짜별 총 매출을 뜻한다. `SUM(A.sales)`를 해줄 경우 GROUP BY 된 A.sales에는 각각 701, 701과 1873, 1873이 있으므로 연도별 총 매출의 합이 아닌 연도별, 매점별 매출의 합계가 계산된다. 이는 잘못된 계산이다.

![](https://velog.velcdn.com/images/ddoddo/post/978a5585-d5e3-4628-8988-17fb7341bf07/image.png) 

<br>

- 위에서 얻어낸 테이블을 서브쿼리로하여 sales_month, kind_of_business, 날짜별 **전체 매출 대비 매장의 매출에 대한 비율**을 계산한다.
- `ORDER BY`를 통해 오름차순 정렬한다.

```sql
SELECT
	sales_month,
	kind_of_business,
	ROUND(sales / total_sales * 100, 2) AS pct_total_sales
FROM(
	SELECT A.sales_month, A.kind_of_business, A.sales, SUM(B.sales) AS total_sales
	FROM retail_sales AS A JOIN retail_sales AS B
		ON A.sales_month = B.sales_month
		AND B.kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
	WHERE A.kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
	GROUP BY A.sales_month, A.kind_of_business, A.sales
	) AS sub
ORDER BY sales_month, kind_of_business;
```
![](https://velog.velcdn.com/images/ddoddo/post/2a26d86e-ff8e-4a5c-84ae-b4e3eb672695/image.png)

<br>

`SUM() OVER()` 함수를 사용하면 훨씬 간단하게 총 매출과 비율을 계산할 수 있다. 하핫... 위에 거 다 뜯어봤는데.. 긍정 긍정..
```sql
SELECT
	sales_month,
	kind_of_business,
	SUM(sales) OVER(PARTITION BY sales_month) AS total_sales,
	ROUND(sales / SUM(sales) OVER(PARTITION BY sales_month) * 100, 2) AS pct_total_sales
FROM retail_sales
WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
ORDER BY sales_month ASC, kind_of_business ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/46a794ae-369c-4c45-b408-316079206f7d/image.png)

<br>

### 연도별 매출 비율
위 쿼리에선 날짜별 여성, 남성 의류점 매출 합계와 비율을 알아봤다.
아래의 쿼리에선 연도별 여성, 남성 의류점 매출 합계와 비율을 알아본다. 쿼리가 작동되는 방식은 **연도별**이 추가된 것을 제외하고는 동일하므로 위 설명을 참고한다.

```sql
SELECT
	sales_month,
	kind_of_business,
	ROUND(sales / yearly_sales * 100, 2) AS pct_yearly
FROM (
	SELECT A.sales_month, A.kind_of_business, A.sales, SUM(B.sales) AS yearly_sales
	FROM retail_sales AS A JOIN retail_sales AS B
		ON DATE_PART('year', A.sales_month) = DATE_PART('year', B.sales_month)
			AND A.kind_of_business = B.kind_of_business
			AND B.kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
			AND B.kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
	GROUP BY A.sales_month, A.kind_of_business, A.sales
	) AS sub
ORDER BY sales_month ASC, kind_of_business ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/47c9fb08-8008-4b8a-8de4-c7c1a98ac7b5/image.png)

마찬가지로 위 쿼리를 `SUM() OVER()` 함수를 통해 간단하게 나타내면 다음과 같다.

```sql
SELECT
	sales_month,
	kind_of_business,
    sales,
	SUM(sales) OVER(PARTITION BY DATE_PART('year', sales_month), kind_of_business) AS yearly_sales,
	ROUND(sales / SUM(sales) OVER(PARTITION BY DATE_PART('year', sales_month), kind_of_business) * 100, 2) AS pct_yearly
FROM retail_sales
WHERE kind_of_business IN('Men''s clothing stores', 'Women''s clothing stores')
ORDER BY sales_month ASC, kind_of_business ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/f928004e-cbad-4eb8-a9d4-524cea5fdc0b/image.png)

