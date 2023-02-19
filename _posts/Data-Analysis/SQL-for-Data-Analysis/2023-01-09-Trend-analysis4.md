---
title: 트렌드 분석 - 시계열 데이터 변화 이해를 위한 인덱싱
date: 2023-01-09

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

문득 궁금해서 아래 쿼리를 서브쿼리와 WITH 구문을 이용하여 실행해보니 다음과 같은 실행 시간이 소요됐다. 따라서 FROM절에 서브쿼리를 많이 사용하거나 긴 쿼리를 작성하게 된다면 소요되는 시간이나 눈으로 보기에 복잡할 수 있으므로 WITH를 사용한다.
- WITH : 00:00:00.397
- 서브쿼리 : 00:00:00.638

<br>

Women store에 해당하는 연도별 매출 합계를 계산한다.
`FIRST_VALUE` 함수를 통해 연도별로 오름차순 정렬한 후 sales의 첫 번째 값을 가져온 index_sales 컬럼을 생성한다.
```sql
SELECT
	sales_year,
	sales,
	FIRST_VALUE(sales) OVER(ORDER BY sales_year ASC) AS index_sales
FROM(
	SELECT
		DATE_PART('year', sales_month) AS sales_year,
		SUM(sales) AS sales
	FROM retail_sales
	WHERE kind_of_business = 'Women''s clothing stores'
	GROUP BY DATE_PART('year', sales_month)
) AS sub;
```
![](https://velog.velcdn.com/images/ddoddo/post/0f305cfc-4d94-46a8-95a8-cdbcf8e7aaca/image.png)

위 결과에서 1992년도 매출 합계인 31815가 index_sales의 값으로 계산되었다. 따라서 1992년도의 매출 합계를 기준으로 비율을 계산한다. 양수일 경우 기준 매출 합계보다 더 많은 매출이 나온 것이고, 음수일 경우 더 적은 매출을 의미한다.
```sql
SELECT
	sales_year,
	sales,
	ROUND((sales / FIRST_VALUE(sales) OVER(ORDER BY sales_year ASC) - 1) * 100, 2) AS pct_from_index
FROM(
	SELECT
		DATE_PART('year', sales_month) AS sales_year,
		SUM(sales) AS sales
	FROM retail_sales
	WHERE kind_of_business = 'Women''s clothing stores'
	GROUP BY DATE_PART('year', sales_month)
) AS sub;
```
![](https://velog.velcdn.com/images/ddoddo/post/b965092d-1a58-45e3-984e-df5def002061/image.png)


서브쿼리에서 연도별 매장별(남성, 여성) 2019년 12월 31일을 포함한 이전 날짜의 매출 합계를 계산한다.

위의 결과에서 연도별로 오름차순 했을 경우 첫 번째 매출 합계 값을 기준으로 비율을 계산한다.
```sql
SELECT
	sales_year,
	kind_of_business,
	sales,
	ROUND((sales / FIRST_VALUE(sales) OVER(PARTITION BY kind_of_business ORDER BY sales_year ASC) - 1) * 100, 2) AS pct_from_index
FROM(
	SELECT
		DATE_PART('year', sales_month) AS sales_year,
		kind_of_business,
		SUM(sales) AS sales
	FROM retail_sales
	WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
		AND sales_month <= '2019-12-31'
	GROUP BY DATE_PART('year', sales_month), kind_of_business
) AS sub
ORDER BY sales_year ASC, kind_of_business ASC;
```
![](https://velog.velcdn.com/images/ddoddo/post/9062e746-5c6c-4927-9481-9e258963bdf5/image.png)


