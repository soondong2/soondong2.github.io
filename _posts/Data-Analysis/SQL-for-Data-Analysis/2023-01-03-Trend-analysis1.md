---
title: 트렌드 분석 - 간단한 트렌드
date: 2023-01-02

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## 간단한 트렌드 분석
앞서 생성한 `retail_sales` 테이블을 활용하여 분석한다.
<br>

비즈니스 종류가 Retail and food services sales, total인 경우의 sales_month 컬럼만 조회한다.
```sql
SELECT sales_month
FROM retail_sales
WHERE kind_of_business = 'Retail and food services sales, total'
ORDER BY sales_month ASC;
```

<br>

비즈니스 종류가 Retail and food services sales, total인 경우의 sales_month 컬럼에서 년도만 조회한다.
```sql
SELECT DATE_PART('year', sales_month) AS sales_year
FROM retail_sales
WHERE kind_of_business = 'Retail and food services sales, total'
GROUP BY DATE_PART('year', sales_month)
ORDER BY DATE_PART('year', sales_month);
```

---
## PostgreSQL vs MySQL
```sql
-- MySQL에서는 가능한 쿼리, PostgreSQL에서는 오류 발생
SELECT *
FROM table명
GROUP BY 컬럼
```
MySQL에서는 GROUP BY 절에 컬럼명을 입력한 후 SELECT에서 *로 모든 컬럼을 조회해도 데이터가 반환되지만, PostgreSQL에서는 GROUP BY에 입력한 컬럼을 반드시 사용해야 데이터가 반환되었다.

---
## PostgreSQL 문법
### DATE_PART()
- `field` : year, month, day 와 같은 날짜/시간 형태의 요소
- `source` : '2020-01-01 10:00:00' 와 같은 실제 시간 값

```sql
-- 사용 방법
DATE_PART(field, source)
```
![](https://velog.velcdn.com/images/ddoddo/post/9eca732e-37a8-4b51-a8cd-ba89f37c6df8/image.png)

반면 MySQL에는 `YEAR()`, `MONTH()`, `DAY()`, `DAYOFF()`, `DATE_FORMAT()` 함수들이 존재한다. 해당 함수들은 PostgreSQL에서는 존재하지 않는다.
