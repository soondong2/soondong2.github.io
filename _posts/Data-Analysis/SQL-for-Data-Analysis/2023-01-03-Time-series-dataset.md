---
title: Time series dataset 준비
date: 2023-01-03

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## csv 파일 다운로드
us_retail_sales.csv 파일을 다운로드 받는다.

## 파일 생성
pdAdmin 4 프로그램을 실행하여 테이블을 생성한다.
`retail_sales` 테이블을 생성하고, 다운 받은 데이터 파일의 값들을 복사하여 집어 넣는다.

```sql
-- retail_sales 테이블이 존재하지 않는다면 테이블을 생성한다.
DROP table if exists retail_sales;
CREATE table retail_sales
(
  sales_month date,
  naics_code varchar,
  kind_of_business varchar,
  reason_for_null varchar,
  sales decimal
)
;

-- 다운 받은 csv 값을 생성한 테이블의 값으로 넣어준다.
-- 본인의 파일 경로를 넣어준다.
COPY retail_sales FROM '파일 경로/us_retail_sales.csv' DELIMITER ',' CSV HEADER;
```

## 데이터 확인
```sql
SELECT * FROM retail_sales;
```
![](https://velog.velcdn.com/images/ddoddo/post/7e4708f1-f810-4bb5-9b2a-3ed74a31da18/image.png)
