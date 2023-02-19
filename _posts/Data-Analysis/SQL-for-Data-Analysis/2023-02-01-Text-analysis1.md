---
title: 텍스트 분석
date: 2023-02

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## 텍스트 분석
UFO 목격 보고 데이터 셋을 사용하여 분석한다.

```sql
SELECT *
FROM ufo;
```
![](https://velog.velcdn.com/images/ddoddo/post/e2183a3d-0822-445c-ad2c-e05368ecef56/image.png)

<br>

sighting_report 글자 수대로 그루핑하였을 때 각각에 해당하는 record 수를 계산한다.
```sql
SELECT LENGTH(sighting_report), COUNT(*) AS records
FROM ufo
GROUP BY 1
ORDER BY 1;
```
![](https://velog.velcdn.com/images/ddoddo/post/1e2ed8d1-0a0c-4a4e-97a1-291c950e567c/image.png)

