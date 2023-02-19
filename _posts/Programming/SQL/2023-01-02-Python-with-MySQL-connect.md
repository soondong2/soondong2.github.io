---
title: Python으로 MySQL 연결하기
date: 2023-01-02

categories:
  - Programming
  - SQL
tags:
    - SQL
---

## 설치 및 연결
```python
pip install PyMySQL
```
```python
import pymysql
```
```python
db = pymysql.connect(
	user='root', 
    passwd='비밀번호', 
    host='127.0.0.1', 
    port=3306,
    db='데이터베이스', 
    charset='utf8'
)
```
위 코드를 실행하다가 여러가지 오류를 만났다.
구글링 해보니 크게 비밀번호, DB명 오류였다.

<br>

### 비밀번호 오류
MySQL Command Line Client 실행해서 비밀번호 재설정을 해주면 해결된다. 
```
ALTER USER 'local'@'host' IDENTIFIED BY '변경할 비밀번호'
```

<br>

### DB명 오류
DB 접속해서 DB명을 제대로 확인하자 ^^.. 엉뚱한 거 적고 삽질 했다능~

<br>

## 연결
`connect()` 함수를 이용하면 MySQL host내 DB와 직접 연결할 수 있다.

<br>

## cursor 설정
연결한 DB와 상호작용하기 위해 cursor 객체를 생성해준다.
```python
cursor = db.cursor(pymysql.cursors.DictCursor)
```
다양한 커서의 종류가 있지만, `Python`에서 데이터 분석을 주로 `pandas`로 하고, `RDBMS`(Relational Database System)를 주로 사용하기 때문에 데이터 분석가에게 익숙한 데이터프레임 형태로 결과를 쉽게 변환할 수 있도록 딕셔너리 형태로 결과를 반환해주는 `DictCursor`를 사용한다.

<br>

## 데이터 조작
```python
sql = "SELECT * FROM `테이블명`;"
cursor.execute(sql)
result = cursor.fetchall()
```
`cursor.execute(sql)`를 통해 SQL문을 실행한다.

- `fetchall()` : 모든 데이터를 한 번에 가져올 때 사용한다.
- `fetchone()` : 한 번 호출에 하나의 행만 가져올 때 사용한다.
- `fetchmany(n)` : n개만큼의 데이터를 가져올 때 사용한다.

<br>

```python
import pandas as pd

result = pd.DataFrame(result)
result
```
위 코드를 실행하면 테이블의 모든 row가 pandas의 데이터프레임 형태로 출력된다.
