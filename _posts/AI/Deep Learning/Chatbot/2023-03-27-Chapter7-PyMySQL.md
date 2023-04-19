---
title: "[Chatbot] Chapter7 파이썬으로 데이터베이스 연동"
date: 2023-03-27

categories:
  - AI
  - Deep Learning
tags:
  - MySQL
  - Python
  - Chatbot
---


## 챗봇 학습툴 만들기

### MySQL
- 오픈소스 관계형 데이터베이스 관리 시스템
- 파이썬을 포함한 다양한 언어에서 사용할 수 있도록 API 지원

### 파이썬으로 데이터베이스 연동하기
- 파이썬에서 MySQL 사용하기 위해서는 `MySQL 클라이언트 라이브러리`를 사용
- 고수준 API를 지원하며 무료로 사용할 수 있는 `PyMySQL` 모듈이 공개되어 있음

|명령어|설명|
|:---:|:---:|
|SELECT|데이터 조회|
|INSERT|데이터 삽입|
|UPDATE|데이터 변경|
|DELETE|데이터 삭제|


```python
import pymysql
```

### 데이터베이스 연결
- `host` : 데이터베이스 서버가 존재하는 호스트 주소
- `user` : 데이터베이스 로그인 유저
- `passwd` : 데이터베이스 로그인 패스워드
- `db` : 데이터베이스명
- `charset` : 데이터베이스에서 사용할 charset 인코딩


```python
db = pymysql.connect(
	user='root', 
    passwd='******', 
    host='127.0.0.1', 
    port=3306,
    db='database name', 
    charset='utf8'
)
```

데이터베이스 사용이 종료된 이후 반드시 DB 연결을 닫아야 함


```python
db.close()
```

### 데이터 조작

- 테이블 생성, 데이터 삽입, 조회, 변경, 삭제


```python
# 테이블 생성
import pymysql

db = None
try:
    # DB 호스트 정보 입력
    db = pymysql.connect(
    user='root', 
    passwd='******', 
    host='127.0.0.1', 
    port=3306,
    db='database name', 
    charset='utf8'
    )

    # 테이블 생성 SQL 정의
    sql = '''
    CREATE TABLE tb_student(
        id int primary key auto_increment not null,
        name varchar(32),
        age int,
        address varchar(32)
    )
    '''

    # 테이블 생성
    with db.cursor() as cursor:
        cursor.execute(sql)

# DB 연결 실패시 오류 내용 출력
except Exception as e:
    print(e)

# DB 연결된 경우에만 접속 닫기 시도
finally:
    if db is not None:
        db.close()
```
![image](https://user-images.githubusercontent.com/100760303/227845719-ce677697-1dd3-4f1c-b713-24f2f5cb5312.png)

```python
# 데이터 삽입
import pymysql

db = None
try:
    # DB 호스트 정보 입력
    db = pymysql.connect(
    user='root', 
    passwd='******', 
    host='127.0.0.1', 
    port=3306,
    db='database name', 
    charset='utf8'
    )

    # 데이터 삽입  SQL 정의
    sql = '''
    INSERT tb_student(name, age, address) VALUES('Kei', 35, 'Korea')
    '''

    # 데이터 삽입
    with db.cursor() as cursor:
        cursor.execute(sql)
    db.commit()

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
```
![image](https://user-images.githubusercontent.com/100760303/227845759-8f1d6376-29b0-4e65-936d-58b24dedc770.png)


```python
# 데이터 변경
import pymysql

db = None
try:
    # DB 호스트 정보 입력
    db = pymysql.connect(
    user='root', 
    passwd='******', 
    host='127.0.0.1', 
    port=3306,
    db='database name', 
    charset='utf8'
    )

    # 데이터 수정 SQL 정의
    id = 1
    sql = '''
    UPDATE tb_student SET name='케이', age=36 WHERE id=%d
    ''' % id

    # 데이터 수정
    with db.cursor() as cursor:
        cursor.execute(sql)

    db.commit()

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
```
![image](https://user-images.githubusercontent.com/100760303/227845819-fe39766c-95cf-4d78-86ec-a131877b0b9b.png)


```python
# 데이터 삭제
import pymysql

db = None
try:
    # DB 호스트 정보 입력
    db = pymysql.connect(
    user='root', 
    passwd='******', 
    host='127.0.0.1', 
    port=3306,
    db='database name', 
    charset='utf8'
    )

    # 데이터 수정 SQL 정의
    id = 1
    sql = '''
    DELETE FROM tb_student WHERE id=%d
    ''' % id

    # 데이터 수정
    with db.cursor() as cursor:
        cursor.execute(sql)

    db.commit()

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
```
![image](https://user-images.githubusercontent.com/100760303/227845854-70be3b48-dc87-4e8d-bdf1-45795f53a5e2.png)

- 다수의 데이터를 임의로 DB에 삽입해 데이터를 조회


```python
import pymysql
import pandas as pd

# DB 호스트 정보 입력
db = pymysql.connect(
user='root', 
passwd='******', 
host='127.0.0.1', 
port=3306,
db='database name', 
charset='utf8'
)
```


```python
# 데이터 DB에 추가
students = [
    {'name':'Kei', 'age':36, 'address':'PUSAN'},
    {'name':'Tony', 'age':34, 'address':'PUSAN'},
    {'name':'Jaeyoo', 'age':39, 'address':'GWANGJU'},
    {'name':'Grace', 'age':28, 'address':'SEOUL'},
    {'name':'Jenny', 'age':27, 'address':'SEOUL'}
]
```


```python
for s in students:
    with db.cursor() as cursor:
        sql = '''
            INSERT tb_student(name, age, address) VALUES('%s', %d, '%s')
        ''' % (s['name'], s['age'], s['address'])
        cursor.execute(sql)
    db.commit()
```


```python
# 30대 학생만 조회
cond_age = 30
with db.cursor(pymysql.cursors.DictCursor) as cursor:
    sql = '''
        SELECT * FROM tb_student WHERE age > %d
    ''' % cond_age
    cursor.execute(sql)
    results = cursor.fetchall()
    
print(results)
```

    [{'id': 1, 'name': 'Kei', 'age': 36, 'address': 'PUSAN'}, {'id': 2, 'name': 'Tony', 'age': 34, 'address': 'PUSAN'}, {'id': 3, 'name': 'Jaeyoo', 'age': 39, 'address': 'GWANGJU'}]
    


```python
# pandas 데이터 프레임으로 표현
df = pd.DataFrame(results)
print(df)
```

       id    name  age  address
    0   1     Kei   36    PUSAN
    1   2    Tony   34    PUSAN
    2   3  Jaeyoo   39  GWANGJU
    


```python
# 이름 검색
cond_name = 'Grace'
with db.cursor(pymysql.cursors.DictCursor) as cursor:
    sql = '''
        SELECT * FROM tb_student WHERE name='%s'
    ''' % cond_name
    cursor.execute(sql)
    result = cursor.fetchone()
    
print(result['name'], result['age'])
```

    Grace 28
    
![image](https://user-images.githubusercontent.com/100760303/227845895-e387a04b-e1bf-48b0-aaac-14f4af26ad76.png)
