---
title: "[Chatbot] Chapter7 챗봇 학습툴 만들기"
date: 2023-03-28

categories:
  - AI
  - Deep Learning
tags:
  - MySQL
  - Python
  - Chatbot
---

## 챗봇 학습툴 만들기

- 챗봇이 학습 데이터를 관리하는 툴 만들기
- 학습 데이터를 DB에 저장했을 때 실시간으로 챗봇 시스템에 적용될 수 있도록 제작
- 챗봇이 이해할 수 있는 질문, 답변 데이터를 관리하기 위한 툴 필요
- 챗봇의 답변 출력을 위해서만 사용되는 데이터
- 챗봇 학습툴을 통해 저장된 질문 유형과 답변만 챗봇 엔진이 처리할 수 있음

### 프로젝트 구조
- `train_tools` : 챗봇 학습툴 관련 파일
- `models` : 챗봇 엔진에서 사용하는 딥러닝 모델 관련 파일
    - `intent` : 의도 분류 모델 관련 파일
    - `ner` : 개체 인식 모델 관련 파일
- `utils` : 챗봇 개발에 필요한 유틸리티
- `config` : 챗봇 개발에 필요한 설정
- `test` : 챗봇 개발에 필요한 테스트 코드

### 학습용 데이터베이스 설계 및 데이터 테이블 생성

|컬럼|속성|설명|
|:---:|:---:|:---:|
|id|int primary key not null|학습 데이터 id|
|intent|varchar(45)|의도명, 의도가 없는 경우 NULL|
|ner|varchar(45)|개체명, 개체명이 없는 경우 NULL|
|query|text null|질문 텍스트|
|answer|text not null|답변 텍스트|
|answer_image|varchar(2048)|답변에 들어갈 이미지 URL, 이미지 URL을 사용하지 않을 경우 NULL|

- DB 서버 접속 정보를 `/config` 디렉토리 내에 파일로 따로 관리
- `DatabaseConfig.py` 파일을 생성하여 아래와 같이 작성


```python
# DatabaseConfig.py - DB 접속 정보
DB_HOST = '127.0.0.1'
DB_USER = 'root'
DB_PASSWORD = '******'
DB_NAME = 'chatbot'

def DatabaseConfig():
    global DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
```

- 챗봇 데이터 학습용 테이블 생성
- `/train_tools/qna` 디렉토리 내에 생성
- `create_train_data_table.py` 파일을 생성하여 아래와 같이 작성


```python
# create_train_data_table.py - 학습용 테이블 생성
import pymysql
from config.DatabaseConfig import *
```


```python
db = None
try:
    db = pymysql.connect(
        user=DB_USER, 
        passwd=DB_PASSWORD, 
        host=DB_HOST, 
        db=DB_NAME, 
        charset='utf8'
    )

    # 테이블 생성 SQL 정의
    sql = '''
        CREATE TABLE IF NOT EXISTS `chatbot_train_data` (
        `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
        `intent` VARCHAR(45) NULL,
        `ner` VARCHAR(45) NULL,
        `query` TEXT NULL,
        `answer` TEXT NOT NULL,
        `answer_image` VARCHAR(2048) NULL,
        PRIMARY KEY (`id`)
        )
        ENGINE = InnoDB DEFAULT CHARSET=utf8
    '''

    # 테이블 생성
    with db.cursor() as cursor:
        cursor.execute(sql)

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
```

### 챗봇 학습 데이터 엑셀 파일 및 DB 연동
- 엑셀 파일을 학습툴에 입력해 DB 내용을 업데이트 하는 형태로 만듦
- `train_data.xlsx`


```python
# load_train_data.py - 엑셀 파일을 읽어와 DB와 데이터 동기화
import pymysql
import openpyxl

from config.DatabaseConfig import *

# 학습 데이터 초기화
def all_clear_train_data(db):
    # 기존 학습 데이터 삭제
    sql = '''
        DELETE FROM chatbot_train_data
    '''

    with db.cursor() as cursor:
        cursor.execute(sql)

    # Auto Increment 초기화
    sql = '''
        ALTER TABLE chatbot_train_data AUTO_INCREMENT=1
    '''

    with db.cursor() as cursor:
        cursor.execute(sql)

# DB에 데이터 저장
def insert_data(db, xls_row):
    intent, ner, query, answer, answer_img_url = xls_row

    sql = '''
        INSERT chatbot_train_data(intent, ner, query, answer, answer_image)
        VALUES('%s', '%s', '%s', '%s', '%s')
    ''' % (intent.value, ner.value, query.value, answer.value, answer_img_url.value)

    # 엑셀에서 불로운 셀에 데이터가 없는 경우 NULL로 치환
    sql = sql.replace("'None'", "null")

    with db.cursor() as cursor:
        cursor.execute(sql)
        print('{} 저장'.format(query.value))
        db.commit()

train_file = 'train_data.xlsx'
db = None
try:
    db = pymysql.connect(
        user=DB_USER, 
        passwd=DB_PASSWORD, 
        host=DB_HOST, 
        port=3306,
        db=DB_NAME, 
        charset='utf8'
    )

    # 기존 학습 데이터 초기화
    all_clear_train_data(db)

    # 학습 엑셀 파일 불러오기
    wb = openpyxl.load_workbook(train_file)
    sheet = wb['Sheet1']
    for row in sheet.iter_rows(min_row=2): # header는 불러오지 않음
        # 데이터 저장
        insert_data(db, row)

    wb.close()

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
```

    안녕하세요 저장
    반가워요 저장
    
![image](https://user-images.githubusercontent.com/100760303/228180939-d780394c-7bcb-42eb-bee2-de79be11def2.png)
