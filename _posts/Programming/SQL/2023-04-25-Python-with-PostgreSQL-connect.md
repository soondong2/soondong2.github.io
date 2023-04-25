---
title: Python으로 PostgreSQL 연결하기
date: 2023-04-25

categories:
  - Programming
  - SQL
tags:
  - Python
  - PostgreSQL
---

## Library Call
```python
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
```

## SQL 결과를 DataFrame으로 불러오기
- {} 안에 비밀번호 입력
```python
conn_string = 'postgresql://postgres:{Password}@localhost:5432/postgres'
postgres_engine = create_engine(conn_string)
```

- `"`을 활용해 쿼리를 작성해주어야 한다.
```python
query = """
SELECT * FROM Database.table;
"""

df = pd.read_sql_query(sql=query, con=postgres_engine)
df.head()
```
