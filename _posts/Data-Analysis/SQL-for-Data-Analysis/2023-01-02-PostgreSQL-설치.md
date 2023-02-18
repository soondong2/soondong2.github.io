---
title: PostgreSQL 설치 및 실행
date: 2023-01-02

categories:
  - Data Analysis
  - SQL for Data Analysis
tags:
    - SQL
---

## PostgreSQL 설치
아래 사이트에서 본인의 운영체제에 맞는 PostgreSQL을 다운로드하여 설치한다.

https://www.enterprisedb.com/downloads/postgres-postgresql-downloads

![](https://velog.velcdn.com/images/ddoddo/post/583b7d76-8f3f-46cf-a910-faafbd4b39bf/image.png) <br>
위 경우만 사진과 같은 상태로 설정하였고, 나머지는 모두 기본값으로 설정하여 Next를 눌러 진행하였다.

![](https://velog.velcdn.com/images/ddoddo/post/2d0abd0b-26aa-4c11-94b7-b94219b666da/image.png) <br>
비밀번호를 설정한다. 추후 DB 접속할 때 필요하므로 잘 기억해둔다.

<br> 

## PostgreSQL 실행
PostgreSQL을 설치할 때 함께 설치했던 PostgreSQL 관리 도구인 `pgAdmin 4`를 사용하여 실습한다. 프로그램을 실행하면 비밀번호를 입력하라는 창이 뜬다. 설치할 때 설정했던 비밀번호를 입력해준다.

![](https://velog.velcdn.com/images/ddoddo/post/8b9bcd34-ebe1-48ef-bf34-74220fa1e9a2/image.png) <br>
중앙에 보이는 `Add New Server`를 클릭한다.

`Register - Server` 창이 뜨게 되는데, [General] 탭에서 `Name`에 원하는 데이터베이스명을 입력해준다. 이후 [Connection] 탭에서 `Host name/addres`에 localhost를 입력해주고, `password` 칸에 비밀번호를 입력한 후 하단의 Save 버튼을 눌러 생성시켜준다.

![](https://velog.velcdn.com/images/ddoddo/post/83f7681d-c230-4a83-aa43-279cd7aeee9e/image.png) <br>
새로 생성한 서버의 데이터베이스 아래 postgres에서 마우스 우클릭하여 `Query Tool`을 선택하면 쿼리 에디터를 열 수 있다. 쿼리 에디터에 테이블을 생성하는 코드를 입력한 후 실행 버튼을 누른다.

<br>

```sql
SELECT * FROM 테이블명;
```
위 코드를 입력하면 생성한 테이블을 확인할 수 있다. 코드 실행 아이콘 클릭 대신 `F5`를 누르면 코드가 실행된다.

---

## 공부 교재
> SQL로 시작하는 데이터 분석
(실무에 꼭 필요한 분석 기법 총정리!)

위 교재 자료를 통해 공부를 하기로 했다. 평소 MySQL을 사용해왔으나 해당 교재에서 PostgreSQL을 기준으로 설명하는 것 같다.

전체적인 문법은 같으니까 세부적으로 다른 부분만 정리하면서 공부해보려고 한다. 본격 공부 진행 전 설치와 데이터 준비를 완료했다.
