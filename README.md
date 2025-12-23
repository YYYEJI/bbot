# 🤖 BeBot: 창조과학 RAG 챗봇 🤖

BeBot은 **PostgreSQL 기반 RAG( Retrieval-Augmented Generation )** 챗봇으로, 창조과학 관련 질문에 대해 텍스트와 임베딩 기반 문서를 참고하여 성경적 관점을 반영한 답변을 제공합니다.  

---

## 🔹 주요 기능

- 로컬 PostgreSQL DB에 크롤링 데이터 저장
- 텍스트 임베딩을 PostgreSQL `vector` 컬럼에 저장 (pgvector 활용)
- 질문에 맞는 관련 문서 검색 (RAG)
- OpenAI / Upstage API 기반 자연어 생성
- 한국어/영어 자동 감지 및 답변 생성

---

## 📂 프로젝트 구조
bbot/

├─ bbot_ui.py          # Streamlit UI 실행 파일

├─ bbot.py             # DB 연결, RAG, 질문 생성, 임베딩 처리 등 핵심 코드

├─ extracted_texts/    # 텍스트 추출 데이터 (title, url, crawl_time, content)

├─ README.md

---

## ⚙️ 설치 및 환경 설정

1. **Python 가상환경 생성**
```bash
conda create -n bbot python=3.12
conda activate bbot
```

2. 필요 패키지 설치
```
pip install -r requirements.txt
```

3. 환경 변수 설정(.env)
```
UPSTAGE_API_KEY=<YOUR_UPSTAGE_API_KEY>
UPSTAGE_BASE_URL=<YOUR_UPSTAGE_BASE_URL>
DB_HOST=localhost
DB_NAME=bbot_db
DB_USER=rosie
DB_PASSWORD=""
DB_PORT=5433
```

4.	PostgreSQL 및 pgvector 설치
```
brew install postgresql@15
brew services start postgresql@15
brew install pgvector
```

---
## 🗄️ DB 초기화 및 데이터 저장
•	bbot.py 내 create_db(metas) 함수 실행 시 자동으로 테이블 생성 및 데이터를 삽입합니다.
•	테이블 구조 (PostgreSQL):
```
CREATE TABLE IF NOT EXISTS crawled_data (
    id SERIAL PRIMARY KEY,
    title TEXT,
    url TEXT,
    crawl_time TIMESTAMP,
    content TEXT,
    content_embedding vector(4096)
);
```

---
## 🚀 실행 방법
1. streamlit UI 실행
```
streamlit run bbot_ui.py
```

2. 웹 브라우저에서 접속 후 질문
•	사용자 질문 입력 → RAG 기반 답변 생성

•	한국어/영어 자동 감지


--- 
## 📦 최종 산출물
	•	bbot_ui.py : Streamlit UI 파일
	•	bbot.py : 핵심 로직 및 DB 연동
	•	extracted_texts : 크롤링 데이터
	•	PostgreSQL DB (bbot_db)
	•	.env : 환경 변수
	•	Streamlit 데모 화면에서 생성된 질의 응답 기록

---
## 📌 주의 사항
	•	pgvector 확장 설치 필수 (PostgreSQL 15 기준)
	•	임베딩 차원 수(4096)는 Upstage 모델 기준
	•	로컬 DB 실행 후, create_db() 실행 필요