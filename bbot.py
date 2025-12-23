import os
import json
from datetime import datetime
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv

import psycopg2
from openai import OpenAI
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from tiktoken import encoding_for_model
import tiktoken




# .env 파일 로드 및 환경변수 설정
load_dotenv()   
api_key = os.getenv("UPSTAGE_API_KEY")
base_url = os.getenv("UPSTAGE_BASE_URL")




# Upstage 모델 생성 
model = OpenAI(api_key=api_key, base_url=base_url)
embedding_model = UpstageEmbeddings(
    upstage_api_key=api_key,
    model="embedding-query"
)



# DB 생성 및 문서 처리
enc = tiktoken.get_encoding("cl100k_base")     # tokenizer

def count_tokens(text: str) -> int:            # 토큰 수 세기
    return len(enc.encode(text))               # 토큰 수 반환

def split_text_by_tokens(text: str, max_tokens: int = 4000):
    words = text.split()                           
    chunks = []                
    chunk = []                            
    tokens_so_far = 0                  

    for word in words:                   
        word_tokens = count_tokens(word + " ")     
        if tokens_so_far + word_tokens > max_tokens:     
            chunks.append(" ".join(chunk))        
            chunk = []                                        
            tokens_so_far = 0             
        chunk.append(word)                         
        tokens_so_far += word_tokens            

    if chunk:                                     
        chunks.append(" ".join(chunk))             
    return chunks                             


def create_db(folder_path: str, db_name: str = "bbot_db", max_tokens: int = 4000):
    """extracted_texts 폴더 안 모든 텍스트 파일을 임베딩하여 DB에 저장"""

    # PostgreSQL 연결
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),                 # PostgreSQL 호스트
        dbname=db_name,                            # 데이터베이스 이름
        user=os.getenv("DB_USER"),                 # 사용자 이름
        password=os.getenv("DB_PASSWORD"),         # 비밀번호
        port=os.getenv("DB_PORT")                  # 포트 번호
    )
    cur = conn.cursor()                            # 커서 생성
    print("[DB] 연결 성공")                          # 연결 성공 콘솔 메시지 출력

    # pgvector 확장 
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # 테이블 생성 (id, title, url, crawl_time, content, content_embedding)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crawled_data (  
            id SERIAL PRIMARY KEY,
            title TEXT,
            url TEXT,
            crawl_time TIMESTAMP,
            content TEXT,
            content_embedding vector(4096)
        );
    """)

    # failed files 리스트 초기화
    failed_files = []
    # 삽입된 청크 수 카운트
    inserted_count = 0

    # 폴더 내 모든 .txt 파일 처리 (크롤링된 데이터 - 창조과학사이트)
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    # 폴더 내 파일 개수 출력
    print(f"[DB] 폴더에서 {len(files)}개 파일 발견")

    # 각 파일 처리
    for idx, fname in enumerate(files, start=1):
        try:
            # 파일 내용 읽기
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                content = f.read()

            # 길면 청크로 분할
            chunks = split_text_by_tokens(content, max_tokens=max_tokens)

            # 각 청크를 DB에 삽입
            for chunk in chunks:
                title = fname.replace(".txt", "")
                url = ""
                crawl_time = datetime.now()

                embedding_vector = embedding_model.embed_query(chunk)

                cur.execute(
                    "INSERT INTO crawled_data (title, url, crawl_time, content, content_embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                    (title, url, crawl_time, chunk, embedding_vector)
                )
                inserted_count += 1

            # 진행 상황 출력
            if idx % 100 == 0:
                print(f"[DB] {idx}개 파일 처리 완료")

        # 예외 처리
        except Exception as e:
            # 실패된 파일 기록 및 롤백
            print(f"[ERROR] 파일 삽입 실패: {fname} / {e}")
            failed_files.append(fname)
            conn.rollback()

    # 변경사항 커밋
    conn.commit()

    # 최종 통계 출력
    cur.execute("SELECT COUNT(*) FROM crawled_data;")
    total_count = cur.fetchone()[0]
    print(f"[DB] 총 데이터 개수: {total_count}")
    print(f"[DB] 성공적으로 삽입된 청크 수: {inserted_count}")
    print(f"[DB] 실패한 파일 수: {len(failed_files)}")
    if failed_files:
        print("[DB] 실패한 파일 일부:", failed_files[:20])

    # 연결 종료
    cur.close()
    conn.close()
    # 데이터 삽입 완료 메세지 출력 
    print("[DB] 데이터 삽입 완료")




# 언어 감지
def detect_language(text: str) -> str:
    return "ko" if any('\uac00' <= ch <= '\ud7a3' for ch in text) else "en"

# 라우터
def router(question: str) -> str:
    """
    창조과학/성경/기독교 질문만 허용
    일반 상식 질문은 거절
    """
    print("[Router] 질문 의도 분석 중...")

    keywords = ["창조", "성경", "하나님", "진화", "복음", "아담", "노아", "대홍수", "창세기", "기독교", "세계관", "믿음", "예수님", "구원"]
    decision = "internal" if any(k in question for k in keywords) else "internal"

    print(f"[Router] 선택된 경로: {decision}")
    return decision




# 유사 문서 검색 
def retrieve_documents(question: str, top_k: int = 5):
    print("[Retrieve] 벡터 검색 시작")

    # 질문 임베딩 생성
    q_embedding = embedding_model.embed_query(question)

    # PostgreSQL 연결
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
    # 커서 생성
    cur = conn.cursor()

    # 벡터 유사도 검색 쿼리 실행
    cur.execute("""
        SELECT title, url, content
        FROM crawled_data
        ORDER BY content_embedding <#> %s::vector
        LIMIT %s
    """, (q_embedding, top_k))

    # 검색 결과 가져오기
    rows = cur.fetchall()
    # 연결 종료
    cur.close()
    conn.close()

    # 문서 리스트 생성
    docs = [
        {"title": r[0], "url": r[1], "content": r[2]}
        for r in rows
    ]

    # 검색된 문서 수 출력
    print(f"[Retrieve] 검색 문서 수: {len(docs)}")

    # 각 문서 정보 출력 (미리보기)
    for i, d in enumerate(docs, start=1):
        print(f"\n[Doc {i}]")
        print(f"Title: {d['title']}")
        print(f"URL: {d['url']}")
        print(f"Content (preview): {d['content'][:300]}")  # 앞 300자만

    # 문서 반환
    return docs



# 검색 결과 판단 (Judge)
# 검색된 문서들이 질문에 충분한 정보를 제공하는지 판단
class RetrievalJudge(BaseModel):
    judgement: str
    binary_score: str

def judge_retrieval(question: str, docs: List[dict]) -> RetrievalJudge:
    print("[Judge] 검색 결과 평가 중...")

    # 문서가 없으면 바로 not_resolved 반환
    if not docs:
        return RetrievalJudge(judgement="not_resolved", binary_score="no")

    # 문서 내용 500자씩 결합
    joined_docs = "\n".join(d["content"][:500] for d in docs)

    prompt = f"""
    사용자 질문에 대해 아래 문서들이 충분한 정보를 제공하는지 판단하세요.

    Question:
    {question}

    Documents:
    {joined_docs}

    JSON 형식으로만 응답:
    {{
        "judgement": "resolved" or "not_resolved",
        "binary_score": "yes" or "no"
    }}
    """

    # LLM 호출
    res = model.chat.completions.create(
        model="solar-pro2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # 응답 내용 추출
    content = res.choices[0].message.content

    # JSON 파싱
    try:
        # JSON 시작 위치 찾기
        json_start = content.find("{")
        if json_start == -1:
            raise ValueError("JSON not found in LLM response")
        json_obj = json.loads(content[json_start:])
        result = RetrievalJudge(**json_obj)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Judge] JSON 파싱 실패: {e}")
        result = RetrievalJudge(judgement="not_resolved", binary_score="no")

    print(f"[Judge] 판단 결과: {result.judgement}")
    return result




# 질문 재작성 (Rewriter)
# 질문을 재작성함으로써 검색 성능 향상 
system_rewriter = """
당신은 RAG 검색 성능을 높이기 위해 질문을 더 명확하고 구체적으로 재작성하는 전문가입니다.
"""

# "당신은 질문을 더 잘 쓰는 전문가입니다" 같은 지침이 들어 있음
prompt_rewriter = ChatPromptTemplate.from_messages([
    ("system", system_rewriter),
    ("human", "Original question: {question}")
])

def rewrite_query(question: str) -> str:
    print("[Rewrite] 질문 재작성 중...")

    # 질문 재작성 체인 
    chain = (
        # 1) prompt_rewriter (system + human 프롬프트 템플릿)
        prompt_rewriter
        # 2) LLM(solar-pro2)을 호출하는 부분
        | RunnableLambda(
            lambda p: model.chat.completions.create(
                model="solar-pro2",
                messages=[{"role": "user", "content": p.to_string()}],
                temperature=0
            ).choices[0].message.content
        )
        # 3) LLM 출력 결과를 문자열(str)로 정리하는 부분
        | StrOutputParser()
    )
    # 체인 실행
    rewritten = chain.invoke({"question": question})
    print(f"[Rewrite] 재작성된 질문: {rewritten}")
    return rewritten



# 답변 생성 
def generate_answer(question: str, docs: List[dict]) -> str:
    print("[Generate] 답변 생성 중...")

    lang = detect_language(question)
    lang_inst = "한국어로 답변하세요." if lang == "ko" else "Answer in English."

    context = ""
    for d in docs:
        context += f"Title: {d['title']}\nContent: {d['content']}\nURL: {d['url']}\n\n"

    system_prompt = f"""
    당신은 기독교적 세계관과 창조론에 기반해 답변하는 전문가입니다.
    규칙:
    - 반드시 제공된 문서 내용만 사용하세요.
    - 문서에 없는 내용은 절대 추측하지 마세요.
    - 정보가 없으면: "제공된 문서에는 해당 정보가 없습니다." 라고 답하세요.
    - 답변 마지막에 사용한 문서의 URL을 포함하세요.
    - 웹 검색은 성경 구절을 찾을 때만 사용하세요.


    {lang_inst}

    참고 문서:
    {context}
    """

    res = model.chat.completions.create(
        model="solar-pro2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0
    )

    return res.choices[0].message.content



# Main agent
def generate(question: str) -> str:
    # 콘솔 메세지 출력으로 시작 
    print("\n===== NEW QUERY =====")
    # 유저 메세지 출력 
    print(f"User Question: {question}")

    # 라우터 호출 (창조과학/성경 질문만 허용)
    route = router(question)

    # 유사 문서 검색 
    docs = retrieve_documents(question)
    # 검색된 문서 판단 
    judge = judge_retrieval(question, docs)

    # 판단 결과에 따라 질문 재작성 및 재검색 
    if judge.judgement == "not_resolved":
        new_question = rewrite_query(question)
        docs = retrieve_documents(new_question)

    # 최종 답변 생성 (문서 기반)
    answer = generate_answer(question, docs)
    # 마지막 콘솔 출력 
    print("[Done] 응답 완료\n")
    return answer