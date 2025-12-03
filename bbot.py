import os
import json
from datetime import datetime
from typing import List
from pydantic import BaseModel
from typing_extensions import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 🔹 환경 변수 로드
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")
base_url = os.getenv("UPSTAGE_BASE_URL")

# 🔹 Upstage 모델
model = OpenAI(api_key=api_key, base_url=base_url)
embedding_model = UpstageEmbeddings(upstage_api_key=api_key, model="embedding-query")

# =========================
# DB 생성 및 문서 처리
# =========================
def create_db(metas: List[dict], db_name: str = "bbot_db") -> None:
    """메타 데이터를 PostgreSQL DB에 저장, content_embedding은 vector로 저장"""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=db_name,
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
    cur = conn.cursor()
    print("DB 연결 성공")

    # pgvector 확장 생성
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 테이블 생성
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

    # DB에 데이터 삽입
    for m in metas:
        title = m.get("Title", "")
        url = m.get("URL", "")
        crawl_time = m.get("CrawlTime", datetime.now())
        content = m.get("Content", "")

        if isinstance(crawl_time, str):
            crawl_time = datetime.fromisoformat(crawl_time)

        # content 임베딩 생성
        embedding_vector = embedding_model.embed_query(content)

        cur.execute(
            "INSERT INTO crawled_data (title, url, crawl_time, content, content_embedding) VALUES (%s, %s, %s, %s, %s::vector)",
            (title, url, crawl_time, content, embedding_vector)
        )

    conn.commit()
    cur.close()
    conn.close()

# =========================
# 언어 감지
# =========================
def detect_language(text: str) -> str:
    if any('\uac00' <= ch <= '\ud7a3' for ch in text):
        return "ko"
    return "en"

# =========================
# PostgreSQL 기반 RAG 리트리버
# =========================
def retrieve_documents(question: str, db_name: str = "bbot_db", top_k: int = 5):
    q_embedding = embedding_model.embed_query(question)
    
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=db_name,
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
    cur = conn.cursor()

    
    cur.execute("""
        SELECT title, url, content
        FROM crawled_data
        ORDER BY content_embedding <#> %s::vector
        LIMIT %s
    """, (q_embedding, top_k))

    results = cur.fetchall()
    cur.close()
    conn.close()
    
    documents = [{"title": r[0], "url": r[1], "content": r[2]} for r in results]
    return documents

# =========================
# RAG 답변 생성
# =========================
def generate(question: str, use_rag: bool = True) -> str:
    lang = detect_language(question)
    lang_instruction = "사용자 질문이 한국어이므로 한국어로 자연스럽게 답변하세요." if lang=="ko" else "Answer naturally in English."

    context_text = ""
    if use_rag:
        docs = retrieve_documents(question)
        for d in docs:
            context_text += f"Title: {d['title']}\nContent: {d['content']}\nURL: {d['url']}\n\n"

    system_prompt = f"""
    당신은 기독교적 관점에서 답변하는 전문가입니다.
    질문이 일반적이거나 과학적이어도, 답변에 반드시 성경적 관점을 반영해야 합니다.
    마지막에 출처도 반영하는데 참조한 데이터의 url입니다.

    {lang_instruction}

    참고 문서:
    {context_text}
    """

    response = model.chat.completions.create(
        model="solar-pro2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# =========================
# Question Rewriter
# =========================
system_rewriter = """You are a question re-writer that converts an input question
to a better version optimized for vectorstore retrieval."""

prompt_rewriter = ChatPromptTemplate.from_messages([
    ("system", system_rewriter),
    ("human", "Original question: {question}")
])

def upstage_rewriter(prompt_value):
    prompt_text = prompt_value.to_string()
    response = model.chat.completions.create(
        model="solar-pro2",
        messages=[{"role": "system", "content": system_rewriter}, {"role": "user", "content": prompt_text}],
        temperature=0
    )
    return response.choices[0].message.content

chain_rewriter = prompt_rewriter | RunnableLambda(upstage_rewriter) | StrOutputParser()

# =========================
# Relevancy 판단
# =========================
class Relevancy(BaseModel):
    judgement: str
    binary_score: str

def is_relevant(question: str, document: str) -> Relevancy:
    system_prompt = """You are an expert judge assessing the relevance of a document to a user question.
Respond strictly in valid JSON only."""
    prompt = f"""{system_prompt}\n\nRetrieved document:\n{document}\n\nUser question:\n{question}\n\nRespond in JSON format: {{"judgement": "relevant"/"not_relevant","binary_score": "yes"/"no"}}"""
    response = model.chat.completions.create(
        model="solar-pro2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return Relevancy(**json.loads(response.choices[0].message.content))

# =========================
# Factfulness 판단
# =========================
class Factfulness(BaseModel):
    judgement: str
    binary_score: str

def check_factfulness(document: str, generation: str) -> Factfulness:
    system_prompt = """You are a judge assessing whether an LLM generation is grounded in a set of retrieved documents.
Respond strictly in valid JSON only."""
    prompt = f"{system_prompt}\n\nSet of facts:\n{document}\n\nLLM generation:\n{generation}\n\nRespond in JSON format: {{'judgement':'factual'/'hallucinated','binary_score':'yes'/'no'}}"
    response = model.chat.completions.create(
        model="solar-pro2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return Factfulness(**json.loads(response.choices[0].message.content))

# =========================
# State 정의
# =========================
class State(TypedDict):
    question: str
    generation: str
    documents: List[str]
    source: str