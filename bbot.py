import os
import json
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_upstage import UpstageEmbeddings
from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
base_url = os.getenv("SUPABASE_URL")

# 🔹 환경 변수 로드
api_key = os.getenv("UPSTAGE_API_KEY")
api_key = os.environ["UPSTAGE_API_KEY"]
base_url = os.environ["UPSTAGE_BASE_URL"]

# 🔹 Upstage 모델

model = OpenAI(api_key=api_key, base_url=base_url)

# =========================
# DB 생성 및 문서 처리
# =========================
def create_db(metas: List[dict], persist_dir: str = "./chroma_db") -> Chroma:
    """메타 데이터를 기반으로 Chroma VectorStore 생성"""
    docs = [
        Document(
            page_content=m["Content"],
            metadata={"URL": m["URL"], "Title": m["Title"], "CrawlTime": m["CrawlTime"]}
        )
        for m in metas
    ]
    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    
    # 임베딩 모델
    embedding_model = UpstageEmbeddings(upstage_api_key=api_key, model="embedding-query")
    
    # Chroma DB
    db = Chroma.from_documents(documents=split_docs, embedding=embedding_model, persist_directory=persist_dir)
    return db


# =========================
# 언어 자동 감지 함수
# =========================
def detect_language(text: str) -> str:
    """한국어 포함 여부로 언어 감지"""
    if any('\uac00' <= ch <= '\ud7a3' for ch in text):
        return "ko"
    return "en"


# =========================
# RAG 관련 함수
# =========================
def web_search(question: str) -> str:
    # 웹 검색 placeholder
    return "웹 검색 결과 텍스트"

def generate(question: str) -> str:
    lang = detect_language(question)

    if lang == "ko":
        lang_instruction = "사용자 질문이 한국어이므로 한국어로 자연스럽게 답변하세요."
    else:
        lang_instruction = "The user asked in English, so answer naturally and fluently in English."

    system_prompt = f"""
    당신은 기독교적 관점에서 답변하는 전문가입니다.
    질문이 일반적이거나 과학적이어도, 답변에 반드시 성경적 또는 기독교적 관점을 반영해야 합니다.

    {lang_instruction}
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
# Question Rewriting
# =========================
system = """You are a question re-writer that converts an input question
to a better version optimized for vectorstore retrieval.
Analyze the input and reason about the underlying semantic intent/need of the question, then rewrite it accordingly."""

prompt_rewriter = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Original question: {question}")
])

def upstage_rewriter(prompt_value):
    prompt_text = prompt_value.to_string()
    response = model.chat.completions.create(
        model="solar-pro2",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt_text}],
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
    data = json.loads(response.choices[0].message.content)
    return Relevancy(**data)

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
    data = json.loads(response.choices[0].message.content)
    return Factfulness(**data)

# =========================
# State 정의
# =========================
class State(TypedDict):
    question: str
    generation: str
    documents: List[str]
    source: str