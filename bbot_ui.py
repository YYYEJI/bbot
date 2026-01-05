# Bebot UI
import streamlit as st
import json
from bbot import create_db, generate
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv() 


# DB 한 번만 생성 
def table_exists(table_name: str) -> bool:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
    cur = conn.cursor()

    cur.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = %s
        );
    """, (table_name,))

    exists = cur.fetchone()[0]
    cur.close()
    conn.close()
    return exists

if "db_ready" not in st.session_state:
    print("✔️ DB 체크 중")
    if not table_exists("crawled_data"):
        with st.spinner("✔️ DB 생성 중…"):
            create_db("./extracted_texts")
    st.session_state.db_ready = True
    print("✔️ DB 준비 완료")



# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Creation Science Q&A ✝️")


# 이전 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 사용자 입력 처리
if prompt := st.chat_input("창조과학·성경이 궁금하신가요? ✨ Ask me about Creation Science 🤖"):
    # 사용자 메시지 출력
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner(" Searching ... "):
            response = generate(prompt)  

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})