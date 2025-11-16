# chat_ui.py
import streamlit as st
import json
from bbot import create_db, generate

# 🔹 metas.json 읽기
with open("metas.json", "r", encoding="utf-8") as f:
    metas = json.load(f)

# 🔹 DB 한 번만 생성
if "db" not in st.session_state:
    st.session_state.db = create_db(metas)

# 🔹 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("창조과학 RAG 챗봇 🦖")

# 🔹 이전 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 🔹 사용자 입력 처리
if prompt := st.chat_input("창조과학 관련 질문해주세요 :)"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 🔹 RAG 답변
    response = generate(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})