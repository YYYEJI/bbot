# Bebot UI
import streamlit as st
import json
from bbot import create_db, generate


# # DB 한 번만 생성 
# if "db_initialized" not in st.session_state:
#     create_db("./extracted_texts")
#     st.session_state.db_initialized = True


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