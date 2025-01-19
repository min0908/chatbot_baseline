import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from setup.bk_logging import langsmith
from setup.st_function import print_messages, add_message
from RAG.retriever import faiss_retriever
from RAG.chain import create_chain
from langchain_core.prompts import load_prompt
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


# API key불러오기
load_dotenv()

# 프로젝트 이름
langsmith("Chatbot_Baseline")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")  # . 은 숨김폴더처리

# 현재 페이지 이름 설정
current_page = "Chatbot_Baseline"

# 상태 초기화 로직
if "current_page" not in st.session_state:
    st.session_state["current_page"] = current_page

if st.session_state["current_page"] != current_page:
    # 페이지 변경 시 모든 상태 초기화
    st.session_state.clear()
    st.session_state["current_page"] = current_page

# title
st.title("챗봇상담 💬")


if "messages" not in st.session_state:
    # 대화내용을 저장
    st.session_state["messages"] = []

# 이전 대화 내용 기억
if "store" not in st.session_state:
    st.session_state["store"] = {}

# 사이드 바 생성
with st.sidebar:
    # 초기화버튼 생성
    clear_btn = st.button("대화내용 초기화")

# 벡터스토어 파일 경로
vectorstore_path = "./document"
st.session_state["retriever"] = faiss_retriever(vectorstore_path)

# 기본 RAG prompt
loaded_prompt = load_prompt("./prompts/basic.yaml", encoding="utf8")

prompt_template = loaded_prompt.template
prompt = PromptTemplate.from_template(prompt_template)
chain = create_chain(
    prompt,
    st.session_state["retriever"],
    temperature=0.5,
    model_name="gpt-4o-mini",
)
st.session_state["chain"] = chain

# 초기화버튼
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 경고 메세지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 사용자 입력
if user_input:
    # chain을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        st.chat_message("user").write(user_input)

        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간 - 여기에 토큰을 스트리밍 출력
            container = st.empty()

            ai_answer = ""

            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)

    else:
        warning_msg.error("모드를 선택해주세요.")
