from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import pickle

# 2. 문서 로딩 함수 (pickle로 저장된 문서 리스트에서 로드)
def load_documents(vectorstore_path):
    document_paths = os.path.join(vectorstore_path, "documents.pkl")
    with open(document_paths, 'rb') as f:
        documents = pickle.load(f)
    return documents

# 3. 검색된 문서들 포맷팅 함수
def format_docs(docs):
    result_str = "\n".join([f"<document>{doc.page_content}<document>" for doc in docs])
    return f"<context>{result_str}<context>"


# 체인 생성
def create_chain(prompt, retriever, temperature, model_name="gpt-4o-mini"):

    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
