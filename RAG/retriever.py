from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def faiss_retriever(vectorstore_path):
    # 로컬 벡터스토어 불러오기
    vectorstore = FAISS.load_local(vectorstore_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever()

    return retriever