import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

def init_session_state():
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[".", ". ", "\n\n"],
        chunk_size=10,
        chunk_overlap=0
    )
    return text_splitter.split_text(raw_text)

def get_vectorstore(chunks, k):
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": "test.pdf"}
        ) for chunk in chunks
    ]
    
    database = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        collection_name='pdf',
        persist_directory='./chroma_pdf'
    )
    return database.as_retriever(search_kwargs={"k": k})

def get_ai_response(retriever, user_msg):
    prompt = hub.pull("rlm/rag-prompt")
    retrieved_docs = retriever.invoke(user_msg)
    formatted_docs = "\n\n".join(doc.page_content for doc in retrieved_docs)
    user_prompt = prompt.invoke({
        "context": formatted_docs, 
        "question": user_msg
    })
    return llm.invoke(user_prompt)

def main():
    load_dotenv()
    init_session_state()
    
    st.set_page_config(page_title="메뉴얼", page_icon=":book:")
    st.title(":book: 메뉴얼 챗봇")
    st.caption("메뉴얼에 관한 질문을 입력해주세요.")

    # 사이드바 설정
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDF here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process") and docs:
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(docs)
                text_chunks = get_chunks(raw_text)
                st.session_state.retriever = get_vectorstore(text_chunks, k=20)
                st.session_state.processed = True
                st.success("Documents processed successfully!")

    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 사용자 입력 처리
    if user_question := st.chat_input("질문을 입력해주세요."):
        # 문서가 처리되었는지 확인
        if not st.session_state.processed:
            st.error("Please upload and process documents first!")
            return

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = get_ai_response(st.session_state.retriever, user_question)
                st.write(response.content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.content}
                )

if __name__ == "__main__":
    main()