import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain import hub
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM


load_dotenv()
st.set_page_config(page_title="메뉴얼", page_icon=":book:")



@st.cache_resource
def model_init():
    chat_model = HuggingFacePipeline.from_model_id(
        model_id='google/gemma-2-2b-it',
        task='text-generation',
        pipeline_kwargs=dict(
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.03
        ),
    )
    return chat_model

llm = model_init()

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
    # documents = []
    # for i, text in enumerate(chunks):
    #     # 빈 텍스트 건너뛰기
    #     if text.strip():
    #         doc = Document(
    #             page_content=text,
    #             metadata={"source": "test.pdf"}
    #             # metadata={"source": f"page_{i+1}"}  # 페이지 번호 메타데이터 추가
    #         )
    #         documents.append(doc)
    
    
    database = Chroma.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct'),
        collection_name='pdf',
        persist_directory='./chroma_pdf'
    )
    return database.as_retriever(search_kwargs={"k": k})

def get_ai_response(retriever, user_msg):
    
    gemma_template = """<bos><start_of_turn>user
    Context: {context}

    Question: {question}

    Please answer based on the given context. If you can't find the answer, say "I don't know".
    **Please answer in Korean**
    <end_of_turn>
    <start_of_turn>model
    """

    retrieved_docs = retriever.invoke(user_msg)
    print(retrieved_docs)
    formatted_docs = "\n\n".join(doc.page_content for doc in retrieved_docs)
    user_prompt = PromptTemplate(
    template=gemma_template,
    input_variables=["context", "question"]
    )
    formatted_prompt = gemma_template.format(
        context=formatted_docs, 
        question=user_msg
    )
    return llm.invoke(formatted_prompt)

def main():
    init_session_state()
    

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
                print(response)
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

if __name__ == "__main__":
    main()