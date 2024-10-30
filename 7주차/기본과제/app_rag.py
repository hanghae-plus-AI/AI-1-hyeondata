import streamlit as st
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

import os

# load .env
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
def load_rag():
    loader = WebBaseLoader(
    web_paths=("https://spartacodingclub.kr/blog/all-in-challenge_winner",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("editedContent")
            )
        ),
    )   
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000, 
    chunk_overlap=1000
    )
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(model="text-embedding-3-large")
    )

    return vectorstore.as_retriever()

@st.cache_resource
def init_rag():
    return load_rag()
retriever = init_rag()

def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)

def get_ai_response(user_msg):
   prompt = hub.pull("rlm/rag-prompt")
   retrieved_docs = retriever.invoke(user_msg)

   user_prompt = prompt.invoke({"context": format_docs(retrieved_docs), "question": user_msg})
   return llm.invoke(user_prompt)

# Session state initialization
if "messages" not in st.session_state:
   st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.markdown(message["content"])

# Handle new messages
if prompt := st.chat_input("What is up?"):
   # Display user message
   with st.chat_message("user"):
       st.markdown(prompt)
   st.session_state.messages.append({"role": "user", "content": prompt})

   # Get and display AI response
   with st.chat_message("assistant"):
       response = get_ai_response(prompt)
       st.markdown(response.content)
   
   st.session_state.messages.append({
       "role": "assistant",
       "content": response.content
   })