import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

GROQ_MODEL="openai/gpt-oss-20b"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(model_name, api_key):
    return ChatGroq(
        model=model_name,
        temperature=0.5,
        max_tokens=512,
        api_key=api_key,
    )


@st.cache_resource
def get_qa_chain(api_key):
    vectorstore=get_vectorstore()
    return RetrievalQA.from_chain_type(
        llm=load_llm(GROQ_MODEL, api_key),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY is not set. Add it to the .env file.")
            return

        try:
            qa_chain=get_qa_chain(GROQ_API_KEY)

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            sources="\n".join(
                f"- {os.path.basename(str(d.metadata.get('source', 'unknown')))}, page {d.metadata.get('page', '?')}: {d.page_content[:150].strip()!r}"
                for d in response["source_documents"]
            )
            result_to_show=result+"\n\n**Source Docs:**\n"+sources
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()