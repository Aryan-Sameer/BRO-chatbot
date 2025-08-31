from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
import streamlit as st
import os

load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API")

def load_llm():
    llm=ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
        temperature=0.0,
        groq_api_key=GROQ_API_KEY
    )
    return llm

def main():
    st.title("Welcome to VNR VJIET")
    st.write("I'am your AI assistant, to help you with your queries about VNR VJIET.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("What is your query")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
            Use the pieces of information provided in the context to answer the user's question.
            If you dont know the answer, just say that you dont know, dont try to make up an answer. 
            Dont provide anything out of the given context.
            Greet the users in short if they greet you.
            <</SYS>>

            Context: {context}
            Question: {question} [/INST]
            """

        try:
            vector_store = get_vectorstore()
            if vector_store is None:
                st.error("Vector store could not be loaded.")

            qa_chain = RetrievalQA.from_chain_type(
                llm = load_llm(),
                chain_type = "stuff",
                retriever = vector_store.as_retriever(),
                return_source_documents = True,
                chain_type_kwargs = {'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]

            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"An error occurred while setting up the QA chain: {e}")


if __name__ == "__main__":
    main()
