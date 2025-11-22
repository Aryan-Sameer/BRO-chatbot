import os
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from llm_memory import get_vectorstore
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def get_qa_chain():
    """Create a retrieval-based QA chain that works with all document types using Ollama locally."""

    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # --- Custom Prompt ---
    CUSTOM_PROMPT_TEMPLATE = """You are a helpful AI assistant developed by students of the EEE department at VNR VJIET.

    Answer the user's question based only on the provided context below. 
    If the answer is not found in the context, respond only with:
    "I don't know."

    Rules:
    0. Keep answers concise and to the point. Do not include greetings or explanations.
    1. Always greet the user when they greet you in short without saying anything extra.
    2. Use only the given context to answer.
    3. If partial information is available, mention that and give the best possible answer.
    4. Do not say anything else unless asked.
    5. Do not use bullet points — use numbers if multiple points exist.
    6. Do not explain what you are doing.

    ---
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=CUSTOM_PROMPT_TEMPLATE
    )

    # --- Use Ollama model (local) ---
    llm = Ollama(model="phi3", temperature=0.3)

    # --- Create Retrieval QA chain ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # --- Simple wrapper ---
    def run(query: str):
        query = query.strip()
        return qa_chain.invoke({"query": query})

    return run
