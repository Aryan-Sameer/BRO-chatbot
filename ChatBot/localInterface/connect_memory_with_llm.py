import os
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from ChatBot.localInterface.llm_memory import get_vectorstore
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def get_qa_chain():
    """Create a retrieval-based QA chain that works with all document types using Ollama locally."""

    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    # --- Custom Prompt ---
    CUSTOM_PROMPT_TEMPLATE = """You are a helpful AI assistant developed by students of the EEE department at VNR VJIET to assist with academic and technical queries.

    Answer the user's questions in short, limiting to the query based only on the provided context below.
    The context contains information from multiple sources including PDFs, Word documents, PowerPoint presentations, Excel sheets, and text files.
    Greet the user when they greet you.

    Answer the question if it is related to engineering branches:
    - computer science
    - mechanical
    - civil
    - electrical
    - electronics
    - auto mobile
    - chemistry
    - physics
    - mathematics

    If the answer is not found in the context or not related to engineering:
    "I don't know."

    If the query is from any different category like food, media and entertainment, sports, respond only with:
    "I don't know."

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
