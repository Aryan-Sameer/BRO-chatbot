import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from llm_memory import get_vectorstore
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def get_qa_chain():
    """Create a retrieval-based QA chain that works with all document types."""

    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )

    # --- Prompt Template ---
    CUSTOM_PROMPT_TEMPLATE = """You are a helpful AI assistant developed by students of the EEE department at VNR VJIET.

    Answer the user's question based on the provided context. The context comes from various sources such as PDFs, Word documents, Excel sheets, PPTs, or text files uploaded by admins.

    Rules:
    1. Use only the given context to answer. 
    2. If only partial information is available, mention that and provide the best possible answer.
    3. If the context does not have relevant data, respond: "Sorry, I don't have enough information in the provided data."
    4. Keep your answers clear, precise, and directly relevant to the question.

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

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3  # lower = more factual
    )

    # --- Create Retrieval QA chain ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # --- Simple wrapper ---
    def run(query: str):
        """Run a query against the knowledge base."""
        query = query.strip()
        return qa_chain.invoke({"query": query})

    return run
