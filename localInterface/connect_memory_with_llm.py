from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables
load_dotenv(find_dotenv())

# Load LLM
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7
    )

# Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful AI assistant developed by students of EEE (Electrical and Electronics Engineering)department to address the queries related to VNR VJIET.
Use the pieces of information provided in the context to answer the user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context.
Greet the users in short if they greet you.
<</SYS>>

Context: {context}
Question: {question} [/INST]
"""

def set_custom_prompt(custom_prompt_template=CUSTOM_PROMPT_TEMPLATE):
    return PromptTemplate(
        template=custom_prompt_template, 
        input_variables=["context", "question"]
    )

# Load Vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"

def get_vectorstore():
    embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Build QA Chain
def get_qa_chain():
    vector_store = get_vectorstore()
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()}
    )
    return qa_chain
