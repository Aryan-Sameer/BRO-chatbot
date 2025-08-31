from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv(find_dotenv())

# Setting up LLM

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def load_llm():
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.0-flash",
        google_api_key = GEMINI_API_KEY,
        temperature = 0.7
    )
    return llm

# Create a prompt template
CUSTOM_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
    Use the pieces of information provided in the context to answer the user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    Dont provide anything out of the given context.
    <</SYS>>
    
    Context: {context}
    Question: {question} [/INST]
    """

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=["context", "question"]
    )
    return prompt

# Load the FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization = True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(),
    chain_type = "stuff",
    retriever = db.as_retriever(),
    return_source_documents = True,
    chain_type_kwargs = {'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Invoke the chain
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
