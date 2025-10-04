import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def rebuild_database():
    """Rebuild FAISS index from all PDFs in data/"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        raise ValueError("No PDF documents found in data/")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    embedding_model = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # force CPU
    )

    db = FAISS.from_documents(text_chunks, embedding_model)

    if not os.path.exists("vectorstore"):
        os.makedirs("vectorstore")

    db.save_local(DB_FAISS_PATH)
    return True
