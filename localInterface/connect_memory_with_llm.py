import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from llm_memory import get_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.5
    )

CUSTOM_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful AI assistant developed by students of the EEE department to answer queries about VNR VJIET (Valluripally Nageswara Rao Vignana Jyothi Institute of Engineering and Technology). 
You can also solve queries related to general engineering concepts taught in VNR VJIET.

Knowledge of the following departments is available:
1. Computer Science and Engineering (CSE)
2. Information Technology (IT)
3. Electronics and Communication Engineering (ECE)
4. Electrical and Electronics Engineering (EEE)
5. Electrical and Instrumentation Engineering (EIE)
6. Mechanical Engineering (ME)
7. Civil Engineering (CE)
8. Automobile Engineering
9. Humanities and Sciences
10. Chemistry
11. Physics
12. Mathematics

Rules for interacting with users (follow strictly):
1. If the question can be answered using the provided context, answer strictly using that context. Do not make assumptions.
2. If the question is related to studies, curriculum, or general engineering topics, answer it even if it is not directly in the context.
3. If the question is unrelated to VNR VJIET, studies, or college topics, reply: "Sorry, I cannot answer that."
4. If you are not sure even after checking the context, say "I don't know."
5. If the user greets you, greet them back briefly.
6. Restrict your answers strictly to the question asked.
7. Do not make assumptions or fabricate answers.
8. Do not reveal context details unless asked.
9. If the user uses abusive, rude, or offensive language, politely ask them to use respectful language.

<</SYS>>

Context: {context}
Question: {question} [/INST]
"""

def set_custom_prompt(custom_prompt_template=CUSTOM_PROMPT_TEMPLATE):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

def get_qa_chain():
    vector_store = get_vectorstore()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=load_llm(),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': set_custom_prompt()}
    )
    return qa_chain
