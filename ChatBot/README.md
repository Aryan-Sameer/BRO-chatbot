# AI-Powered College Chatbot (RAG + LangChain + FAISS + Ollama Phi-3)

## Overview
An advanced AI-powered institutional assistant built using a Retrieval-Augmented Generation (RAG) pipeline.<br> The chatbot intelligently answers queries based on college documents uploaded by admins, such as PDFs, Word files, PPTs, Excel sheets, and text files.

It Features:

- 🔍 RAG Pipeline (LangChain + FAISS)
- 🧠 Local LLM inference via Ollama Phi-3 (offline capable)
- 📄 Supabase Storage for document management
- 🎛️ Admin Panel for uploading & managing knowledge base
- 🔊 Voice Input and Output (speech_recognition + pyttsx3)
- ⚡ Streamlit UI
- 📎 Multi-format document parsing (PDF, DOCX, PPTX, TXT, CSV, XLSX)
- 🚀 Optimized for Jetson Orin Nano (CUDA-accelerated)

---

## Features
### 1. Local LLM (Ollama Φ-3)
- Runs fully offline
- No API keys or cloud required
- Lightweight and efficient
- Perfect for edge devices like Jetson Orin Nano

### 2. Retrieval-Augmented Generation (RAG)
- Uses FAISS vector database for fast semantic search
- Only answers using your uploaded documents
- If info is missing → responds “I don’t know” (as per custom prompt rules)

### 3. Supabase Knowledge Storage
- Admin can upload documents (PDF, Excel, Word, PPT, TXT)
→ Bot automatically rebuilds the vector DB
→ New knowledge becomes instantly available

### 4. Voice Interaction
- speech_recognition for speech recognition
- pyttsx3 for offline speech synthesis
- Works without internet
- User can ask questions or hear responses via audio

### 5. Streamlit Web Interface
- Clean and simple UI
- Chat-style interface
- Admin panel integrated into the same UI
- Real-time response streaming

---

## Tech Stack
| Component      | Technology                                    |
|----------------|-----------------------------------------------|
| LLM            | Ollama Phi-3                                  |
| Framework      | LangChain                                     |
| Embeddings     | Sentence Transformers                         |
| Vector DB      | FAISS                                         |
| Storage        | Supabase                                      |
| UI             | Streamlit                                     |
| Speech-to-Text | speech_recognition                            |
| Text-to-Speech | pyttsx3                                       |
| File Parsing   | PyPDFLoader, python-docx, python-pptx, pandas |

---

## How It Works (Pipeline Overview)
### 1. Admin uploads files (PDF, Word, Excel, PPT, TXT)
```
Files go to Supabase
↓
Files are downloaded locally
↓
System extracts text using specialized parsers
↓
Text is converted into meaning-based embeddings
↓
FAISS index is rebuilt
```
### 2. User asks a question
```
Query → Embedded → FAISS retrieves relevant text pieces
↓
Context is injected into the system prompt
↓
Phi-3 generates the final response
```

---

## Installation & Setup
### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
Create a `.env` file:
```
SUPABASE_URL=your_supabase_bucket_url
SUPABASE_KEY=your_supabase_key
ADMIN_PASS=secret_key_for_admin_access
```

### 3. Install Ollama
[Ollama Download](https://ollama.com/download)

Then pull Phi-3:
```
ollama pull phi3
```

### 4. Run the chatbot
To run the chatbot
```
cd localInterface
streamlit run chatUI.py
```
To run the admin app
```
cd adminApp
streamlit run app.py
```
---

## Project Structure
```
project_root/
│
├── adminApp/
│    ├── admin.py
│    └── requirements.txt
│
├── localInterface/
│    ├── chatUI.py
│    ├── connect_memory_with_llm.py
│    ├── extract_excel_data.py
│    ├── llm_memory.py
│    └── sync_docs.py
│
├── .env
└── requirements.txt
```
---
## Example RAG Behavior
The bot only answers from uploaded documents.

_Who is the HOD of Electricala and Electronics Engineering department?_

✔ If found → Returns exact description<br>
✖ If missing → Responds “I don't know”

## Why It Works on Jetson Orin Nano
ecause Ollama runs efficiently on Jetson using CUDA-accelerated inference, and all components (Phi-3, embeddings, FAISS, Vosk, pyttsx3) operate fully offline.

## Show Your Support
If you like this project, consider giving it a star ⭐ on GitHub!