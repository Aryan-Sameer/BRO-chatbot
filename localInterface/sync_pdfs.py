import os
import requests
from dotenv import load_dotenv
from supabase import create_client
from llm_memory import rebuild_database

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET = "pdfs"
LOCAL_DATA = "data/"

os.makedirs(LOCAL_DATA, exist_ok=True)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def list_remote_files():
    items = supabase.storage.from_(BUCKET).list()
    return [it["name"] for it in items]

def download_file(name):
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{name}"
    resp = requests.get(public_url, stream=True, timeout=60)
    resp.raise_for_status()
    local_path = os.path.join(LOCAL_DATA, name)
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Downloaded:", name)
    return local_path

def sync_and_rebuild():
    remote = set(list_remote_files())
    local = set([f for f in os.listdir(LOCAL_DATA) if f.lower().endswith(".pdf")])

    # Download new files
    for name in remote - local:
        print("Downloading new:", name)
        download_file(name)

    # Remove deleted files locally
    for name in local - remote:
        print("Removing local file (deleted remotely):", name)
        os.remove(os.path.join(LOCAL_DATA, name))

    # Rebuild FAISS if anything changed
    if remote != local or not os.path.exists("vectorstore/db_faiss/index.faiss"):
        print("Changes detected, rebuilding FAISS DB...")
        rebuild_database()
        print("Rebuild done.")
    else:
        print("No changes. FAISS is up-to-date.")
