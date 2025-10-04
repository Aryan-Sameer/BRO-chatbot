# sync_pdfs.py
import os
import requests
from dotenv import load_dotenv
from supabase import create_client
from llm_memory import rebuild_database   # your file name earlier: llm_memory.py

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # anon key is enough if bucket is public
BUCKET = "pdfs"
LOCAL_DATA = "data/"

os.makedirs(LOCAL_DATA, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def list_remote_files():
    items = supabase.storage().from_(BUCKET).list()
    # items is a list of dicts with at least "name"
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
    to_download = remote - local
    for name in to_download:
        print("Downloading new:", name)
        download_file(name)

    # Remove deleted files locally
    to_remove = local - remote
    for name in to_remove:
        print("Removing local file (deleted remotely):", name)
        os.remove(os.path.join(LOCAL_DATA, name))

    # If anything changed, rebuild FAISS
    if to_download or to_remove or not os.path.exists("./vectorstore/db_faiss/index.faiss"):
        print("Changes detected, rebuilding FAISS DB...")
        try:
            rebuild_database()
            print("Rebuild done.")
        except Exception as e:
            print("Rebuild failed:", e)
    else:
        print("No changes. FAISS is up-to-date.")
