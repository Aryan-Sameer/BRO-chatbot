import os
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Environment vars (set these in your deployment platform's secrets)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_PASSWORD = os.getenv("ADMIN_PASS", "change_me")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("SUPABASE_URL and SUPABASE_KEY must be set as environment variables.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET = "pdfs"
storage = supabase.storage.from_(BUCKET)

st.title("📂 College Documents Admin")

# simple auth
password = st.sidebar.text_input("Admin password", type="password")
if password != ADMIN_PASSWORD:
    st.sidebar.warning("Enter admin password to manage documents.")
    st.stop()

action = st.sidebar.radio("Action", ["Upload documents", "List / Delete documents"])

if action == "Upload documents":
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Upload to database"):
            for u in uploaded_files:
                file_bytes = u.read()
                fname = u.name
                # upload to bucket (upsert True behavior if name exists)
                try:
                    res = storage.upload(fname, file_bytes)
                    # create public url
                    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{fname}"
                    # insert metadata (optional)
                    uploaded_at = datetime.utcnow().isoformat()
                    supabase.table("pdfs").upsert({
                        "filename": fname,
                        "url": public_url,
                        "uploaded_at": uploaded_at
                    }).execute()
                    st.success(f"Uploaded {fname}")
                except Exception as e:
                    st.error(f"Failed to upload {fname}: {e}")

    else:
        st.info("Drag-and-drop or select documents to upload.")

elif action == "List / Delete documents":
    st.subheader("Files in bucket")
    try:
        files = supabase.storage.from_(BUCKET).list()
        if not files:
            st.info("No files in bucket.")
        else:
            for f in files:
                name = f["name"]
                public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{name}"
                cols = st.columns([6,1])
                cols[0].write(f"**{name}**")
                if cols[1].button("Delete", key=name):
                    # delete from storage
                    supabase.storage.from_(BUCKET).remove([name])
                    # delete metadata row too (optional)
                    supabase.table("pdfs").delete().eq("filename", name).execute()
                    st.rerun()
            st.write("---")
    except Exception as e:
        st.error(f"Failed to list files: {e}")
