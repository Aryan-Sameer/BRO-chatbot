import os
import streamlit as st
from llm_memory import rebuild_database, DATA_PATH

def main():
    st.title("📂 Admin Panel - Manage PDF Knowledge Base")

    st.sidebar.header("Actions")
    action = st.sidebar.radio("Select Action", ["Add PDFs", "Remove PDF", "Rebuild Database"])

    # Add PDFs
    if action == "Add PDFs":
        uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DATA_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            if st.button("Process and Add to Database"):
                with st.spinner("Rebuilding database with new PDFs..."):
                    rebuild_database()
                st.success("✅ PDFs added and database rebuilt!")

    # Remove PDFs
    elif action == "Remove PDF":
        if not os.path.exists(DATA_PATH):
            st.warning("No PDFs found in data/")
        else:
            pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
            if pdf_files:
                pdf_to_remove = st.selectbox("Select PDF to remove", pdf_files)

                if st.button("Remove PDF"):
                    with st.spinner(f"Removing {pdf_to_remove} and rebuilding database..."):
                        os.remove(os.path.join(DATA_PATH, pdf_to_remove))
                        try:
                            rebuild_database()
                            st.success(f"✅ {pdf_to_remove} removed and database rebuilt!")
                        except ValueError:
                            st.warning("⚠️ Database empty after removal.")
            else:
                st.info("No PDFs available in data/")

    # Rebuild entire database
    elif action == "Rebuild Database":
        if st.button("Rebuild Database"):
            with st.spinner("Rebuilding FAISS database from all PDFs..."):
                try:
                    rebuild_database()
                    st.success("✅ Database rebuilt successfully!")
                except ValueError:
                    st.error("❌ No PDFs found in data/. Please upload files first.")

if __name__ == "__main__":
    main()
