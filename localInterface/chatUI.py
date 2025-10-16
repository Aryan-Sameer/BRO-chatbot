import streamlit as st
from sync_pdfs import sync_and_rebuild
from connect_memory_with_llm import get_qa_chain

st.set_page_config(page_title="VNR VJIET Assistant", page_icon="🤖")

st.title("Welcome to VNR VJIET")
st.write("I am your AI assistant, here to help you with your queries about VNR VJIET.")

# Sidebar Sync button
st.sidebar.header("Admin Controls")
if st.sidebar.button("🔄 Update latest data"):
    with st.spinner("Rebuilding database..."):
        try:
            sync_and_rebuild()
            st.sidebar.success("✅ Database updated.")
        except Exception as e:
            st.sidebar.error(f"❌ Sync failed: {e}")

# Chat session messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# User input
prompt = st.chat_input("What is your query?")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        with st.spinner("Thinking..."):
            qa_chain = get_qa_chain()
            response = qa_chain(prompt)
            result = response["result"] if isinstance(response, dict) else response
            
            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
            st.session_state.chat_history.append((prompt, result))

    except Exception as e:
        st.error(f"An error occurred while processing your query: {e}")
