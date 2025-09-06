import streamlit as st
from connect_memory_with_llm import get_qa_chain

def main():
    st.title("Welcome to VNR VJIET")
    st.write("I am your AI assistant, here to help you with your queries about VNR VJIET.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    prompt = st.chat_input("What is your query")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            with st.spinner("Thinking..."):
                qa_chain = get_qa_chain()
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]

            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"An error occurred while setting up the QA chain: {e}")

if __name__ == "__main__":
    main()
