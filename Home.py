import streamlit as st

st.title("Buddy Robot On-campus")

hide_streamlit_style = """
        <style>
            /* Hide main menu (the top-right dots) */
            #MainMenu {display: none;}

            /* Hide deploy button */
            .st-emotion-cache-zq5wmm {display: none;}

            /* remove padding on top */
            .block-container {padding-top: 1rem;}

            /* reduce padding on bottom */
            .st-emotion-cache-139wi93 {padding-bottom: 2rem;}
        </style>
    """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
