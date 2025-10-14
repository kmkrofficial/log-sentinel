import streamlit as st
import multiprocessing

def run_streamlit():
    st.set_page_config(
        page_title="LogSentinel",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.switch_page("pages/0_Landing.py")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_streamlit()