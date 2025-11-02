import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="LogSentinel",
    page_icon="🛠️",
    layout="wide"
)

st.switch_page("pages/Train_and_Evaluate.py")