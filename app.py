import streamlit as st

# Set the global page configuration for the entire app
st.set_page_config(
    page_title="LogSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Hide Streamlit default footer and hamburger menu ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Immediately switch to the landing page.
# The user will not see this page, it's just a host.
st.switch_page("pages/0_Landing.py")