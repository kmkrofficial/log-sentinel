import streamlit as st
import queue

class GlobalState:
    _instance = None

    @staticmethod
    def get_instance():
        if 'is_training' not in st.session_state:
            st.session_state.is_training = False
        if 'train_queue' not in st.session_state:
            st.session_state.train_queue = None
        
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []
        if 'metrics' not in st.session_state:
            st.session_state.metrics = {}
        if 'progress' not in st.session_state:
            st.session_state.progress = 0.0
        if 'status' not in st.session_state:
            st.session_state.status = "Idle"

        if GlobalState._instance is None:
            GlobalState._instance = GlobalState()
        return GlobalState._instance

    def __init__(self):
        pass

    @property
    def is_training(self):
        return st.session_state.is_training

    @property
    def queue(self):
        return st.session_state.train_queue

    def set_train_state(self, is_training, queue):
        st.session_state.is_training = is_training
        st.session_state.train_queue = queue