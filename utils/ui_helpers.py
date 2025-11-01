import streamlit as st
import pandas as pd
from pathlib import Path

VRAM_WARNING_THRESHOLD = 10
RAM_WARNING_THRESHOLD = 12

def get_dataset_options(data_dir: Path):
    if not data_dir.exists():
        return []
    return [d.name for d in data_dir.iterdir() if d.is_dir() and (d / 'train.csv').exists()]

def render_progress_bar(placeholder):
    progress = st.session_state.get('progress', 0.0)
    status = st.session_state.get('status', 'Idle')
    placeholder.progress(progress, text=status)

def render_metrics(placeholder):
    metrics = st.session_state.get('metrics', {})
    if metrics:
        placeholder.json(metrics)

def render_logs(placeholder):
    logs = st.session_state.get('log_messages', [])
    if logs:
        with placeholder.container(height=400):
            for log in logs:
                st.text(log)