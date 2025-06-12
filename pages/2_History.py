import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add root directory to path to import local modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.database_manager import DatabaseManager
from utils.report_generator import create_report
from config import DB_PATH

st.set_page_config(page_title="Run History", layout="wide")
st.title("📜 Run History")
st.markdown("---")

db_manager = DatabaseManager(DB_PATH)
runs_df = pd.DataFrame(db_manager.get_all_runs())

# --- Display All Runs with View Buttons ---
st.header("All Training Runs")

# Table Headers
cols = st.columns((2, 1, 2, 2, 2, 1, 1))
headers = ["Run ID", "Status", "Model", "Dataset", "Start Time", "Duration(s)", "Actions"]
for col, header in zip(cols, headers):
    col.markdown(f"**{header}**")

if runs_df.empty:
    st.info("No training runs found in the database yet.")
else:
    # Reverse the order for chronological display
    runs_df = runs_df.iloc[::-1]
    
    for _, row in runs_df.iterrows():
        run_id = row['run_id']
        start_time_dt = pd.to_datetime(row['start_time'], unit='s')
        end_time_dt = pd.to_datetime(row['end_time'], unit='s', errors='coerce')
        duration = (end_time_dt - start_time_dt).total_seconds() if pd.notna(end_time_dt) else 0

        col1, col2, col3, col4, col5, col6, col7 = st.columns((2, 1, 2, 2, 2, 1, 1))
        with col1:
            st.text(run_id)
        with col2:
            st.text(row['status'])
        with col3:
            st.text(row['model_name'])
        with col4:
            st.text(row['dataset_name'])
        with col5:
            st.text(start_time_dt.strftime('%Y-%m-%d %H:%M:%S'))
        with col6:
            st.text(f"{duration:.0f}")
        with col7:
            if st.button("View", key=f"view_{run_id}"):
                st.session_state['selected_run_id'] = run_id
        st.markdown("---", unsafe_allow_html=True)


# --- Details Section ---
if 'selected_run_id' in st.session_state:
    selected_run_id = st.session_state['selected_run_id']
    
    st.header(f"Details for Run: `{selected_run_id}`")
    details = db_manager.get_run_details(selected_run_id)

    if not details:
        st.error(f"Could not retrieve details for Run ID: {selected_run_id}")
    else:
        run_info = details.get('run_info', {})
        hyperparams = details.get('hyperparameters', {})
        perf_metrics = details.get('performance_metrics', {})
        resource_metrics = details.get('resource_metrics', {})
        
        raw_report_path = run_info.get('report_path')
        report_path = Path(raw_report_path) if raw_report_path else None
        
        # Action Buttons for the selected run
        action_cols = st.columns(4)
        with action_cols[0]:
            if st.button("🔄 Rerun this Configuration", key=f"rerun_{selected_run_id}"):
                st.session_state['rerun_config'] = {
                    "model_name": run_info.get('model_name'),
                    "dataset_name": run_info.get('dataset_name'),
                    "hyperparameters": hyperparams
                }
                st.success("Configuration loaded. Navigate to the 'Train & Evaluate' page to start the new run.")

        with action_cols[1]:
            disable_pdf = not (report_path and report_path.is_dir())
            if st.button("📄 Generate PDF Report", key=f"pdf_{selected_run_id}", disabled=disable_pdf):
                pdf_path_str = create_report(details, report_path)
                with open(pdf_path_str, "rb") as pdf_file:
                    st.download_button("Download Report PDF", pdf_file, f"LogSentinel_Report_{selected_run_id[:8]}.pdf", "application/octet-stream")

        # Tabs for detailed info
        tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "⚙️ Hyperparameters", "📈 Resource Usage"])
        
        with tab1:
            st.json(perf_metrics, expanded=True)
            if report_path and report_path.is_dir():
                st.image(str(report_path / 'confusion_matrix.png'), caption='Confusion Matrix')
                st.image(str(report_path / 'overall_metrics.png'), caption='Overall Metrics')

        with tab2:
            st.json(hyperparams, expanded=True)

        with tab3:
            st.json(resource_metrics.get('summary', {}))
            if report_path and report_path.is_dir():
                st.image(str(report_path / 'resource_usage.png'), caption='Resource Usage Over Time')

db_manager.close()