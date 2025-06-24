import streamlit as st
import pandas as pd
import json
import time
import math
from pathlib import Path

from utils.database_manager import DatabaseManager
from config import DB_PATH

st.set_page_config(page_title="Run History", page_icon="📜", layout="wide")

def format_duration(seconds):
    if seconds is None: return "N/A"
    seconds = int(seconds); hours, remainder = divmod(seconds, 3600); minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# --- FIX: New robust function to display a metric block ---
def display_metric_block(title, metrics_data):
    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics_data.get('accuracy', 0):.4f}")
    c2.metric("Precision", f"{metrics_data.get('precision', 0):.4f}")
    c3.metric("Recall", f"{metrics_data.get('recall', 0):.4f}")
    c4.metric("F1-Score", f"{metrics_data.get('f1_score', 0):.4f}")
    st.divider()

st.title("📜 Run History")
st.write("Review details, metrics, and reports from past training, testing, and inference runs.")

db = DatabaseManager(DB_PATH)

# --- TABS FOR DIFFERENT VIEWS ---
tab1, tab2 = st.tabs(["All Runs", "Trained Model Performance"])

# --- TAB 1: ALL RUNS ---
with tab1:
    all_runs = db.get_all_runs()
    if not all_runs:
        st.info("No runs found in the database. Please start a new run from the other pages.")
    else:
        df_runs = pd.DataFrame(all_runs)
        
        if 'page_number' not in st.session_state: st.session_state.page_number = 0
        if 'selected_run_id_for_view' not in st.session_state: st.session_state.selected_run_id_for_view = None

        with st.expander("🔎 Filter and Search Runs", expanded=True):
            col1, col2 = st.columns(2);
            with col1:
                available_types = sorted(df_runs['run_type'].unique().tolist()); selected_types = st.multiselect("Filter by Run Type", options=available_types, default=available_types)
                available_statuses = sorted(df_runs['status'].unique().tolist()); selected_statuses = st.multiselect("Filter by Status", options=available_statuses, default=available_statuses)
            with col2: search_query = st.text_input("Search by ID, Nickname, Model, or Dataset")
        
        filtered_df = df_runs.copy()
        if selected_types: filtered_df = filtered_df[filtered_df['run_type'].isin(selected_types)]
        if selected_statuses: filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]
        if search_query:
            query = search_query.strip()
            filtered_df = filtered_df[
                filtered_df['run_id'].astype(str).str.contains(query, case=False, na=False) |
                filtered_df['nickname'].str.contains(query, case=False, na=False) |
                filtered_df['model_name'].str.contains(query, case=False, na=False) |
                filtered_df['dataset_name'].str.contains(query, case=False, na=False)
            ]

        st.markdown(f"**{len(filtered_df)} runs found.**")
        ITEMS_PER_PAGE = 10; total_items = len(filtered_df); total_pages = math.ceil(total_items / ITEMS_PER_PAGE) if total_items > 0 else 1
        st.session_state.page_number = min(st.session_state.page_number, total_pages - 1)
        start_idx = st.session_state.page_number * ITEMS_PER_PAGE; end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
        paginated_df = filtered_df.iloc[start_idx:end_idx]

        st.markdown("---")
        header_cols = st.columns([1, 2, 2, 2, 1, 2, 1]); 
        header_cols[0].write("**ID**"); header_cols[1].write("**Nickname**"); header_cols[2].write("**Model**"); header_cols[3].write("**Dataset**"); header_cols[4].write("**Type**"); header_cols[5].write("**Start Time**"); header_cols[6].write("**Actions**")
        
        for _, row in paginated_df.iterrows():
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 2, 2, 2, 1, 2, 1]); 
            c1.markdown(f"`{row['run_id']}`"); c2.write(row['nickname']); c3.write(row['model_name']); c4.write(row['dataset_name']); 
            c5.markdown(f"`{row['run_type']}`")
            c6.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row['start_time'])))
            action_col = c7.columns([1, 1])
            if action_col[0].button("👁️", key=f"view_{row['run_id']}", help="View Details"):
                st.session_state.selected_run_id_for_view = row['run_id']; st.rerun()
            if row['run_type'] == 'Training':
                if action_col[1].button("🔁", key=f"rerun_{row['run_id']}", help="Rerun with this configuration"):
                    run_details_for_rerun = db.get_run_details(row['run_id'])
                    if run_details_for_rerun and run_details_for_rerun.get('hyperparameters'):
                        st.session_state['rerun_config'] = {'model_name': run_details_for_rerun['run_info'].get('model_name'), 'dataset_name': run_details_for_rerun['run_info'].get('dataset_name'), 'hyperparameters': run_details_for_rerun.get('hyperparameters', {})}; st.switch_page("pages/1_Train_and_Evaluate.py")
        
        if total_pages > 1:
            st.markdown("---"); page_cols = st.columns([1, 1, 1])
            if page_cols[0].button("⬅️ Previous", disabled=(st.session_state.page_number == 0)): st.session_state.page_number -= 1; st.rerun()
            page_cols[1].write(f"Page **{st.session_state.page_number + 1}** of **{total_pages}**")
            if page_cols[2].button("Next ➡️", disabled=(st.session_state.page_number >= total_pages - 1)): st.session_state.page_number += 1; st.rerun()

# --- TAB 2: TRAINED MODEL PERFORMANCE ---
with tab2:
    training_runs = db.get_all_runs(run_type='Training')
    if not training_runs:
        st.info("No completed training runs found. Train a model to see its performance here.")
    else:
        perf_data = []
        for run in training_runs:
            if run['status'] == 'COMPLETED':
                details = db.get_run_details(run['run_id'])
                if details and details.get('performance_metrics', {}).get('test', {}).get('overall'):
                    p_metrics = details['performance_metrics']['test']['overall']
                    entry = {
                        'ID': run['run_id'],
                        'Nickname': run['nickname'],
                        'Dataset': run['dataset_name'],
                        'Accuracy': p_metrics.get('accuracy', 0),
                        'Precision': p_metrics.get('precision', 0),
                        'Recall': p_metrics.get('recall', 0),
                        'F1-Score': p_metrics.get('f1_score', 0),
                    }
                    perf_data.append(entry)
        
        if not perf_data:
            st.info("No completed training runs with test metrics found.")
        else:
            st.subheader("Final Test Set Performance")
            df_perf = pd.DataFrame(perf_data)
            st.dataframe(
                df_perf,
                column_config={
                    "ID": st.column_config.NumberColumn(format="%d"),
                    "Accuracy": st.column_config.ProgressColumn(format="%.4f", min_value=0, max_value=1),
                    "Precision": st.column_config.ProgressColumn(format="%.4f", min_value=0, max_value=1),
                    "Recall": st.column_config.ProgressColumn(format="%.4f", min_value=0, max_value=1),
                    "F1-Score": st.column_config.ProgressColumn(format="%.4f", min_value=0, max_value=1),
                },
                use_container_width=True,
                hide_index=True
            )

# --- DETAILS VIEW (COMMON TO BOTH TABS) ---
if st.session_state.selected_run_id_for_view:
    st.markdown("---")
    with st.container(border=True):
        run_details = db.get_run_details(st.session_state.selected_run_id_for_view)
        if run_details:
            info, h_params, p_metrics, r_metrics = run_details.values()
            st.subheader(f"Details for Run: `{info.get('nickname', info.get('run_id'))}`")
            if st.button("Close Details"): st.session_state.selected_run_id_for_view = None; st.rerun()
            duration = info.get('end_time', 0) - info.get('start_time', 0) if info.get('end_time') else None
            
            is_test_run_info = h_params.get('is_test_run', False)
            if is_test_run_info:
                test_perc = h_params.get('test_run_percentage', 0.3) * 100
                st.info(f"This was a **Quick Test Run** using {test_perc:.0f}% of the data.")

            summary_cols = st.columns(4); summary_cols[0].metric("Run Type", info.get('run_type')); summary_cols[1].metric("Status", info.get('status')); summary_cols[2].metric("Start Time", time.strftime('%H:%M:%S', time.localtime(info.get('start_time')))); summary_cols[3].metric("Total Duration", format_duration(duration))
            with st.expander("📊 Performance Metrics", expanded=True):
                if p_metrics:
                    has_displayed_metrics = False
                    
                    # Training Run Metrics
                    if 'validation' in p_metrics and 'overall' in p_metrics['validation']:
                        display_metric_block("Validation Set", p_metrics['validation']['overall'])
                        has_displayed_metrics = True
                    if 'test' in p_metrics and 'overall' in p_metrics['test']:
                        display_metric_block("Test Set", p_metrics['test']['overall'])
                        has_displayed_metrics = True
                    
                    # Standalone Testing/Inference Run Metrics
                    if 'overall' in p_metrics and 'accuracy' in p_metrics['overall']:
                        display_metric_block("Classification Results", p_metrics['overall'])
                        has_displayed_metrics = True
                    
                    # Always display timing info if available
                    if 'overall' in p_metrics and 'total_run_time_sec' in p_metrics['overall']:
                        st.subheader("Processing Speed")
                        timing_metrics = p_metrics['overall']
                        timing_cols = st.columns(2)
                        timing_cols[0].metric("Total Run Time", f"{timing_metrics.get('total_run_time_sec', 0):.2f}s")
                        timing_cols[1].metric("Time / Record", f"{timing_metrics.get('time_per_record_ms', 0):.2f} ms")
                        has_displayed_metrics = True

                    if not has_displayed_metrics:
                        st.warning("No classification or timing metrics available for this run.")
                else:
                     st.warning("No performance metrics available.")

            with st.expander("🛠️ Resource Metrics", expanded=True):
                if r_metrics and 'summary' in r_metrics:
                    summary = r_metrics.get('summary', {}); cpu, ram, gpu = summary.get('cpu', {}), summary.get('ram', {}), summary.get('gpu', {})
                    st.subheader("CPU"); cpu_cols = st.columns(2); cpu_cols[0].metric("Avg CPU Usage", f"{cpu.get('avg_cpu_usage_percent', 0):.2f}%"); cpu_cols[1].metric("P95 CPU Usage", f"{cpu.get('p95_cpu_usage_percent', 0):.2f}%"); st.divider()
                    st.subheader("RAM"); ram_cols = st.columns(3); ram_cols[0].metric("Avg RAM Usage", f"{ram.get('avg_ram_usage_gb', 0):.2f} GB"); ram_cols[1].metric("P95 RAM Usage", f"{ram.get('p95_ram_usage_gb', 0):.2f} GB"); ram_cols[2].metric("Avg RAM %", f"{ram.get('avg_ram_usage_percent', 0):.2f}%", help=f"Total: {ram.get('total_system_ram_gb', 0):.2f} GB")
                    if gpu:
                        st.divider(); st.subheader("GPU"); gpu_cols = st.columns(4); gpu_cols[0].metric("Avg GPU Util", f"{gpu.get('avg_gpu_util_percent', 0):.2f}%"); gpu_cols[1].metric("P95 GPU Util", f"{gpu.get('p95_gpu_util_percent', 0):.2f}%"); gpu_cols[2].metric("Avg Power", f"{gpu.get('avg_power_watts', 0):.2f}W"); gpu_cols[3].metric("P95 Power", f"{gpu.get('p95_power_watts', 0):.2f}W")
                        gpu_mem_cols = st.columns(4); gpu_mem_cols[0].metric("Avg VRAM", f"{gpu.get('avg_gpu_mem_gb', 0):.2f} GB", help=f"Total: {gpu.get('total_gpu_mem_gb', 0):.2f} GB"); gpu_mem_cols[1].metric("P95 VRAM", f"{gpu.get('p95_gpu_mem_gb', 0):.2f} GB"); gpu_mem_cols[2].metric("Avg VRAM %", f"{gpu.get('avg_gpu_mem_percent', 0):.2f}%"); gpu_mem_cols[3].metric("P95 VRAM %", f"{gpu.get('p95_gpu_mem_percent', 0):.2f}%")
                else: st.warning("No resource metrics available.")
            with st.expander("📄 Reports & Visualizations", expanded=True):
                report_path_str = info.get('report_path')
                if report_path_str and Path(report_path_str).exists():
                    report_dir = Path(report_path_str)
                    if report_dir.is_dir():
                        if info.get('run_type') == 'Inference':
                             csv_files = list(report_dir.glob("*.csv"));
                             if csv_files:
                                with open(csv_files[0], "rb") as fp: st.download_button("Download Results CSV", fp, csv_files[0].name, "text/csv")
                        
                        plot_files = {
                            'training_loss.png': "Shows the batch loss over the course of the training run.",
                            'test_confusion_matrix.png': "Compares predicted labels to true labels on the Test set.",
                            'validation_confusion_matrix.png': "Compares predicted labels to true labels on the Validation set.",
                            'test_roc_curve.png': "Plots true positive rate against false positive rate on the Test set.",
                            'validation_roc_curve.png': "Plots true positive rate against false positive rate on the Validation set.",
                            'test_overall_metrics.png': "A bar chart summarizing key metrics on the Test set.",
                            'validation_overall_metrics.png': "A bar chart summarizing key metrics on the Validation set.",
                            'confusion_matrix.png': "Compares predicted labels to true labels.",
                            'roc_curve.png': "Plots true positive rate against false positive rate.",
                            'overall_metrics.png': "A bar chart summarizing key classification metrics.",
                            'cpu_usage.png': "Tracks CPU utilization percentage over the duration of the run.",
                            'ram_usage.png': "Tracks RAM consumption in Gigabytes (GB) over the duration of the run.",
                            'gpu_utilization.png': "Tracks GPU processing utilization percentage over time.",
                            'gpu_memory.png': "Tracks dedicated GPU Memory (VRAM) consumption over time."
                        }
                        
                        image_paths = []
                        for filename, caption in plot_files.items():
                            path = report_dir / filename
                            if path.exists() and path not in [p for p, c in image_paths]: 
                                image_paths.append((path, caption))

                        if image_paths:
                            plot_cols = st.columns(2)
                            for i, (path, caption) in enumerate(image_paths):
                                with plot_cols[i % 2]: st.image(str(path), use_container_width=True, caption=caption)
                        else: st.info("No plot images found in the report directory.")
                else: st.warning("No report or visualizations available for this run.")

db.close()