import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from engine.inference_controller import InferenceController
from utils.database_manager import DatabaseManager
from config import DB_PATH

st.set_page_config(page_title="Inference", layout="wide")
st.title("🔍 Perform Inference")
st.markdown("---")

db_manager = DatabaseManager(DB_PATH)

st.header("1. Select a Trained Model")
runs_df = pd.DataFrame(db_manager.get_all_runs())

if runs_df.empty or 'status' not in runs_df.columns or runs_df[runs_df['status'] == 'COMPLETED'].empty:
    st.warning("No successfully trained models available. Please complete a training run first.")
else:
    completed_runs = runs_df[runs_df['status'] == 'COMPLETED'].copy()
    completed_runs['display'] = completed_runs.apply(
        lambda row: f"{row['model_name']} on {row['dataset_name']} (Run ID: {row['run_id'][:8]}...)",
        axis=1
    )
    selected_display = st.selectbox("Choose a model from a completed run:", completed_runs['display'])
    selected_run_id = completed_runs[completed_runs['display'] == selected_display]['run_id'].iloc[0]

    st.info(f"Selected model from Run ID: `{selected_run_id}`")
    st.markdown("---")

    st.header("2. Provide Input for Prediction")
    tab1, tab2 = st.tabs(["Single Sequence Prediction", "Batch Prediction from CSV"])

    with tab1:
        st.subheader("Analyze a single log sequence")
        sequence_input = st.text_area(
            "Paste the log sequence here (use ' ;-; ' as a separator):",
            height=150,
            placeholder="e.g., Receiving block blk_... src: /... ;-; BLOCK* NameSystem.allocateBlock: /..."
        )
        
        if st.button("Analyze Sequence"):
            if not sequence_input.strip():
                st.warning("Please enter a log sequence.")
            else:
                try:
                    with st.spinner("Loading model and performing inference..."):
                        controller = InferenceController(run_id=selected_run_id)
                        result = controller.predict_single(sequence_input)
                    
                    if result['prediction'] == "Anomalous":
                        st.error(f"**Prediction: {result['prediction']}** (Confidence: {result['confidence']:.2%})")
                    else:
                        st.success(f"**Prediction: {result['prediction']}** (Confidence: {result['confidence']:.2%})")
                
                except Exception as e:
                    print(f"Inference Error (Single): {e}") # CONSOLE LOGGING
                    st.error(f"An error occurred: {e}")

    with tab2:
        st.subheader("Upload a CSV file with a 'Content' column")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                if 'Content' not in df_input.columns:
                    st.error("The uploaded CSV must contain a 'Content' column.")
                else:
                    st.write("Preview of uploaded data:", df_input.head())
                    
                    if st.button("Run Batch Inference"):
                        with st.spinner("Loading model and processing batch..."):
                            controller = InferenceController(run_id=selected_run_id)
                            df_result = controller.predict_batch(df_input)
                        
                        st.success("Batch inference complete!")
                        st.dataframe(df_result)
                        
                        csv_result = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Results as CSV", csv_result, f"predictions_{selected_run_id[:8]}.csv", "text/csv")

            except Exception as e:
                print(f"Inference Error (Batch): {e}") # CONSOLE LOGGING
                st.error(f"Failed to process file: {e}")

db_manager.close()