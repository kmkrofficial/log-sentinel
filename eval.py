# --- START OF FILE eval.py ---

import os
import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
# Import updated model (which now includes classifier)
from model import LogLLM
from customDataset import CustomDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import gc

# --- Configuration ---
max_content_len = 100
max_seq_len = 128
batch_size = 16 # Keep inference batch size reasonable
dataset_name = 'BGL'

# --- Set Correct Test Data Path ---
base_data_dir = r'E:\research-stuff\LogLLM-3b\dataset'
test_data_filename = 'train.csv'
data_path = os.path.join(base_data_dir, test_data_filename)
if not os.path.exists(data_path):
    print(f"WARNING: Test data file '{test_data_filename}' not found at '{data_path}'.")
    train_data_filename = 'train.csv'
    data_path_fallback = os.path.join(base_data_dir, train_data_filename)
    if os.path.exists(data_path_fallback):
        print(f"Falling back to using '{train_data_filename}' for evaluation.")
        data_path = data_path_fallback
    else:
        print(f"ERROR: Neither '{test_data_filename}' nor '{train_data_filename}' found in '{base_data_dir}'.")
        # Exit if no data found
        exit(f"Evaluation cannot proceed without data at {base_data_dir}")
# --- End Path Setting ---

# Paths
Bert_path = r"E:\research-stuff\LogLLM-3b\models\bert-base-uncased"
Llama_path = r"E:\research-stuff\LogLLM-3b\models\Llama-3.2-3B"

ROOT_DIR = Path(__file__).resolve().parent
# --- Point to CLASSIFICATION model path ---
ft_path = os.path.join(ROOT_DIR, r"ft_model_cls_{}".format(dataset_name))
# --- End Path Setting ---

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("--- Evaluation Configuration (Classification Head) ---")
print(f'dataset_name: {dataset_name}')
print(f'batch_size: {batch_size}')
print(f'max_content_len: {max_content_len}')
print(f'max_seq_len: {max_seq_len}')
print(f'Using device: {device}')
print(f'Llama model path: {Llama_path}')
print(f'BERT model path: {Bert_path}')
print(f'Evaluation data path: {data_path}')
print(f'Fine-tuned model path: {ft_path}') # Should be ft_model_cls_...
print("-----------------------------")

# Removed parse_generated_output function

def evalModel(model, dataset, batch_size):
    """Evaluates the classification model on the test dataset."""
    model.eval()
    all_preds_numeric = []
    all_gt_labels = dataset.get_label() # This should return a NumPy array

    # --- Convert GT labels to numeric (0/1) ---
    try:
        # FIX: Check array size instead of direct boolean evaluation
        if all_gt_labels.size == 0: # Handle empty dataset case
             print("Error: Ground truth labels list is empty.")
             return None # Return None to indicate failure

        # Check the type of the first element to determine conversion method
        # This is safe now because we know the array is not empty
        if isinstance(all_gt_labels[0], (str, np.str_)): # Check for python or numpy string types
            gt_numeric = np.array([1 if lbl == 'anomalous' else 0 for lbl in all_gt_labels], dtype=int)
            print("Converted string GT labels to numeric (0=normal, 1=anomalous).")
        elif np.issubdtype(all_gt_labels.dtype, np.integer): # Check if already integer type
             gt_numeric = all_gt_labels.astype(int) # Ensure it's standard int if needed
             print("GT labels are already numeric.")
        else:
             # Attempt conversion if it's some other type (e.g., float, object containing numbers)
             print(f"Warning: GT labels are of unexpected type {all_gt_labels.dtype}. Attempting conversion to int.")
             try:
                 gt_numeric = all_gt_labels.astype(int)
             except ValueError as e:
                 print(f"ERROR: Could not convert GT labels to numeric integers: {e}")
                 print("Please ensure the 'Label' column in your CSV contains only 'normal'/'anomalous' strings or 0/1 integers.")
                 return None # Return None to indicate failure

        print(f"First 10 GT labels (numeric): {gt_numeric[:10]}")
    except Exception as e:
        # Catch potential index errors if somehow the size check failed or other issues
        print(f"Error during GT label processing: {e}. Ensure dataset provides 'normal'/'anomalous' strings or 0/1 integers.")
        return None # Return None to indicate failure
    # --- End GT Conversion ---

    print(f"\n--- Starting Evaluation on {len(dataset)} samples ---")
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Batches"):
            batch_start = i
            batch_end = min(i + batch_size, len(dataset))
            this_batch_indexes = list(range(batch_start, batch_end))
            if not this_batch_indexes: continue

            this_batch_seqs, _ = dataset.get_batch(this_batch_indexes) # We only need sequences for inference

            try:
                # Get Classification Logits
                logits = model(this_batch_seqs) # model.forward() returns logits now

                if logits.shape[0] == 0:
                     print(f"Warning: Skipping batch {i // batch_size} due to empty logits output.")
                     num_skipped = len(this_batch_indexes)
                     # Use a placeholder that's clearly not 0 or 1
                     all_preds_numeric.extend([-1] * num_skipped) # Mark skipped
                     continue

                # Get Predictions from Logits
                preds = torch.argmax(logits, dim=-1)
                all_preds_numeric.extend(preds.cpu().numpy())

            except Exception as e:
                print(f"\nError during model inference in batch {i // batch_size}: {e}")
                # Use a placeholder that's clearly not 0 or 1
                all_preds_numeric.extend([-1] * len(this_batch_indexes)) # Mark errors


    print("\n--- Processing Predictions ---")
    preds_numeric = np.array(all_preds_numeric)

    # Ensure gt_numeric and preds_numeric have the same length before filtering
    if len(gt_numeric) != len(preds_numeric):
        print(f"CRITICAL ERROR: Length mismatch between ground truth labels ({len(gt_numeric)}) and raw predictions ({len(preds_numeric)}). This indicates a fundamental issue in batch processing or data handling.")
        return None # Return None to indicate failure

    valid_indices = (preds_numeric != -1) # Indices where prediction was successful
    num_errors = (~valid_indices).sum()
    if num_errors > 0:
        print(f"Warning: {num_errors} predictions failed during inference and were excluded from metrics.")
        preds_numeric_filtered = preds_numeric[valid_indices]
        gt_numeric_filtered = gt_numeric[valid_indices] # Filter GT labels accordingly
    else:
        preds_numeric_filtered = preds_numeric
        gt_numeric_filtered = gt_numeric # Use original GT if no errors

    if len(preds_numeric_filtered) == 0:
        print("Error: No valid predictions available to calculate metrics (all predictions might have failed).")
        return None # Return None to indicate failure

    # --- Calculate Metrics ---
    print("\n--- Calculating Performance Metrics ---")
    # This check should ideally not be needed if the initial length check passed, but good for safety
    if len(preds_numeric_filtered) != len(gt_numeric_filtered):
        print(f"Error: Mismatch after filtering predictions ({len(preds_numeric_filtered)}) and GT labels ({len(gt_numeric_filtered)}).")
        return None # Return None to indicate failure

    # Calculate metrics using the filtered arrays
    precision, recall, f1, _ = precision_recall_fscore_support(gt_numeric_filtered, preds_numeric_filtered, average='binary', pos_label=1, zero_division=0)
    accuracy = accuracy_score(gt_numeric_filtered, preds_numeric_filtered)
    # Get detailed metrics per class (0 and 1)
    p_detailed, r_detailed, f1_detailed, s_detailed = precision_recall_fscore_support(gt_numeric_filtered, preds_numeric_filtered, labels=[0, 1], zero_division=0)

    # Calculate Confusion Matrix
    try:
        # Ensure labels=[0, 1] to handle cases where one class might be missing in predictions/GT
        cm = confusion_matrix(gt_numeric_filtered, preds_numeric_filtered, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except ValueError as cm_error:
        print(f"Warning: Confusion matrix calculation encountered an issue: {cm_error}")
        # Fallback or report raw matrix if ravel fails (e.g., only one class present)
        cm_raw = confusion_matrix(gt_numeric_filtered, preds_numeric_filtered)
        print("Raw Confusion Matrix:")
        print(cm_raw)
        # Attempt to extract values carefully, defaulting to 0
        tn = cm_raw[0, 0] if cm_raw.shape == (2, 2) else 0
        fp = cm_raw[0, 1] if cm_raw.shape == (2, 2) else 0
        fn = cm_raw[1, 0] if cm_raw.shape == (2, 2) else 0
        tp = cm_raw[1, 1] if cm_raw.shape == (2, 2) else 0
        # Handle cases where only one class exists in both gt and pred
        if cm_raw.shape == (1, 1):
             unique_label = np.unique(gt_numeric_filtered)[0]
             if unique_label == 0: tn = cm_raw[0,0]
             elif unique_label == 1: tp = cm_raw[0,0]


    # Calculate counts based on the filtered ground truth and predictions
    num_anomalous_gt = (gt_numeric_filtered == 1).sum()
    num_normal_gt = (gt_numeric_filtered == 0).sum()
    pred_num_anomalous = (preds_numeric_filtered == 1).sum()
    pred_num_normal = (preds_numeric_filtered == 0).sum()

    print("\n--- Ground Truth Distribution (Evaluated Samples) ---")
    print(f"Anomalous sequences: {num_anomalous_gt}")
    print(f"Normal sequences:    {num_normal_gt}")
    print("\n--- Prediction Distribution (Evaluated Samples) ---")
    print(f"Predicted anomalous: {pred_num_anomalous}")
    print(f"Predicted normal:    {pred_num_normal}")
    if num_errors > 0:
        print(f"({num_errors} samples excluded due to inference errors)")

    print("\n--- Confusion Matrix ---")
    print(f"          Predicted Normal | Predicted Anomalous")
    print(f"Actual Normal:    {tn:6d}       | {fp:6d}")
    print(f"Actual Anomalous: {fn:6d}       | {tp:6d}")
    print("\n--- Overall Metrics (Positive Class = Anomalous) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\n--- Metrics Per Class ---")
    # Use the detailed support counts (s_detailed)
    print(f"Class 'Normal' (0):    Precision={p_detailed[0]:.4f}, Recall={r_detailed[0]:.4f}, F1={f1_detailed[0]:.4f}, Support={s_detailed[0]}")
    print(f"Class 'Anomalous' (1): Precision={p_detailed[1]:.4f}, Recall={r_detailed[1]:.4f}, F1={f1_detailed[1]:.4f}, Support={s_detailed[1]}")
    print("--------------------------------------")

    # Optionally return metrics if needed elsewhere
    # return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


if __name__ == '__main__':
    print("--- Initializing Dataset and Model for Evaluation ---")
    # Ensure data path exists after fallback logic
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Evaluation data not found at the specified/fallback path: {data_path}")
    if not os.path.exists(Bert_path):
        raise FileNotFoundError(f"BERT model not found at: {Bert_path}")
    if not os.path.exists(Llama_path):
        raise FileNotFoundError(f"Llama model not found at: {Llama_path}")

    # Check for CLASSIFICATION model components
    llama_adapter_config = os.path.join(ft_path, 'Llama_ft', 'adapter_config.json')
    projector_file = os.path.join(ft_path, 'projector.pt')
    classifier_file = os.path.join(ft_path, 'classifier.pt') # Check for classifier file

    # Check if the fine-tuned model directory itself exists first
    if not os.path.isdir(ft_path):
         raise FileNotFoundError(f"Fine-tuned model directory not found at {ft_path}. Cannot evaluate.")
    # Check for essential files within the directory
    if not os.path.exists(llama_adapter_config):
         print(f"Warning: Llama adapter config not found at {llama_adapter_config}. Llama PEFT adapters might not load correctly.")
         # Decide if this is critical - for now, allow proceeding but warn.
         # raise FileNotFoundError(f"Llama adapter config not found at {llama_adapter_config}. Cannot evaluate.")
    if not os.path.exists(projector_file):
         raise FileNotFoundError(f"Projector weights not found at {projector_file}. Cannot evaluate.")
    if not os.path.exists(classifier_file):
         raise FileNotFoundError(f"Classifier weights not found at {classifier_file}. Cannot evaluate.")

    try:
        dataset = CustomDataset(data_path)
        print(f"Evaluation dataset loaded: {len(dataset)} sequences.")
        if len(dataset) == 0:
            print("ERROR: Loaded dataset is empty. Cannot evaluate.")
            exit()
    except Exception as e:
        print(f"ERROR: Failed to load dataset from {data_path}: {e}")
        exit()

    try:
        # Load the classification model
        model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                       max_content_len=max_content_len, max_seq_len=max_seq_len)
        print("Model loaded for evaluation.")
    except Exception as e:
        print(f"ERROR: Failed to load the LogLLM model: {e}")
        # Add more specific error checking if needed (e.g., check paths again)
        exit()

    # Run evaluation
    evalModel(model, dataset, batch_size)

    print("\n--- Evaluation Script Finished ---")

# --- END OF FILE eval.py ---