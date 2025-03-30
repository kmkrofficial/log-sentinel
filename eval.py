import os
import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from model import LogLLM # Assuming model.py is accessible
from customDataset import CustomDataset # Assuming customDataset.py is accessible
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import gc

# --- Configuration ---
max_content_len = 100 # Should match training
max_seq_len = 128   # Should match training
batch_size = 16     # Keep reduced batch size for inference
dataset_name = 'BGL'   # Options: 'Thunderbird', 'HDFS_v1', 'BGL', 'Liberty'

# --- !!! CRITICAL: SET CORRECT TEST DATA PATH !!! ---
# Assuming test data is in the same directory as train.csv
base_data_dir = r'E:\research-stuff\LogLLM-3b\dataset'
# *** Verify this path points to your ACTUAL test data file ***
data_path = os.path.join(base_data_dir, 'test.csv') # <--- Set to test.csv
if not os.path.exists(data_path):
    print(f"WARNING: Test data file not found at {data_path}. Falling back to train.csv for evaluation.")
    data_path = os.path.join(base_data_dir, 'train.csv') # Fallback ONLY if test.csv missing
# --- End Critical Path Setting ---


# Paths (Using user's paths)
Bert_path = r"E:\research-stuff\LogLLM-3b\models\bert-base-uncased"
Llama_path = r"E:\research-stuff\LogLLM-3b\models\Llama-3.2-3B"

ROOT_DIR = Path(__file__).resolve().parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("--- Evaluation Configuration ---")
print(f'dataset_name: {dataset_name}')
print(f'batch_size: {batch_size}')
print(f'max_content_len: {max_content_len}')
print(f'max_seq_len: {max_seq_len}')
print(f'Using device: {device}')
print(f'Llama model path: {Llama_path}')
print(f'BERT model path: {Bert_path}')
print(f'Evaluation data path: {data_path}') # Verify this path is correct!
print(f'Fine-tuned model path: {ft_path}')
print("-----------------------------")


def parse_generated_output(text):
    """
    Parses the raw output text from the LLM to extract the classification.
    Looks for 'normal' or 'anomalous' keywords.
    Handles variations like periods, EOS tokens, etc.
    Returns 'normal', 'anomalous', or 'unknown'.
    """
    text_lower = text.lower().strip()
    if re.search(r'\banomalous\b', text_lower):
        return 'anomalous'
    elif re.search(r'\bnormal\b', text_lower):
        return 'normal'
    elif 'anomalous' in text_lower:
        # print(f"  [Parser Warning] Found 'anomalous' via substring: '{text[:50]}...'")
        return 'anomalous'
    elif 'normal' in text_lower:
        # print(f"  [Parser Warning] Found 'normal' via substring: '{text[:50]}...'")
        return 'normal'
    else:
        # print(f"  [Parser Warning] Could not parse output: '{text[:50]}...'")
        return 'unknown'


def evalModel(model, dataset, batch_size):
    """Evaluates the model on the test dataset."""
    model.eval()
    all_preds_text = []
    all_gt_labels = dataset.get_label()

    try:
        print(f"First 10 GT labels: {all_gt_labels[:10]}")
        print(f"Label type: {type(all_gt_labels[0]) if len(all_gt_labels) > 0 else 'N/A'}")
    except Exception as e: print(f"Could not print initial GT labels: {e}")

    print(f"\n--- Starting Evaluation on {len(dataset)} samples ---")
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Batches"):
            batch_start = i
            batch_end = min(i + batch_size, len(dataset))
            this_batch_indexes = list(range(batch_start, batch_end))
            if not this_batch_indexes: continue

            this_batch_seqs, _ = dataset.get_batch(this_batch_indexes)

            try:
                outputs_ids = model(this_batch_seqs)
                outputs_text_skip_special = model.Llama_tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
                outputs_text_no_skip = model.Llama_tokenizer.batch_decode(outputs_ids, skip_special_tokens=False)
                all_preds_text.extend(outputs_text_skip_special)

                # --- Debug Logging for Raw Outputs (Anomalous GT) ---
                gt_labels_batch = all_gt_labels[batch_start:batch_end]
                for idx, text_skip in enumerate(outputs_text_skip_special):
                    try:
                         gt = int(gt_labels_batch[idx])
                         if gt == 1: # If ground truth is anomalous
                             parsed_pred_debug = parse_generated_output(text_skip)
                             raw_no_skip = outputs_text_no_skip[idx]
                             # Only print if parsing failed or prediction is wrong
                             if parsed_pred_debug != 'anomalous':
                                 print(f"  [Debug Eval GT=1] Parsed: {parsed_pred_debug}, Raw (skip_special): '{text_skip}', Raw (no_skip): '{raw_no_skip}'")
                    except (ValueError, IndexError, TypeError) as e:
                         print(f"  [Debug Eval Error] Could not process GT label for index {idx} in batch: {e}")
                # --- End Debug Logging ---

            except Exception as e:
                print(f"\nError during model inference or decoding in batch {i // batch_size}: {e}")
                all_preds_text.extend(['error'] * len(this_batch_indexes))


    print("\n--- Processing Predictions ---")
    parsed_preds = [parse_generated_output(text) for text in all_preds_text]
    preds_numeric = np.zeros(len(parsed_preds), dtype=int)
    unknown_count, error_count = 0, 0
    for idx, p in enumerate(parsed_preds):
        raw_text = all_preds_text[idx]
        if p == 'anomalous': preds_numeric[idx] = 1
        elif p == 'normal': preds_numeric[idx] = 0
        elif raw_text == 'error': preds_numeric[idx] = 0; error_count += 1
        else: preds_numeric[idx] = 0; unknown_count += 1 # 'unknown'

    if unknown_count > 0: print(f"Warning: {unknown_count} predictions were 'unknown' (parsing issues) and defaulted to 'normal'.")
    if error_count > 0: print(f"Warning: {error_count} predictions failed (inference/decoding errors) and defaulted to 'normal'.")

    try:
        gt_numeric = np.array(all_gt_labels, dtype=int)
    except ValueError as e:
        print(f"Error: Could not convert ground truth labels to integers: {e}"); return

    print("\n--- Calculating Performance Metrics ---")
    if len(preds_numeric) != len(gt_numeric):
        print(f"Error: Mismatch between predictions ({len(preds_numeric)}) and GT labels ({len(gt_numeric)})."); return

    precision, recall, f1, _ = precision_recall_fscore_support(gt_numeric, preds_numeric, average='binary', pos_label=1, zero_division=0)
    accuracy = accuracy_score(gt_numeric, preds_numeric)
    p_detailed, r_detailed, f1_detailed, s_detailed = precision_recall_fscore_support(gt_numeric, preds_numeric, labels=[0, 1], zero_division=0)

    try:
        cm = confusion_matrix(gt_numeric, preds_numeric, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except ValueError:
        print("Warning: Confusion matrix could not be unpacked normally.")
        print(confusion_matrix(gt_numeric, preds_numeric)) # Print raw matrix
        tn, fp, fn, tp = 0, 0, 0, 0 # Default values

    num_anomalous_gt = (gt_numeric == 1).sum()
    num_normal_gt = (gt_numeric == 0).sum()
    pred_num_anomalous = (preds_numeric == 1).sum()
    pred_num_normal = (preds_numeric == 0).sum()

    print("\n--- Ground Truth Distribution ---")
    print(f"Anomalous sequences: {num_anomalous_gt}")
    print(f"Normal sequences:    {num_normal_gt}")
    print("\n--- Prediction Distribution ---")
    print(f"Predicted anomalous: {pred_num_anomalous}")
    print(f"Predicted normal:    {pred_num_normal}")
    if unknown_count > 0 or error_count > 0: print(f"(Includes {unknown_count} unknown/parse errors + {error_count} inference errors defaulted to normal)")
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
    print(f"Class 'Normal' (0):    Precision={p_detailed[0]:.4f}, Recall={r_detailed[0]:.4f}, F1={f1_detailed[0]:.4f}, Support={s_detailed[0]}")
    print(f"Class 'Anomalous' (1): Precision={p_detailed[1]:.4f}, Recall={r_detailed[1]:.4f}, F1={f1_detailed[1]:.4f}, Support={s_detailed[1]}")
    print("--------------------------------------")


if __name__ == '__main__':
    print("--- Initializing Dataset and Model for Evaluation ---")
    if not os.path.exists(data_path): raise FileNotFoundError(f"Evaluation data not found at: {data_path}")
    if not os.path.exists(Bert_path): raise FileNotFoundError(f"BERT model not found at: {Bert_path}")
    if not os.path.exists(Llama_path): raise FileNotFoundError(f"Llama model not found at: {Llama_path}")

    llama_adapter_config = os.path.join(ft_path, 'Llama_ft', 'adapter_config.json')
    projector_file = os.path.join(ft_path, 'projector.pt')
    if not os.path.exists(llama_adapter_config) or not os.path.exists(projector_file):
         raise FileNotFoundError(f"Fine-tuned model components not found at {ft_path}. Cannot evaluate.")

    dataset = CustomDataset(data_path)
    print(f"Evaluation dataset loaded: {len(dataset)} sequences.")

    model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)
    print("Model loaded for evaluation.")

    evalModel(model, dataset, batch_size)

    print("\n--- Evaluation Script Finished ---")