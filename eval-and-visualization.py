# --- START OF FILE eval.py ---

import os
import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import time
import gc
import statistics # For mean calculation, handles empty lists gracefully

# Import visualization and monitoring libraries
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

# Import updated model (which now includes classifier)
from model import LogLLM
from customDataset import CustomDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- GPU Monitoring Setup ---
gpu_monitoring_available = False
nvml = None
gpu_handle = None
gpu_device_index = 0 # Default to 0
try:
    # Use pynvml for NVIDIA GPU monitoring
    import pynvml as nvml
    nvml.nvmlInit()
    # Assuming device 0, adjust if necessary based on `device` variable later if needed
    if torch.cuda.is_available():
        gpu_device_index = torch.cuda.current_device() # Get index used by torch
    gpu_handle = nvml.nvmlDeviceGetHandleByIndex(gpu_device_index)
    gpu_monitoring_available = True
    print(f"PyNVML initialized successfully for GPU {gpu_device_index}. GPU monitoring enabled.")
except Exception as e:
    print(f"Warning: PyNVML initialization failed: {e}. GPU monitoring disabled.")
    print("Ensure you have an NVIDIA GPU, CUDA drivers, and 'nvidia-ml-py' installed for GPU monitoring.")
# --- End GPU Monitoring Setup ---

# --- Configuration ---
max_content_len = 100
max_seq_len = 128
batch_size = 16 # Keep inference batch size reasonable
dataset_name = 'BGL' # Still used for model path, but not plot dir name

# --- Baseline Monitoring Configuration ---
baseline_duration_sec = 5 # How long to measure baseline usage
baseline_interval_sec = 0.5 # How often to sample during baseline measurement
# --- End Baseline Monitoring Configuration ---

# --- Set Correct Test Data Path ---
base_data_dir = r'E:\research-stuff\LogLLM-3b\dataset'
test_data_filename = 'test.csv'
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
        if nvml and gpu_monitoring_available: nvml.nvmlShutdown() # Cleanup NVML
        exit(f"Evaluation cannot proceed without data at {base_data_dir}")
# --- End Path Setting ---

# Paths
Bert_path = r"E:\research-stuff\LogLLM-3b\models\bert-base-uncased"
Llama_path = r"E:\research-stuff\LogLLM-3b\models\Llama-3.2-3B"

ROOT_DIR = Path(__file__).resolve().parent
# --- Point to CLASSIFICATION model path ---
# Use dataset_name here for model path consistency
ft_path = os.path.join(ROOT_DIR, r"ft_model_cls_{}".format(dataset_name))
# --- End Path Setting ---

# --- Directory for saving plots ---
# Changed directory name as requested
plot_dir = os.path.join(ROOT_DIR, 'visualizations')
os.makedirs(plot_dir, exist_ok=True)
print(f"Plots will be saved to: {plot_dir}")
# --- End Plot Directory ---

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("--- Evaluation Configuration (Classification Head) ---")
print(f'dataset_name: {dataset_name}') # Keep printing dataset name for clarity
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

def evalModel(model, dataset, batch_size, plot_dir, dataset_name_for_plots):
    """Evaluates the classification model on the test dataset and generates plots."""
    model.eval()
    all_preds_numeric = []
    all_gt_labels = dataset.get_label() # This should return a NumPy array

    # --- Resource Monitoring Setup ---
    process = psutil.Process(os.getpid())
    # Lists for INFERENCE measurements
    cpu_usage_list = []
    ram_usage_mb_list = []
    gpu_util_list = []
    gpu_mem_mb_list = []
    inference_times = []
    batch_indices_for_plots = []
    # Lists for BASELINE measurements
    baseline_cpu_usage = []
    baseline_gpu_util = []
    # --- End Resource Monitoring Setup ---

    # --- Measure Baseline Usage ---
    print(f"\n--- Measuring baseline resource usage for {baseline_duration_sec} seconds ---")
    # Initialize CPU measurement
    process.cpu_percent(interval=None)
    start_baseline_time = time.time()
    while time.time() - start_baseline_time < baseline_duration_sec:
        # Measure CPU
        baseline_cpu_usage.append(process.cpu_percent(interval=None))
        # Measure GPU (if available)
        if gpu_monitoring_available and gpu_handle:
            try:
                gpu_util = nvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                baseline_gpu_util.append(gpu_util)
            except nvml.NVMLError as nvml_error:
                # Only warn once during baseline
                if not baseline_gpu_util: # Check if it's the first attempt
                     print(f"Warning: NVML error during baseline GPU measurement: {nvml_error}. Further baseline errors suppressed.")
                baseline_gpu_util.append(None) # Append placeholder if error
        time.sleep(baseline_interval_sec)

    # Calculate average baseline usage
    avg_baseline_cpu = statistics.mean(u for u in baseline_cpu_usage if u is not None) if any(u is not None for u in baseline_cpu_usage) else 0
    avg_baseline_gpu_util = statistics.mean(u for u in baseline_gpu_util if u is not None) if any(u is not None for u in baseline_gpu_util) else 0
    print(f"Average Baseline CPU Usage: {avg_baseline_cpu:.2f}%")
    if gpu_monitoring_available:
        print(f"Average Baseline GPU Utilization: {avg_baseline_gpu_util:.2f}%")
    print("--- Baseline measurement finished ---")
    # --- End Baseline Measurement ---


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
    # Re-initialize CPU measurement before the loop
    process.cpu_percent(interval=None)
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Batches"):
            batch_start_idx = i
            batch_end_idx = min(i + batch_size, len(dataset))
            this_batch_indexes = list(range(batch_start_idx, batch_end_idx))
            if not this_batch_indexes: continue

            this_batch_seqs, _ = dataset.get_batch(this_batch_indexes) # We only need sequences for inference

            # --- Record Resources Before Inference ---
            batch_start_time = time.time()
            # Record CPU % right before the work
            cpu_usage_list.append(process.cpu_percent(interval=None))
            ram_usage_mb_list.append(process.memory_info().rss / (1024 * 1024)) # Resident Set Size in MB

            if gpu_monitoring_available and gpu_handle:
                try:
                    gpu_util = nvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                    gpu_mem_info = nvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    gpu_util_list.append(gpu_util)
                    gpu_mem_mb_list.append(gpu_mem_info.used / (1024 * 1024)) # Used GPU memory in MB
                except nvml.NVMLError as nvml_error:
                    # Don't print warning every batch if it fails repeatedly
                    if i == 0: print(f"Warning: NVML error getting GPU stats during eval: {nvml_error}. Further errors will be suppressed.")
                    gpu_util_list.append(None) # Append placeholder
                    gpu_mem_mb_list.append(None)
            # --- End Resource Recording ---

            try:
                # Get Classification Logits
                logits = model(this_batch_seqs) # model.forward() returns logits now

                # --- Record Time After Inference ---
                batch_end_time = time.time()
                inference_times.append(batch_end_time - batch_start_time)
                batch_indices_for_plots.append(i // batch_size) # Store batch number
                # --- End Time Recording ---


                if logits.shape[0] == 0:
                     print(f"Warning: Skipping batch {i // batch_size} due to empty logits output.")
                     num_skipped = len(this_batch_indexes)
                     # Use a placeholder that's clearly not 0 or 1
                     all_preds_numeric.extend([-1] * num_skipped) # Mark skipped
                     # Add placeholder time/resources if needed, or skip this batch in plots
                     continue

                # Get Predictions from Logits
                preds = torch.argmax(logits, dim=-1)
                all_preds_numeric.extend(preds.cpu().numpy())

            except Exception as e:
                print(f"\nError during model inference in batch {i // batch_size}: {e}")
                # Use a placeholder that's clearly not 0 or 1
                all_preds_numeric.extend([-1] * len(this_batch_indexes)) # Mark errors
                # Record time even if error occurred during prediction part
                batch_end_time = time.time()
                inference_times.append(batch_end_time - batch_start_time)
                batch_indices_for_plots.append(i // batch_size) # Store batch number


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
    # Overall metrics (treating 'anomalous' as the positive class)
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

    # --- Performance Metrics (Averages, Peaks, Percentiles) ---
    # Filter out None values before calculations
    valid_cpu_usage = [c for c in cpu_usage_list if c is not None]
    valid_ram_usage = [r for r in ram_usage_mb_list if r is not None]
    valid_gpu_util = [g for g in gpu_util_list if g is not None]
    valid_gpu_mem = [m for m in gpu_mem_mb_list if m is not None]
    valid_inference_times = [t for t in inference_times if t is not None]

    # Averages
    avg_inference_time = statistics.mean(valid_inference_times) if valid_inference_times else 0
    avg_cpu_usage = statistics.mean(valid_cpu_usage) if valid_cpu_usage else 0
    avg_ram_usage = statistics.mean(valid_ram_usage) if valid_ram_usage else 0
    avg_gpu_util = statistics.mean(valid_gpu_util) if valid_gpu_util else 0
    avg_gpu_mem = statistics.mean(valid_gpu_mem) if valid_gpu_mem else 0

    # --- MODIFIED: Calculate Peak CPU/GPU Util, 95th Percentile RAM/GPU Mem ---
    peak_cpu_usage = max(valid_cpu_usage) if valid_cpu_usage else 0
    peak_gpu_util = max(valid_gpu_util) if valid_gpu_util else 0
    p95_ram_usage = np.percentile(valid_ram_usage, 95) if valid_ram_usage else 0
    p95_gpu_mem = np.percentile(valid_gpu_mem, 95) if valid_gpu_mem else 0
    p95_inference_time = np.percentile(valid_inference_times, 95) if valid_inference_times else 0 # Keep 95p for time

    # Adjusted Peak Usage (Subtracting Baseline Average)
    # Clamp at 0 to avoid negative usage
    adjusted_peak_cpu_usage = max(0, peak_cpu_usage - avg_baseline_cpu)
    adjusted_peak_gpu_util = max(0, peak_gpu_util - avg_baseline_gpu_util) if gpu_monitoring_available else 0
    # --- END MODIFICATION ---

    print("\n--- Performance & Resource Usage ---")
    print("  Average per Batch:")
    print(f"    Inference Time: {avg_inference_time:.4f} seconds")
    print(f"    CPU Usage:      {avg_cpu_usage:.2f}%")
    print(f"    RAM Usage:      {avg_ram_usage:.2f} MB")
    if gpu_monitoring_available:
        print(f"    GPU Utilization:{avg_gpu_util:.2f}%")
        print(f"    GPU Memory Used:{avg_gpu_mem:.2f} MB")
    else:
        print("    GPU Monitoring was disabled.")

    # --- MODIFIED: Report Peak CPU/GPU Util, 95th Percentile RAM ---
    print("\n  Peak/High Usage during Inference (Raw):")
    print(f"    Peak CPU Usage:          {peak_cpu_usage:.2f}%")
    print(f"    95th Percentile RAM Usage: {p95_ram_usage:.2f} MB")
    if gpu_monitoring_available:
        print(f"    Peak GPU Utilization:    {peak_gpu_util:.2f}%")
        print(f"    95th Percentile GPU Mem: {p95_gpu_mem:.2f} MB")
    print(f"    95th Percentile Time:    {p95_inference_time:.4f} seconds") # Keep 95p time

    print("\n  Adjusted Peak Usage (Inference Peak - Baseline Avg):")
    print(f"    Adjusted Peak CPU Usage:      {adjusted_peak_cpu_usage:.2f}%")
    if gpu_monitoring_available:
        print(f"    Adjusted Peak GPU Utilization:{adjusted_peak_gpu_util:.2f}%")
    # --- END MODIFICATION ---

    print("--------------------------------------")


    # --- Generate Visualizations ---
    print("\n--- Generating Visualizations ---")

    # 1. Confusion Matrix Heatmap (Existing)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal (0)', 'Anomalous (1)'],
                yticklabels=['Normal (0)', 'Anomalous (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {dataset_name_for_plots}')
    plt.tight_layout()
    cm_filename = os.path.join(plot_dir, f'confusion_matrix_{dataset_name_for_plots}.png')
    plt.savefig(cm_filename)
    print(f"Confusion matrix saved to: {cm_filename}")
    plt.close() # Close the figure

    # 2. Overall Metrics Bar Chart (Existing)
    overall_metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    overall_metrics_values = [accuracy, precision, recall, f1]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(overall_metrics_names, overall_metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.ylabel('Score')
    plt.title(f'Overall Classification Metrics - {dataset_name_for_plots}')
    plt.ylim(0, 1.1) # Scores are between 0 and 1
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center') # Adjust position and format
    plt.tight_layout()
    overall_metrics_filename = os.path.join(plot_dir, f'overall_metrics_{dataset_name_for_plots}.png')
    plt.savefig(overall_metrics_filename)
    print(f"Overall metrics plot saved to: {overall_metrics_filename}")
    plt.close()

    # 3. Per-Class Metrics Grouped Bar Chart (Existing)
    class_labels = ['Normal (0)', 'Anomalous (1)']
    metric_types = ['Precision', 'Recall', 'F1-Score']
    normal_metrics = [p_detailed[0], r_detailed[0], f1_detailed[0]]
    anomalous_metrics = [p_detailed[1], r_detailed[1], f1_detailed[1]]

    x = np.arange(len(metric_types))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, normal_metrics, width, label='Normal', color='cornflowerblue')
    rects2 = ax.bar(x + width/2, anomalous_metrics, width, label='Anomalous', color='salmon')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Score')
    ax.set_title(f'Metrics per Class - {dataset_name_for_plots}')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_types)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Function to attach a text label above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    per_class_metrics_filename = os.path.join(plot_dir, f'per_class_metrics_{dataset_name_for_plots}.png')
    plt.savefig(per_class_metrics_filename)
    print(f"Per-class metrics plot saved to: {per_class_metrics_filename}")
    plt.close(fig)

    # 4. Ground Truth vs. Predicted Distribution (Existing)
    dist_labels = ['Normal', 'Anomalous']
    gt_counts = [num_normal_gt, num_anomalous_gt]
    pred_counts = [pred_num_normal, pred_num_anomalous]

    x_dist = np.arange(len(dist_labels))
    width_dist = 0.35

    fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
    rects_gt = ax_dist.bar(x_dist - width_dist/2, gt_counts, width_dist, label='Ground Truth', color='darkseagreen')
    rects_pred = ax_dist.bar(x_dist + width_dist/2, pred_counts, width_dist, label='Predicted', color='lightsteelblue')

    ax_dist.set_ylabel('Number of Samples')
    ax_dist.set_title(f'Ground Truth vs. Predicted Class Distribution - {dataset_name_for_plots}')
    ax_dist.set_xticks(x_dist)
    ax_dist.set_xticklabels(dist_labels)
    ax_dist.legend()

    # Function to attach count labels
    def autolabel_counts(rects):
        for rect in rects:
            height = rect.get_height()
            ax_dist.annotate(f'{height}',
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom')

    autolabel_counts(rects_gt)
    autolabel_counts(rects_pred)

    fig_dist.tight_layout()
    distribution_filename = os.path.join(plot_dir, f'distribution_comparison_{dataset_name_for_plots}.png')
    plt.savefig(distribution_filename)
    print(f"Distribution comparison plot saved to: {distribution_filename}")
    plt.close(fig_dist)


    # 5. Inference Time per Batch (Existing)
    plt.figure(figsize=(10, 5))
    plt.plot(batch_indices_for_plots, valid_inference_times, marker='o', linestyle='-', markersize=4) # Plot valid times
    plt.xlabel('Batch Index')
    plt.ylabel('Inference Time (seconds)')
    plt.title(f'Inference Time per Batch - {dataset_name_for_plots}')
    plt.grid(True)
    plt.tight_layout()
    time_filename = os.path.join(plot_dir, f'inference_time_{dataset_name_for_plots}.png')
    plt.savefig(time_filename)
    print(f"Inference time plot saved to: {time_filename}")
    plt.close() # Close the figure

    # 6. Resource Usage Plots (Existing - plotting raw per-batch usage)
    num_batches_plot = len(batch_indices_for_plots)
    if num_batches_plot > 0: # Ensure there's data to plot
        fig_res, axs_res = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # Shared X-axis (Batch Index)

        # CPU and RAM Usage
        # Plot only valid points if Nones exist, otherwise plot directly
        valid_cpu_indices = [idx for idx, val in enumerate(cpu_usage_list[:num_batches_plot]) if val is not None]
        valid_batch_indices_cpu = [batch_indices_for_plots[i] for i in valid_cpu_indices]
        plot_cpu_usage = [cpu_usage_list[i] for i in valid_cpu_indices]

        valid_ram_indices = [idx for idx, val in enumerate(ram_usage_mb_list[:num_batches_plot]) if val is not None]
        valid_batch_indices_ram = [batch_indices_for_plots[i] for i in valid_ram_indices]
        plot_ram_usage = [ram_usage_mb_list[i] for i in valid_ram_indices]


        if valid_batch_indices_cpu:
            axs_res[0].plot(valid_batch_indices_cpu, plot_cpu_usage, label='CPU Usage (%)', color='tab:blue', marker='.', linestyle='-', markersize=4)
        axs_res[0].set_ylabel('CPU Usage (%)', color='tab:blue')
        axs_res[0].tick_params(axis='y', labelcolor='tab:blue')
        axs_res[0].grid(True, axis='y')

        ax0_twin = axs_res[0].twinx() # instantiate a second axes that shares the same x-axis
        if valid_batch_indices_ram:
            ax0_twin.plot(valid_batch_indices_ram, plot_ram_usage, label='RAM Usage (MB)', color='tab:green', marker='.', linestyle='-', markersize=4)
        ax0_twin.set_ylabel('RAM Usage (MB)', color='tab:green')
        ax0_twin.tick_params(axis='y', labelcolor='tab:green')

        axs_res[0].set_title(f'CPU and RAM Usage During Inference - {dataset_name_for_plots}')
        # Combine legends
        lines, labels = axs_res[0].get_legend_handles_labels()
        lines2, labels2 = ax0_twin.get_legend_handles_labels()
        ax0_twin.legend(lines + lines2, labels + labels2, loc='upper left')


        # GPU Usage (if available)
        if gpu_monitoring_available and any(g is not None for g in gpu_util_list):
            valid_gpu_indices = [idx for idx, val in enumerate(gpu_util_list[:num_batches_plot]) if val is not None]
            valid_batch_indices_gpu = [batch_indices_for_plots[i] for i in valid_gpu_indices]
            plot_gpu_util = [gpu_util_list[i] for i in valid_gpu_indices]

            valid_gpu_mem_indices = [idx for idx, val in enumerate(gpu_mem_mb_list[:num_batches_plot]) if val is not None]
            # Assume same indices are valid for mem as for util if util worked
            plot_gpu_mem = [gpu_mem_mb_list[i] for i in valid_gpu_indices]


            if valid_gpu_indices: # Check if there's any valid GPU data
                axs_res[1].plot(valid_batch_indices_gpu, plot_gpu_util, label='GPU Utilization (%)', color='tab:red', marker='.', linestyle='-', markersize=4)
                axs_res[1].set_ylabel('GPU Utilization (%)', color='tab:red')
                axs_res[1].tick_params(axis='y', labelcolor='tab:red')
                axs_res[1].grid(True, axis='y')

                ax1_twin = axs_res[1].twinx()
                ax1_twin.plot(valid_batch_indices_gpu, plot_gpu_mem, label='GPU Memory Used (MB)', color='tab:purple', marker='.', linestyle='-', markersize=4)
                ax1_twin.set_ylabel('GPU Memory Used (MB)', color='tab:purple')
                ax1_twin.tick_params(axis='y', labelcolor='tab:purple')

                axs_res[1].set_title(f'GPU Usage During Inference - {dataset_name_for_plots}')
                # Combine legends
                lines, labels = axs_res[1].get_legend_handles_labels()
                lines2, labels2 = ax1_twin.get_legend_handles_labels()
                ax1_twin.legend(lines + lines2, labels + labels2, loc='upper left')
            else:
                 axs_res[1].set_title(f'GPU Usage During Inference - Data Unavailable ({dataset_name_for_plots})')
                 axs_res[1].text(0.5, 0.5, 'GPU monitoring data points missing or invalid.', horizontalalignment='center', verticalalignment='center', transform=axs_res[1].transAxes)

        else:
            axs_res[1].set_title(f'GPU Usage During Inference - Monitoring Disabled ({dataset_name_for_plots})')
            axs_res[1].text(0.5, 0.5, 'GPU monitoring was not available or failed.', horizontalalignment='center', verticalalignment='center', transform=axs_res[1].transAxes)

        axs_res[1].set_xlabel('Batch Index')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
        resource_filename = os.path.join(plot_dir, f'resource_usage_{dataset_name_for_plots}.png')
        plt.savefig(resource_filename)
        print(f"Resource usage plot saved to: {resource_filename}")
        plt.close(fig_res) # Close the figure
    else:
        print("Skipping resource usage plots as no batch data was collected.")

    print("--------------------------------------")

    # Optionally return metrics if needed elsewhere
    # return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


if __name__ == '__main__':
    print("--- Initializing Dataset and Model for Evaluation ---")
    # Ensure data path exists after fallback logic
    if not os.path.exists(data_path):
        if nvml and gpu_monitoring_available: nvml.nvmlShutdown() # Cleanup NVML
        raise FileNotFoundError(f"Evaluation data not found at the specified/fallback path: {data_path}")
    if not os.path.exists(Bert_path):
        if nvml and gpu_monitoring_available: nvml.nvmlShutdown() # Cleanup NVML
        raise FileNotFoundError(f"BERT model not found at: {Bert_path}")
    if not os.path.exists(Llama_path):
        if nvml and gpu_monitoring_available: nvml.nvmlShutdown() # Cleanup NVML
        raise FileNotFoundError(f"Llama model not found at: {Llama_path}")

    # Check for CLASSIFICATION model components
    llama_adapter_config = os.path.join(ft_path, 'Llama_ft', 'adapter_config.json')
    projector_file = os.path.join(ft_path, 'projector.pt')
    classifier_file = os.path.join(ft_path, 'classifier.pt') # Check for classifier file

    # Check if the fine-tuned model directory itself exists first
    if not os.path.isdir(ft_path):
         if nvml and gpu_monitoring_available: nvml.nvmlShutdown() # Cleanup NVML
         raise FileNotFoundError(f"Fine-tuned model directory not found at {ft_path}. Cannot evaluate.")
    # Check for essential files within the directory
    if not os.path.exists(llama_adapter_config):
         print(f"Warning: Llama adapter config not found at {llama_adapter_config}. Llama PEFT adapters might not load correctly.")
         # Decide if this is critical - for now, allow proceeding but warn.
         # raise FileNotFoundError(f"Llama adapter config not found at {llama_adapter_config}. Cannot evaluate.")
    if not os.path.exists(projector_file):
         if nvml and gpu_monitoring_available: nvml.nvmlShutdown() # Cleanup NVML
         raise FileNotFoundError(f"Projector weights not found at {projector_file}. Cannot evaluate.")
    if not os.path.exists(classifier_file):
         if nvml and gpu_monitoring_available: nvml.nvmlShutdown() # Cleanup NVML
         raise FileNotFoundError(f"Classifier weights not found at {classifier_file}. Cannot evaluate.")

    try:
        dataset = CustomDataset(data_path)
        print(f"Evaluation dataset loaded: {len(dataset)} sequences.")
        if len(dataset) == 0:
            print("ERROR: Loaded dataset is empty. Cannot evaluate.")
            if nvml and gpu_monitoring_available: nvml.nvmlShutdown() # Cleanup NVML
            exit()
    except Exception as e:
        print(f"ERROR: Failed to load dataset from {data_path}: {e}")
        if nvml and gpu_monitoring_available: nvml.nvmlShutdown() # Cleanup NVML
        exit()

    try:
        # Load the classification model
        model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                       max_content_len=max_content_len, max_seq_len=max_seq_len)
        print("Model loaded for evaluation.")
    except Exception as e:
        print(f"ERROR: Failed to load the LogLLM model: {e}")
        # Add more specific error checking if needed (e.g., check paths again)
        if nvml and gpu_monitoring_available: nvml.nvmlShutdown() # Cleanup NVML
        exit()

    # Run evaluation, passing the plot directory and dataset name for plot titles/filenames
    evalModel(model, dataset, batch_size, plot_dir, dataset_name)

    # --- Cleanup NVML ---
    if nvml and gpu_monitoring_available:
        try:
            nvml.nvmlShutdown()
            print("PyNVML shut down successfully.")
        except Exception as e:
            print(f"Warning: Error shutting down PyNVML: {e}")
    # --- End Cleanup ---

    print("\n--- Evaluation Script Finished ---")

# --- END OF FILE eval.py ---