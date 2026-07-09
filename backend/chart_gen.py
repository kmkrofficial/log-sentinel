import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os

def create_visualizations_seaborn():
    """
    Main function to load data and generate all comparison plots using Seaborn,
    saving them to a local directory with white backgrounds, as individual files,
    and with all data values printed.
    """
    
    # --- 0. Setup Save Directory ---
    SAVE_DIR = "model_visualizations"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")
    
    # --- 1. Data Transcription from Images ---
    # (Data transcription is identical to the previous script)
    
    # Based on "TABLE 3: Average Time take to complete the whole run"
    avg_metrics_data = {
        'Model': ['DeepLog', 'LogAnomaly', 'PLELog', 'FastLogAD', 'LogBERT', 
                  'LogRobust', 'CNN', 'NeuralLog', 'RAPID', 'LogLLM', 'LogSentinel'],
        'Avg. F1-Score': [0.506, 0.521, 0.81, 0.341, 0.456, 
                          0.771, 0.818, 0.893, 0.602, 0.959, 0.945],
        'Avg. Train Time': [72.17, 156.16, 315.47, 108.42, 98.16, 
                            254.17, 429.04, 267.46, 63.98, 1065.15, 877.02],
        'Avg. Test Time': [3.42, 7.25, 33.59, 2.48, 2.16, 
                           0.29, 43.77, 21.44, 38.43, 64.48, 3.7979]
    }
    df_avg = pd.DataFrame(avg_metrics_data).set_index('Model')

    # Based on "TABLE 1: Dataset Analysis"
    total_test_sequences = 115013 + 9427 + 10000 + 20000

    # Based on "TABLE 2: Final Metrics"
    data_table2 = {
        'Model': ['LogLLM', 'LogSentinel', 'NeuralLog', 'CNN'] * 4,
        'Dataset': ['HDFS'] * 4 + ['BGL'] * 4 + ['Liberty'] * 4 + ['Thunderbird'] * 4,
        'Prec': [0.994, 0.9891, 0.971, 0.966,  # HDFS
                 0.861, 0.9879, 0.792, 0.698,  # BGL
                 0.992, 1, 0.875, 0.58,        # Liberty
                 0.966, 0.9484, 0.794, 0.87],  # Thunderbird
        'Recall': [1, 0.9968, 0.988, 1,          # HDFS
                   0.979, 0.8786, 0.884, 0.965,  # BGL
                   0.926, 0.801, 0.926, 0.914,   # Liberty
                   0.966, 1, 0.931, 0.69],      # Thunderbird
        'F1': [0.997, 0.9929, 0.979, 0.982,   # HDFS
               0.916, 0.9263, 0.835, 0.81,    # BGL
               0.958, 0.8895, 0.9, 0.709,     # Liberty
               0.966, 0.9735, 0.857, 0.769]   # Thunderbird
    }
    df_dataset = pd.DataFrame(data_table2)


    # --- 2. Data Processing ---
    # (Data processing is identical to the previous script)
    
    top_4_models = df_avg.nlargest(4, 'Avg. F1-Score').index.tolist()
    df_top4_avg = df_avg.loc[top_4_models].sort_values(by='Avg. F1-Score', ascending=False)
    df_top4_avg['Inference Time per Record (ms)'] = \
        (df_top4_avg['Avg. Test Time'] / total_test_sequences) * 1000

    logsentinel_color = '#ff7f0e'
    other_colors = ['#1f77b4', '#2ca02c', '#9467bd']
    
    color_map = {}
    other_idx = 0
    model_order = df_top4_avg.index
    for model in model_order:
        if model == 'LogSentinel':
            color_map[model] = logsentinel_color
        else:
            color_map[model] = other_colors[other_idx]
            other_idx += 1
            
    print(f"--- Analysis Parameters ---")
    print(f"Top 4 Models (by Avg. F1): {top_4_models}")
    print(f"LogSentinel highlight color: {logsentinel_color}")
    print(f"---------------------------\n")


    # --- 3. Plotting (Seaborn) ---
    
    sns.set_theme(style="whitegrid")
    warnings.filterwarnings("ignore")

    # --- Plot 1: Overall Performance (Avg. F1-Score) ---
    print("Generating Plot 1: Average F1-Score...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(
        ax=ax1,
        x=df_top4_avg.index, 
        y=df_top4_avg['Avg. F1-Score'], 
        palette=color_map,
        order=model_order
    )
    ax1.set_title('Overall Performance: Average F1-Score (Top 4 Models)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Average F1-Score', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylim(0.8, 1.0)
    
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, 
                    xytext=(0, 9), textcoords='offset points')
    
    plt.tight_layout()
    save_path1 = os.path.join(SAVE_DIR, "plot_1_avg_f1_score.png")
    fig1.savefig(save_path1, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {save_path1}\n")

    # --- Plot 2: Average Training Time (INDIVIDUAL) ---
    print("Generating Plot 2: Average Training Time...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(
        ax=ax2,
        x=df_top4_avg.index,
        y=df_top4_avg['Avg. Train Time'],
        palette=color_map,
        order=model_order
    )
    ax2.set_title('Average Training Time (Logarithmic Scale)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Time (seconds, log scale)', fontsize=12)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_yscale('log')

    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.2f}s', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, 
                    xytext=(0, 9), textcoords='offset points')
    
    plt.tight_layout()
    save_path2 = os.path.join(SAVE_DIR, "plot_2_avg_train_time.png")
    fig2.savefig(save_path2, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {save_path2}\n")

    # --- Plot 3: Average Test Time (INDIVIDUAL) ---
    print("Generating Plot 3: Average Test Time...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(
        ax=ax3,
        x=df_top4_avg.index,
        y=df_top4_avg['Avg. Test Time'],
        palette=color_map,
        order=model_order
    )
    ax3.set_title('Average Inference (Test) Time (Logarithmic Scale)', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Time (seconds, log scale)', fontsize=12)
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_yscale('log')

    for p in ax3.patches:
        ax3.annotate(f'{p.get_height():.3f}s', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, 
                    xytext=(0, 9), textcoords='offset points')
    
    plt.tight_layout()
    save_path3 = os.path.join(SAVE_DIR, "plot_3_avg_test_time.png")
    fig3.savefig(save_path3, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {save_path3}\n")

    # --- Plot 4: Inference Time per Record (INDIVIDUAL) ---
    print("Generating Plot 4: Inference Time per Record...")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    df_top4_avg_sorted_inference = df_top4_avg.sort_values(by='Inference Time per Record (ms)')
    
    sns.barplot(
        ax=ax4,
        x=df_top4_avg_sorted_inference.index, 
        y=df_top4_avg_sorted_inference['Inference Time per Record (ms)'], 
        palette=color_map,
        order=df_top4_avg_sorted_inference.index
    )
    ax4.set_title('Efficiency: Avg. Inference Time per Record (Logarithmic Scale)', 
                 fontsize=16, fontweight='bold')
    ax4.set_ylabel('Time per Record (milliseconds, log scale)', fontsize=12)
    ax4.set_xlabel('Model', fontsize=12)
    ax4.set_yscale('log') 
    
    for p in ax4.patches:
        ax4.annotate(f'{p.get_height():.4f} ms', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, 
                    xytext=(0, 9), textcoords='offset points')

    plt.tight_layout()
    save_path4 = os.path.join(SAVE_DIR, "plot_4_inference_per_record.png")
    fig4.savefig(save_path4, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {save_path4}\n")

    # --- Plots 5, 6, 7: Per-Dataset Metrics (INDIVIDUAL PLOTS) ---
    
    df_dataset_long = df_dataset.melt(
        id_vars=['Model', 'Dataset'], 
        value_vars=['F1', 'Prec', 'Recall'], 
        var_name='Metric', 
        value_name='Score'
    )
    df_dataset_long = df_dataset_long[df_dataset_long['Model'].isin(top_4_models)]
    metric_map = {'F1': 'F1-Score', 'Prec': 'Precision', 'Recall': 'Recall'}
    df_dataset_long['Metric'] = df_dataset_long['Metric'].map(metric_map)

    # Loop to create three separate files
    plot_number = 5
    for metric_name in ['F1-Score', 'Precision', 'Recall']:
        print(f"Generating Plot {plot_number}: {metric_name} by Dataset...")
        
        metric_df = df_dataset_long[df_dataset_long['Metric'] == metric_name]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        sns.barplot(
            ax=ax,
            data=metric_df,
            x='Dataset',
            y='Score',
            hue='Model',
            palette=color_map,
            hue_order=model_order
        )
        
        ax.set_title(f'Model Comparison by Dataset: {metric_name}', fontsize=18, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=14)
        ax.set_xlabel('Dataset', fontsize=14)
        ax.set_ylim(0, 1.15) # Increased Y-limit for label space
        
        # *** ADDED THIS LOOP to print values on bars ***
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.3f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', fontsize=8, 
                            rotation=90, # Rotate for grouped bars
                            xytext=(0, 5), textcoords='offset points')
        
        ax.legend(title='Models', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1]) 
        
        filename_safe_metric = metric_name.lower().replace('-', '_')
        save_path = os.path.join(SAVE_DIR, f"plot_{plot_number}_{filename_safe_metric}.png")
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {save_path}\n")
        plot_number += 1

    print("--- All visualizations saved successfully. ---")

# --- Run the script ---
if __name__ == "__main__":
    create_visualizations_seaborn()