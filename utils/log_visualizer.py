import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import pandas as pd
import numpy as np

class LogVisualizer:
    def __init__(self, plot_dir):
        self.plot_dir = plot_dir
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        sns.set(style="whitegrid")

    def _save_plot(self, fig, filename):
        path = self.plot_dir / f"{filename}.png"
        fig.savefig(path)
        plt.close(fig)

    def plot_resource_usage(self, resource_df):
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.set_xlabel('Time (measurement point)')
            ax1.set_ylabel('RAM Usage (GB)', color='tab:blue')
            ax1.plot(resource_df.index, resource_df['ram_usage_gb'], color='tab:blue', label='RAM Used (GB)')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.set_title('RAM and VRAM Usage Over Time')
            fig.tight_layout()

            if 'gpu_vram_used_gb' in resource_df.columns:
                ax2 = ax1.twinx()
                ax2.set_ylabel('VRAM Usage (GB)', color='tab:red')
                ax2.plot(resource_df.index, resource_df['gpu_vram_used_gb'], color='tab:red', label='VRAM Used (GB)')
                ax2.tick_params(axis='y', labelcolor='tab:red')

            fig.tight_layout()
            self._save_plot(fig, "resource_usage_over_time")

        except Exception as e:
            print(f"Error plotting resource usage: {e}")

    def plot_training_loss(self, losses):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(losses, label='Batch Loss')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Time')
        ax.legend()
        self._save_plot(fig, "training_loss")

    def plot_confusion_matrix(self, y_true, y_pred, filename):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        self._save_plot(fig, filename)

    def plot_roc_curve(self, y_true, y_prob, filename):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        self._save_plot(fig, filename)

    def plot_precision_recall_curve(self, y_true, y_prob, filename):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:0.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        self._save_plot(fig, filename)

    def plot_distributions(self, y_prob, y_true, filename):
        df = pd.DataFrame({'probability': y_prob, 'label': y_true})
        normal_probs = df[df['label'] == 0]['probability']
        anomaly_probs = df[df['label'] == 1]['probability']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(normal_probs, color="blue", label='Normal', stat="density", kde=True, ax=ax)
        sns.histplot(anomaly_probs, color="red", label='Anomaly', stat="density", kde=True, ax=ax)
        ax.set_title('Anomaly Score Distributions')
        ax.set_xlabel('Anomaly Probability')
        ax.legend()
        self._save_plot(fig, filename)