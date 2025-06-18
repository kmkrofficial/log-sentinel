import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc

class LogVisualizer:
    def __init__(self, plot_dir):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.plot_paths = {}

    def _save_plot(self, fig, filename):
        path = self.plot_dir / filename
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        self.plot_paths[filename.split('.')[0]] = str(path)
        print(f"Saved plot: {path}")

    def plot_confusion_matrix(self, cm, class_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        self._save_plot(fig, 'confusion_matrix.png')

    def plot_overall_metrics(self, metrics):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(metrics.keys(), metrics.values(), color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax.set_ylabel('Score')
        ax.set_title('Overall Classification Metrics')
        ax.set_ylim(0, 1.1)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
        self._save_plot(fig, 'overall_metrics.png')
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True)
        self._save_plot(fig, 'roc_curve.png')

    def plot_training_loss(self, loss_data):
        if not loss_data:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(loss_data)), loss_data, label='Batch Loss')
        # Simple moving average to show the trend
        if len(loss_data) > 100:
            moving_avg = np.convolve(loss_data, np.ones(100)/100, mode='valid')
            ax.plot(np.arange(99, len(loss_data)), moving_avg, color='red', linestyle='--', label='100-step Moving Avg')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Time')
        ax.legend()
        ax.grid(True)
        self._save_plot(fig, 'training_loss.png')

    def plot_resource_usage(self, resource_metrics):
        ts_data = resource_metrics.get('time_series', {})
        summary = resource_metrics.get('summary', {})
        timestamps = ts_data.get('timestamps')
        if not timestamps: return

        if 'cpu_usage_percent' in ts_data and ts_data['cpu_usage_percent']:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, ts_data['cpu_usage_percent'], color='tab:blue', label='CPU Usage (%)')
            avg = summary.get('cpu', {}).get('avg_cpu_usage_percent', 0)
            ax.axhline(y=avg, color='r', linestyle='--', label=f"Avg: {avg:.2f}%")
            ax.set_ylabel('CPU Usage (%)'); ax.set_title('CPU Utilization Over Time')
            ax.legend(); ax.grid(True); ax.set_xlabel('Time (seconds)')
            self._save_plot(fig, 'cpu_usage.png')

        if 'ram_usage_gb' in ts_data and ts_data['ram_usage_gb']:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, ts_data['ram_usage_gb'], color='tab:green', label='RAM Usage (GB)')
            avg = summary.get('ram', {}).get('avg_ram_usage_gb', 0)
            ax.axhline(y=avg, color='r', linestyle='--', label=f"Avg: {avg:.2f} GB")
            ax.set_ylabel('RAM Usage (GB)'); ax.set_title('RAM Consumption Over Time')
            ax.legend(); ax.grid(True); ax.set_xlabel('Time (seconds)')
            self._save_plot(fig, 'ram_usage.png')

        if 'gpu' in summary:
            if 'gpu_util_percent' in ts_data and any(v is not None for v in ts_data['gpu_util_percent']):
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(timestamps, ts_data['gpu_util_percent'], color='tab:red', label='GPU Utilization (%)')
                avg = summary.get('gpu', {}).get('avg_gpu_util_percent', 0)
                ax.axhline(y=avg, color='b', linestyle='--', label=f"Avg: {avg:.2f}%")
                ax.set_ylabel('GPU Utilization (%)'); ax.set_title('GPU Utilization Over Time')
                ax.legend(); ax.grid(True); ax.set_xlabel('Time (seconds)')
                self._save_plot(fig, 'gpu_utilization.png')

            if 'gpu_mem_used_gb' in ts_data and any(v is not None for v in ts_data['gpu_mem_used_gb']):
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(timestamps, ts_data['gpu_mem_used_gb'], color='tab:purple', label='GPU Memory (GB)')
                avg = summary.get('gpu', {}).get('avg_gpu_mem_gb', 0)
                ax.axhline(y=avg, color='b', linestyle='--', label=f"Avg: {avg:.2f} GB")
                ax.set_ylabel('GPU Memory (GB)'); ax.set_title('GPU Memory Consumption Over Time')
                ax.legend(); ax.grid(True); ax.set_xlabel('Time (seconds)')
                self._save_plot(fig, 'gpu_memory.png')
    
    def get_plot_paths(self):
        return self.plot_paths