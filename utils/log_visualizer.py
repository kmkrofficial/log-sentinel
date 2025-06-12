import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

class LogVisualizer:
    def __init__(self, plot_dir):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.plot_paths = {}

    def _save_plot(self, fig, filename):
        """Saves a matplotlib figure to the plot directory."""
        path = self.plot_dir / filename
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        self.plot_paths[filename.split('.')[0]] = str(path)
        print(f"Saved plot: {path}")

    def plot_confusion_matrix(self, cm, class_names):
        """Generates and saves a confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        self._save_plot(fig, 'confusion_matrix.png')

    def plot_overall_metrics(self, metrics):
        """Generates and saves a bar chart for overall performance metrics."""
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(metrics.keys())
        values = list(metrics.values())
        bars = ax.bar(names, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax.set_ylabel('Score')
        ax.set_title('Overall Classification Metrics')
        ax.set_ylim(0, 1.1)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
        self._save_plot(fig, 'overall_metrics.png')

    def plot_per_class_metrics(self, metrics_per_class, class_names):
        """Generates a grouped bar chart for per-class metrics."""
        metric_types = ['Precision', 'Recall', 'F1-Score']
        normal_metrics = metrics_per_class['normal']
        anomalous_metrics = metrics_per_class['anomalous']
        
        x = np.arange(len(metric_types))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, [normal_metrics[m.lower()] for m in metric_types], width, label=class_names[0], color='cornflowerblue')
        rects2 = ax.bar(x + width/2, [anomalous_metrics[m.lower()] for m in metric_types], width, label=class_names[1], color='salmon')

        ax.set_ylabel('Score')
        ax.set_title('Metrics per Class')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_types)
        ax.legend()
        ax.set_ylim(0, 1.1)
        self._save_plot(fig, 'per_class_metrics.png')

    def plot_resource_usage(self, resource_metrics):
        """Plots CPU, RAM, and GPU usage over time."""
        ts_data = resource_metrics.get('time_series', {})
        timestamps = ts_data.get('timestamps')
        if not timestamps:
            print("No resource time-series data to plot.")
            return

        has_gpu_data = 'gpu_util' in ts_data and any(v is not None for v in ts_data['gpu_util'])
        num_subplots = 2 if has_gpu_data else 1
        
        fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 6 * num_subplots), sharex=True)
        
        # Ensure axs is always a list for consistent indexing
        if num_subplots == 1:
            axs = [axs]

        # CPU and RAM Plot
        ax_cpu = axs[0]
        ax_cpu.plot(timestamps, ts_data.get('cpu_usage', []), color='tab:blue', label='CPU Usage (%)')
        ax_cpu.set_ylabel('CPU Usage (%)', color='tab:blue')
        ax_cpu.tick_params(axis='y', labelcolor='tab:blue')
        ax_cpu.grid(True, axis='y')

        ax_ram = ax_cpu.twinx()
        ax_ram.plot(timestamps, ts_data.get('ram_usage_mb', []), color='tab:green', label='RAM Usage (MB)')
        ax_ram.set_ylabel('RAM Usage (MB)', color='tab:green')
        ax_ram.tick_params(axis='y', labelcolor='tab:green')
        ax_cpu.set_title('CPU and RAM Usage')
        
        # GPU Plot
        if has_gpu_data:
            ax_gpu = axs[1]
            ax_gpu.plot(timestamps, ts_data.get('gpu_util', []), color='tab:red', label='GPU Utilization (%)')
            ax_gpu.set_ylabel('GPU Utilization (%)', color='tab:red')
            ax_gpu.tick_params(axis='y', labelcolor='tab:red')
            ax_gpu.grid(True, axis='y')
            
            ax_gpu_mem = ax_gpu.twinx()
            ax_gpu_mem.plot(timestamps, ts_data.get('gpu_mem_mb', []), color='tab:purple', label='GPU Memory (MB)')
            ax_gpu_mem.set_ylabel('GPU Memory (MB)', color='tab:purple')
            ax_gpu_mem.tick_params(axis='y', labelcolor='tab:purple')
            ax_gpu.set_title('GPU Usage')
            ax_gpu.set_xlabel('Time (seconds)')
        else:
            ax_cpu.set_xlabel('Time (seconds)')

        self._save_plot(fig, 'resource_usage.png')
    
    def get_plot_paths(self):
        """Returns a dictionary of all generated plot paths."""
        return self.plot_paths