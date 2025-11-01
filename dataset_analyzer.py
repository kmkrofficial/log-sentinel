import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import sys
import numpy as np

# Add parent dir to path to import config
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from config import DATA_DIR, get_hyperparameters
except ImportError:
    print("Error: Could not import config.py.")
    print("Please make sure this script is in the root directory of your project.")
    sys.exit(1)

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

def analyze_file(file_path: Path):
    stats = {
        "total_sequences": 0,
        "normal_sequences (label 0)": 0,
        "anomalous_sequences (label 1)": 0,
        "total_log_messages": 0,
        "avg_log_messages_per_sequence": 0
    }
    
    try:
        for chunk in tqdm(pd.read_csv(file_path, chunksize=100000), desc=f"Analyzing {file_path.name}"):
            stats["total_sequences"] += len(chunk)
            
            label_counts = chunk['Label'].value_counts()
            stats["normal_sequences (label 0)"] += label_counts.get(0, 0)
            stats["anomalous_sequences (label 1)"] += label_counts.get(1, 0)
            
            stats["total_log_messages"] += chunk['Content'].str.count(';').sum() + len(chunk)

        if stats["total_sequences"] > 0:
            stats["avg_log_messages_per_sequence"] = round(
                stats["total_log_messages"] / stats["total_sequences"], 2
            )
            
    except FileNotFoundError:
        print(f"Warning: {file_path.name} not found. Skipping.")
        return None
    except Exception as e:
        print(f"Error analyzing {file_path.name}: {e}")
        return None
        
    return stats

def calculate_projected_epoch_size(train_stats, dataset_name):
    try:
        hp = get_hyperparameters(dataset_name)
        min_less_portion = hp.get('min_less_portion', 0.5)
        
        normal_count = train_stats.get("normal_sequences (label 0)", 0)
        anomaly_count = train_stats.get("anomalous_sequences (label 1)", 0)
        
        if normal_count == 0 or anomaly_count == 0:
            return train_stats.get("total_sequences", 0)

        majority_count = max(normal_count, anomaly_count)
        less_count = min(normal_count, anomaly_count)
        
        oversampled_less_count = int(majority_count * min_less_portion)
        num_to_add = max(0, oversampled_less_count - less_count)
        
        projected_epoch_size = majority_count + less_count + num_to_add
        return projected_epoch_size
        
    except Exception as e:
        print(f"Error calculating epoch size for {dataset_name}: {e}")
        return 0

def main():
    all_stats = {}
    
    if not DATA_DIR.exists():
        print(f"Error: Datasets directory not found at {DATA_DIR}")
        return

    dataset_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if not dataset_dirs:
        print(f"No dataset folders found in {DATA_DIR}.")
        return

    print(f"Found {len(dataset_dirs)} dataset(s). Analyzing...")

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        all_stats[dataset_name] = {}
        print(f"\n--- Analyzing Dataset: {dataset_name} ---")

        train_stats = analyze_file(dataset_dir / "train.csv")
        if train_stats:
            all_stats[dataset_name]["train.csv"] = train_stats
            
            epoch_size = calculate_projected_epoch_size(train_stats, dataset_name)
            all_stats[dataset_name]["train.csv"]["projected_samples_per_epoch"] = epoch_size

        val_stats = analyze_file(dataset_dir / "validation.csv")
        if val_stats:
            all_stats[dataset_name]["validation.csv"] = val_stats
            
        test_stats = analyze_file(dataset_dir / "test.csv")
        if test_stats:
            all_stats[dataset_name]["test.csv"] = test_stats

    print("\n\n--- Analysis Complete ---")
    print(json.dumps(all_stats, indent=2, cls=NumpyJSONEncoder))

if __name__ == "__main__":
    main()