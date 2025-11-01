import sqlite3
import json
import pandas as pd
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                nickname TEXT,
                run_type TEXT,
                model_name TEXT,
                dataset_name TEXT,
                status TEXT DEFAULT 'PENDING',
                report_path TEXT,
                
                total_run_time_sec REAL,
                training_time_sec REAL,
                testing_time_sec REAL,
                
                accuracy REAL,
                precision REAL,
                f1_score REAL,
                recall REAL,
                
                avg_ram_usage_gb REAL,
                peak_95_ram_usage_gb REAL,
                avg_gpu_vram_gb REAL,
                peak_95_gpu_vram_gb REAL,
                
                hyperparameters TEXT
            )
            ''')
            conn.commit()

    def create_new_run(self, run_type, model_name, dataset_name, hyperparameters, nickname):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            hp_json = json.dumps(hyperparameters)
            cursor.execute('''
            INSERT INTO runs (run_type, model_name, dataset_name, hyperparameters, nickname, status)
            VALUES (?, ?, ?, ?, ?, 'RUNNING')
            ''', (run_type, model_name, dataset_name, hp_json, nickname))
            conn.commit()
            return cursor.lastrowid

    def update_run_status(self, run_id, status, report_path=None):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE runs SET status = ?, report_path = ?
            WHERE id = ?
            ''', (status, report_path, run_id))
            conn.commit()

    def save_final_metrics(self, run_id, metrics):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE runs SET
                total_run_time_sec = ?,
                training_time_sec = ?,
                testing_time_sec = ?,
                accuracy = ?,
                precision = ?,
                f1_score = ?,
                recall = ?,
                avg_ram_usage_gb = ?,
                peak_95_ram_usage_gb = ?,
                avg_gpu_vram_gb = ?,
                peak_95_gpu_vram_gb = ?
            WHERE id = ?
            ''', (
                metrics.get('total_run_time_sec'),
                metrics.get('training_time_sec'),
                metrics.get('testing_time_sec'),
                metrics.get('accuracy'),
                metrics.get('precision'),
                metrics.get('f1_score'),
                metrics.get('recall'),
                metrics.get('avg_ram_usage_gb'),
                metrics.get('peak_95_ram_usage_gb'),
                metrics.get('avg_gpu_vram_gb'),
                metrics.get('peak_95_gpu_vram_gb'),
                run_id
            ))
            conn.commit()

    def get_all_runs(self):
        with self._get_conn() as conn:
            query = """
                SELECT 
                    id, start_time, nickname, dataset_name, status,
                    total_run_time_sec, f1_score, precision, recall, accuracy
                FROM runs 
                ORDER BY start_time DESC
            """
            df = pd.read_sql_query(query, conn)
            return df

    def get_run_details(self, run_id):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                details = dict(row)
                details['hyperparameters'] = json.loads(details.get('hyperparameters', '{}'))
                return details
            return None

    def get_runs_by_nickname_prefix(self, prefix):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nickname FROM runs WHERE nickname LIKE ?", (f"{prefix}%",))
            return [row[0] for row in cursor.fetchall()]