import sqlite3
import json
import time
import uuid
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path='logsentinel.db'):
        self.db_path = Path(db_path)
        self.conn = None
        self._connect(); self._create_tables()

    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            print(f"Database connection error: {e}"); raise

    def _create_tables(self):
        if not self.conn: return
        cursor = self.conn.cursor()
        try:
            # FIX: Added run_type column
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                run_type TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                status TEXT NOT NULL,
                model_name TEXT,
                dataset_name TEXT,
                report_path TEXT
            );
            """)
            cursor.execute("CREATE TABLE IF NOT EXISTS hyperparameters (run_id TEXT PRIMARY KEY, params_json TEXT NOT NULL, FOREIGN KEY (run_id) REFERENCES runs (run_id));")
            cursor.execute("CREATE TABLE IF NOT EXISTS performance_metrics (run_id TEXT PRIMARY KEY, metrics_json TEXT NOT NULL, FOREIGN KEY (run_id) REFERENCES runs (run_id));")
            cursor.execute("CREATE TABLE IF NOT EXISTS resource_metrics (run_id TEXT PRIMARY KEY, metrics_json TEXT NOT NULL, FOREIGN KEY (run_id) REFERENCES runs (run_id));")
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
        finally:
            cursor.close()

    def create_new_run(self, run_type, model_name, dataset_name, hyperparameters):
        if not self.conn: return None
        run_id = str(uuid.uuid4())
        start_time = time.time()
        cursor = self.conn.cursor()
        try:
            cursor.execute("INSERT INTO runs (run_id, run_type, start_time, status, model_name, dataset_name) VALUES (?, ?, ?, ?, ?, ?)", (run_id, run_type, start_time, 'STARTED', model_name, dataset_name))
            if hyperparameters:
                cursor.execute("INSERT INTO hyperparameters (run_id, params_json) VALUES (?, ?)", (run_id, json.dumps(hyperparameters)))
            self.conn.commit()
            return run_id
        except sqlite3.Error as e:
            print(f"Error creating new run: {e}"); self.conn.rollback(); return None
        finally:
            cursor.close()

    def save_performance_metrics(self, run_id, metrics_dict):
        # ... (This method remains unchanged) ...
        if not self.conn or not run_id: return
        cursor = self.conn.cursor()
        try:
            cursor.execute("INSERT OR REPLACE INTO performance_metrics (run_id, metrics_json) VALUES (?, ?)",(run_id, json.dumps(metrics_dict))); self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving performance metrics for run {run_id}: {e}"); self.conn.rollback()
        finally: cursor.close()

    def save_resource_metrics(self, run_id, resource_dict):
        # ... (This method remains unchanged) ...
        if not self.conn or not run_id: return
        cursor = self.conn.cursor()
        try:
            cursor.execute("INSERT OR REPLACE INTO resource_metrics (run_id, metrics_json) VALUES (?, ?)", (run_id, json.dumps(resource_dict))); self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving resource metrics for run {run_id}: {e}"); self.conn.rollback()
        finally: cursor.close()

    def update_run_status(self, run_id, status, report_path=None):
        # ... (This method remains unchanged) ...
        if not self.conn or not run_id: return
        end_time = time.time(); cursor = self.conn.cursor()
        try:
            if report_path: cursor.execute("UPDATE runs SET status = ?, end_time = ?, report_path = ? WHERE run_id = ?", (status, end_time, report_path, run_id))
            else: cursor.execute("UPDATE runs SET status = ?, end_time = ? WHERE run_id = ?", (status, end_time, run_id))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error updating run status for {run_id}: {e}"); self.conn.rollback()
        finally: cursor.close()

    def get_all_runs(self):
        # ... (This method remains unchanged) ...
        if not self.conn: return []
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT * FROM runs ORDER BY start_time DESC"); return [dict(run) for run in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting all runs: {e}"); return []
        finally: cursor.close()

    def get_run_details(self, run_id):
        # ... (This method remains unchanged) ...
        if not self.conn or not run_id: return None
        details = {}; cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)); run_data = cursor.fetchone()
            if not run_data: return None
            details['run_info'] = dict(run_data)
            cursor.execute("SELECT params_json FROM hyperparameters WHERE run_id = ?", (run_id,)); hp_data = cursor.fetchone()
            details['hyperparameters'] = json.loads(hp_data['params_json']) if hp_data else {}
            cursor.execute("SELECT metrics_json FROM performance_metrics WHERE run_id = ?", (run_id,)); perf_data = cursor.fetchone()
            details['performance_metrics'] = json.loads(perf_data['metrics_json']) if perf_data else {}
            cursor.execute("SELECT metrics_json FROM resource_metrics WHERE run_id = ?", (run_id,)); res_data = cursor.fetchone()
            details['resource_metrics'] = json.loads(res_data['metrics_json']) if res_data else {}
            return details
        except sqlite3.Error as e:
            print(f"Error getting details for run {run_id}: {e}"); return None
        finally: cursor.close()
    
    def close(self):
        if self.conn: self.conn.close(); self.conn = None