import sqlite3
import json
import time
import uuid
from pathlib import Path

class DatabaseManager:
    """
    Handles all database operations for LogSentinel, including creating tables,
    inserting and retrieving run data, metrics, and configurations.
    """
    def __init__(self, db_path='logsentinel.db'):
        """
        Initializes the DatabaseManager and connects to the SQLite database.
        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establishes a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise

    def _create_tables(self):
        """
        Creates the necessary database tables if they do not already exist.
        """
        if not self.conn:
            return

        cursor = self.conn.cursor()
        try:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                status TEXT NOT NULL,
                model_name TEXT,
                dataset_name TEXT,
                report_path TEXT
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hyperparameters (
                run_id TEXT PRIMARY KEY,
                params_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                run_id TEXT PRIMARY KEY,
                metrics_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            );
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS resource_metrics (
                run_id TEXT PRIMARY KEY,
                metrics_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            );
            """)

            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
        finally:
            cursor.close()

    # FIX: Use uuid to generate a random run_id for every new run.
    def _generate_run_id(self):
        """Generates a unique random identifier for a run."""
        return str(uuid.uuid4())

    def create_new_run(self, model_name, dataset_name, hyperparameters):
        """
        Creates a new record for a training run.
        Generates a unique random run_id.
        """
        if not self.conn:
            return None

        run_id = self._generate_run_id()
        start_time = time.time()
        
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO runs (run_id, start_time, status, model_name, dataset_name) VALUES (?, ?, ?, ?, ?)",
                (run_id, start_time, 'STARTED', model_name, dataset_name)
            )
            cursor.execute(
                "INSERT INTO hyperparameters (run_id, params_json) VALUES (?, ?)",
                (run_id, json.dumps(hyperparameters))
            )
            self.conn.commit()
            return run_id
        except sqlite3.Error as e:
            print(f"Error creating new run: {e}")
            self.conn.rollback()
            return None
        finally:
            cursor.close()

    def save_performance_metrics(self, run_id, metrics_dict):
        if not self.conn or not run_id: return
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO performance_metrics (run_id, metrics_json) VALUES (?, ?)",
                (run_id, json.dumps(metrics_dict))
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving performance metrics for run {run_id}: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
            
    def save_resource_metrics(self, run_id, resource_dict):
        if not self.conn or not run_id: return
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO resource_metrics (run_id, metrics_json) VALUES (?, ?)",
                (run_id, json.dumps(resource_dict))
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving resource metrics for run {run_id}: {e}")
            self.conn.rollback()
        finally:
            cursor.close()

    def update_run_status(self, run_id, status, report_path=None):
        if not self.conn or not run_id: return
        end_time = time.time()
        cursor = self.conn.cursor()
        try:
            if report_path:
                cursor.execute(
                    "UPDATE runs SET status = ?, end_time = ?, report_path = ? WHERE run_id = ?",
                    (status, end_time, report_path, run_id)
                )
            else:
                cursor.execute(
                    "UPDATE runs SET status = ?, end_time = ? WHERE run_id = ?",
                    (status, end_time, run_id)
                )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error updating run status for {run_id}: {e}")
            self.conn.rollback()
        finally:
            cursor.close()
            
    def get_all_runs(self):
        if not self.conn: return []
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT * FROM runs ORDER BY start_time DESC")
            return [dict(run) for run in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting all runs: {e}")
            return []
        finally:
            cursor.close()

    def get_run_details(self, run_id):
        if not self.conn or not run_id: return None
        details = {}
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            run_data = cursor.fetchone()
            if not run_data: return None
            details['run_info'] = dict(run_data)

            cursor.execute("SELECT params_json FROM hyperparameters WHERE run_id = ?", (run_id,))
            hp_data = cursor.fetchone()
            details['hyperparameters'] = json.loads(hp_data['params_json']) if hp_data else {}

            cursor.execute("SELECT metrics_json FROM performance_metrics WHERE run_id = ?", (run_id,))
            perf_data = cursor.fetchone()
            details['performance_metrics'] = json.loads(perf_data['metrics_json']) if perf_data else {}
            
            cursor.execute("SELECT metrics_json FROM resource_metrics WHERE run_id = ?", (run_id,))
            res_data = cursor.fetchone()
            details['resource_metrics'] = json.loads(res_data['metrics_json']) if res_data else {}

            return details
        except sqlite3.Error as e:
            print(f"Error getting details for run {run_id}: {e}")
            return None
        finally:
            cursor.close()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None