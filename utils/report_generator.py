from fpdf import FPDF
from pathlib import Path
import json
import time

class ReportGenerator(FPDF):
    """
    Generates a PDF report for a completed LogSentinel run.
    Inherits from FPDF to create a custom header and layout.
    """
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, '🛡️ LogSentinel Training Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, data):
        self.set_font('Helvetica', '', 10)
        if isinstance(data, dict):
            for key, value in data.items():
                self.multi_cell(0, 6, f"{key}: {json.dumps(value, indent=4)}")
        elif isinstance(data, str):
            self.multi_cell(0, 6, data)
        self.ln()

    def add_metrics_table(self, metrics_dict):
        self.set_font('Helvetica', 'B', 10)
        # Headers
        self.cell(40, 7, 'Metric', 1)
        self.cell(40, 7, 'Precision', 1)
        self.cell(40, 7, 'Recall', 1)
        self.cell(40, 7, 'F1-Score', 1)
        self.ln()
        # Data
        self.set_font('Helvetica', '', 10)
        for class_name, metrics in metrics_dict.items():
            if 'precision' in metrics:
                self.cell(40, 7, class_name.capitalize(), 1)
                self.cell(40, 7, f"{metrics['precision']:.4f}", 1)
                self.cell(40, 7, f"{metrics['recall']:.4f}", 1)
                self.cell(40, 7, f"{metrics.get('f1_score', metrics.get('f1')):.4f}", 1)
                self.ln()


def create_report(run_details: dict, output_dir: Path) -> str:
    """
    Orchestrates the creation of the PDF report from run details.
    
    Args:
        run_details (dict): A dictionary containing all info for a run.
        output_dir (Path): The directory where the report will be saved.
        
    Returns:
        str: The path to the generated PDF file.
    """
    run_info = run_details.get('run_info', {})
    hyperparams = run_details.get('hyperparameters', {})
    perf_metrics = run_details.get('performance_metrics', {})
    
    pdf = ReportGenerator('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # --- Run Summary ---
    pdf.chapter_title('1. Run Summary')
    summary_body = (
        f"Run ID: {run_info.get('run_id')}\n"
        f"Model: {run_info.get('model_name')}\n"
        f"Dataset: {run_info.get('dataset_name')}\n"
        f"Status: {run_info.get('status')}\n"
        f"Start Time: {time.ctime(run_info.get('start_time'))}\n"
        f"End Time: {time.ctime(run_info.get('end_time'))}\n"
        f"Duration: {run_info.get('end_time', 0) - run_info.get('start_time', 0):.2f} seconds"
    )
    pdf.chapter_body(summary_body)

    # --- Hyperparameters ---
    pdf.chapter_title('2. Hyperparameters')
    pdf.chapter_body(hyperparams)
    
    # --- Performance Metrics ---
    pdf.chapter_title('3. Performance Metrics')
    pdf.add_metrics_table({
        'overall': perf_metrics.get('overall', {}),
        'normal': perf_metrics.get('per_class', {}).get('normal', {}),
        'anomalous': perf_metrics.get('per_class', {}).get('anomalous', {})
    })
    pdf.ln(10)

    # --- Visualizations ---
    pdf.add_page()
    pdf.chapter_title('4. Visualizations')
    
    # Get plot paths from the directory where they were saved
    report_path = Path(run_info.get('report_path', ''))
    
    if report_path.is_dir():
        image_files = {
            'confusion_matrix': report_path / 'confusion_matrix.png',
            'overall_metrics': report_path / 'overall_metrics.png',
            'resource_usage': report_path / 'resource_usage.png'
        }
        
        for title, img_path in image_files.items():
            if img_path.exists():
                pdf.set_font('Helvetica', 'B', 11)
                pdf.cell(0, 10, title.replace('_', ' ').title(), 0, 1)
                pdf.image(str(img_path), w=180) # A4 width is 210mm, 180 leaves margins
                pdf.ln(5)
    
    # --- Save PDF ---
    pdf_output_path = output_dir / "training_report.pdf"
    pdf.output(str(pdf_output_path))
    print(f"Report generated successfully: {pdf_output_path}")
    
    return str(pdf_output_path)