# LogSentinel 🛡️

LogSentinel is a comprehensive, UI-driven toolkit for training, evaluating, and deploying log anomaly detection models. Inspired by the [LogLLM research paper](https://arxiv.org/abs/2411.08561), this application provides a complete lifecycle for building and using edge-optimized models based on large language model (LLM) architectures.

The entire application is managed through a user-friendly Streamlit interface, allowing users to run complex machine learning pipelines without writing any code.

## Key Features

-   **Train & Evaluate:** Launch new training runs with custom Hugging Face or local models. Configure all hyperparameters, monitor live progress with real-time metrics, and receive a full evaluation report upon completion.
-   **Evaluate & Predict:** Use a previously trained model for:
    -   **Testing:** Run a full evaluation against a new dataset with ground-truth labels to get detailed performance metrics (Accuracy, Precision, Recall, F1-Score).
    -   **Inference:** Perform fast, batch predictions on new log data without labels.
-   **Run History:** A persistent, browsable history of all past runs stored in an SQLite database. Review detailed performance metrics, resource consumption charts (CPU, RAM, GPU, VRAM), and generated visualizations for every run.
-   **Edge-Optimized Architecture:** Employs a memory-efficient, two-phase pipeline for inference and testing, ensuring that even large models can run on resource-constrained hardware by loading only one model into VRAM at a time.
-   **Resource Monitoring:** Every run is automatically profiled for CPU, RAM, and NVIDIA GPU resource utilization, providing deep insights into the performance characteristics of each model and configuration.

## Core Technologies

LogSentinel integrates several state-of-the-art technologies to achieve its performance and efficiency:

-   **PyTorch:** The core deep learning framework for all model training and inference.
-   **Streamlit:** The framework used to build the entire interactive graphical user interface.
-.  **Hugging Face `transformers`:** For loading and using state-of-the-art LLMs (e.g., Llama, BERT).
-   **PEFT & QLoRA:** For highly memory-efficient fine-tuning of large models using Low-Rank Adaptation on quantized models.
-   **`bitsandbytes`:** For 4-bit and 8-bit model quantization, drastically reducing the VRAM footprint of the models.
-   **Flash Attention 2 (Optional):** Utilizes `flash-attn` if available for a significant boost in speed and memory efficiency during attention calculations. The application gracefully falls back to PyTorch's native `sdpa` if it's not installed.

## Setup and Installation

Follow these steps to set up the LogSentinel environment on your machine.

### Prerequisites

-   **Git:** To clone the repository.
-   **Python:** Version 3.11 is recommended.
-   **NVIDIA GPU:** An NVIDIA GPU with CUDA support is required for training and accelerated inference.
-   **NVIDIA CUDA Toolkit:** While the PyTorch installation handles the runtime libraries, ensure your **NVIDIA driver is up to date**.

### 1. Clone the Repository

```bash
git clone https://github.com/kmkrofficial/log-sentinel
cd log-sentinel
```

### 2. Create a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv .venv
```

**Activate the environment:**
-   On Windows:
    ```cmd
    .\.venv\Scripts\activate
    ```
-   On Linux/macOS:
    ```bash
    source .venv/bin/activate
    ```

### 3. Install PyTorch

Install the PyTorch version that matches your system's CUDA capabilities. For best compatibility with `flash-attn`, we recommend **PyTorch 2.3.1 with CUDA 12.1**.

-   **On Windows & Linux:**
    ```bash
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```

### 4. Install `flash-attn` (Optional but Highly Recommended)

This package provides a significant performance boost.

#### Path A (Recommended): Install a Pre-compiled Wheel

This is the easiest and most reliable method, as it avoids local compilation.

1.  **Verify your environment.** Run this Python script to confirm your Python, PyTorch, and CUDA versions.
    ```python
    import torch
    print(f"Python Version: {'.'.join(str(v) for v in __import__('sys').version_info[:2])}")
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    ```
    You should see output for Python `3.11`, PyTorch `2.3.1...`, and CUDA `12.1`.

2.  **Download the correct `.whl` file** from the [flash-attn GitHub releases page](https://github.com/Dao-AILab/flash-attention/releases). Find the file that exactly matches your environment (e.g., `...cp311-cu121-torch2.3.1...`).

3.  **Install the downloaded file** using pip (replace the path with your actual download path):
    ```bash
    pip install "C:\path\to\downloaded\flash_attn-2.5.8-cp311-cp311-win_amd64.whl"
    ```

#### Path B (Advanced): Compile from Source

This is much more difficult and prone to errors. Only attempt this if a pre-compiled wheel is not available.

-   **On Windows:**
    1.  Install [**Visual Studio 2022** with the **"Desktop development with C++"** workload](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    2.  Open the **"x64 Native Tools Command Prompt for VS 2022"** as an administrator.
    3.  Navigate to your project directory and activate your virtual environment inside this special command prompt.
    4.  Set the required environment variable: `set DISTUTILS_USE_SDK=1`
    5.  Run the install command: `pip install flash-attn --no-build-isolation`

-   **On Linux:**
    1.  Install the necessary build tools: `sudo apt-get update && sudo apt-get install build-essential`
    2.  Run the install command: `pip install flash-attn --no-build-isolation`

### 5. Install Remaining Dependencies

Once PyTorch and `flash-attn` are set up, install the rest of the required packages.

```bash
pip install -r requirements.txt
```

## Running the Application

After the installation is complete, you can launch the Streamlit application with the following command:

```bash
streamlit run app.py
```

This will open the LogSentinel interface in your web browser.

## Project Structure

```
/logsentinel/
├── pages/                # Contains the Streamlit pages for each app feature.
├── engine/               # Core logic controllers for training and inference.
├── utils/                # Helper modules for data, database, monitoring, etc.
├── datasets/             # Place your datasets here (e.g., /BGL/train.csv).
├── models/               # Caching directory for downloaded Hugging Face models.
├── reports/              # Output directory for run reports, plots, and models.
├── app.py                # The main Streamlit landing page.
├── config.py             # Central configuration for paths and hyperparameters.
└── logsentinel.db        # The SQLite database for storing run history.
```