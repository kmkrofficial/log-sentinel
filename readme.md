# LogSentinel: Anomaly Detection using Llama 3.2 3B

LogSentinel is a system designed for detecting anomalies in sequences of system log messages. It leverages a combination of pre-trained language models, specifically BERT and Llama 3.2 3B, to classify log sequences as either "Normal" or "Anomalous".

## Architecture

LogSentinel employs a multi-stage architecture to process and classify log sequences:

1.  **Input:** The system takes sequences of log messages as input. Each sequence represents a window of activity (e.g., logs within a specific time frame or session). Each message within the sequence is treated as a string.

2.  **BERT Encoder (`bert-base-uncased`):**
    *   **Role:** Encodes the semantic meaning of *individual* log messages within a sequence.
    *   **Process:** Each log message string is passed through the BERT tokenizer and then the BERT model.
    *   **Output:** For each log message, the `pooler_output` (an embedding representing the `[CLS]` token, often used as a sentence-level representation) is extracted. This results in a sequence of embeddings, one for each log message in the input sequence.

3.  **MLP Projector:**
    *   **Role:** Acts as a bridge between the BERT embedding space and the Llama embedding space. It transforms the BERT embeddings into a format suitable for input to the Llama model. This is necessary because BERT and Llama have different hidden state dimensions and may represent information differently.
    *   **Structure:** A simple Multi-Layer Perceptron (MLP) consisting of:
        *   Linear Layer (BERT hidden size -> Llama hidden size)
        *   GELU Activation Function (Introduces non-linearity)
        *   Linear Layer (Llama hidden size -> Llama hidden size)
    *   **Process:** The sequence of BERT embeddings is passed through the projector, resulting in a new sequence of embeddings with the dimensionality expected by Llama.

4.  **Llama 3.2 3B Model:**
    *   **Role:** Processes the *entire sequence* of projected log embeddings to capture contextual information and sequence-level patterns. It acts as a powerful feature extractor for the sequence classification task.
    *   **Configuration:**
        *   Uses the Llama 3.2 3B parameter model.
        *   Loaded with **8-bit quantization** (`BitsAndBytesConfig`) to significantly reduce memory requirements, making it feasible to run on GPUs with limited VRAM (e.g., 16GB).
        *   Utilizes **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA (Low-Rank Adaptation)**. This means only small adapter layers are added and trained during fine-tuning, keeping most of the large Llama model frozen, further reducing memory usage and training time compared to full fine-tuning.
    *   **Process:**
        *   A simple instruction prompt ("Below is a sequence of system log messages:") is prepended to the sequence of projected log embeddings.
        *   This combined sequence of embeddings is fed into the Llama model.
        *   The model processes the sequence, attending to relationships between the log embeddings.
        *   **Crucially, it is *not* used for text generation here.** Its purpose is to output hidden states that encode information about the input sequence.

5.  **Classifier Head:**
    *   **Role:** Makes the final binary classification decision (Normal vs. Anomalous).
    *   **Structure:** A single Linear layer.
    *   **Process:**
        *   Takes the final hidden state corresponding to the *last token* of the sequence from the Llama model's output. This hidden state is assumed to summarize the relevant information from the entire sequence processed by Llama.
        *   Passes this hidden state through the linear layer.
        *   **Output:** Produces two raw output values (logits), representing the model's confidence score for the sequence being "Normal" (class 0) or "Anomalous" (class 1). During evaluation, an `argmax` function determines the final predicted class.

6.  **Training & Loss:**
    *   During training, the logits from the classifier head are compared against the true labels (0 or 1) using **Cross-Entropy Loss**.
    *   Gradients are calculated and used to update the weights of the *trainable* components (Projector, Classifier, LoRA adapters) via backpropagation.

**(Optional: Insert a visual diagram here if possible)**
```
[Log Sequence] -> For each Log Message: ([Log String] -> BERT Tokenizer -> BERT Model -> [CLS] Embedding)
              -> Sequence of [CLS] Embeddings
              -> MLP Projector
              -> Sequence of Projected Embeddings
              -> Prepend Instruction Embedding
              -> Llama 3.2 3B (Quantized + LoRA)
              -> Final Hidden State (Last Token)
              -> Classifier Head (Linear Layer)
              -> [Logits (Normal, Anomalous)] -> ArgMax -> Final Prediction
```

## Setup Instructions (First Time)

Follow these steps to set up the LogSentinel environment:

1.  **Prerequisites:**
    *   Python 3.8+
    *   `pip` package manager
    *   Git
    *   **NVIDIA GPU:** Required for running the model efficiently. Ensure you have compatible NVIDIA drivers installed.
    *   **CUDA Toolkit:** Install a version compatible with your drivers and PyTorch.

2.  **Clone Repository:**
    ```bash
    git clone https://github.com/kmkrofficial/LogSentinel-3b.git
    cd LogSentinel-3b
    ```

3.  **Create Virtual Environment (Recommended):**
    *   Using `venv`:
        ```bash
        python -m venv venv
        # Activate (Linux/macOS)
        source venv/bin/activate
        # Activate (Windows - Command Prompt)
        venv\Scripts\activate.bat
        # Activate (Windows - PowerShell)
        venv\Scripts\Activate.ps1
        ```
    *   Using `conda`:
        ```bash
        conda create -n logsentinel python=3.9 # Or your preferred Python version
        conda activate logsentinel
        ```

4.  **Install Dependencies:**
    *   Create a `requirements.txt` file with the following content (adjust versions if necessary, especially for `torch` based on your CUDA version):
        ```txt
        torch
        transformers
        pandas
        numpy
        scikit-learn
        tqdm
        peft
        bitsandbytes
        matplotlib
        seaborn
        psutil
        # Use nvidia-ml-py OR pynvml, pynvml is often preferred
        # nvidia-ml-py # For GPU monitoring (alternative)
        pynvml       # For GPU monitoring (used in current eval.py)
        accelerate   # Often needed by transformers/peft
        ```
    *   Install the requirements:
        ```bash
        pip install -r requirements.txt
        ```
        *(Note: `bitsandbytes` might require specific build steps on some systems, especially Windows. Check its official documentation if you encounter issues.)*

5.  **Download Pre-trained Models:**
    *   The code expects the BERT and Llama models to be available locally.
    *   **BERT:** Download `bert-base-uncased`. You can often let `transformers` download it automatically on first use, but for explicit control, download it using tools like `git lfs` or manually from the Hugging Face Hub.
    *   **Llama:** Download the **Llama 3.2 3B** model weights from their official source (e.g., Meta, Hugging Face). This will likely be a large download.
    *   **Update Paths:** Note the paths where you saved these models. Open `train.py` and `eval.py` and update the `Bert_path` and `Llama_path` variables to point to the correct directories on your system.
        ```python
        # Example paths in train.py / eval.py - CHANGE THESE
        Bert_path = r"/path/to/your/models/bert-base-uncased"
        Llama_path = r"/path/to/your/models/Llama-3.2-3B"
        ```

6.  **Prepare Dataset:**
    *   Ensure your training and testing data are in CSV format.
    *   Place `train.csv` and `test.csv` (or just `train.csv` if `test.csv` is missing, as `eval.py` has fallback logic) in a `dataset` directory relative to the scripts, or update the paths in the code.
    *   The CSV files must contain at least two columns:
        *   `Content`: Contains the log message sequences. Log messages within a sequence should be delimited by ` ;-; ` (space, semicolon, hyphen, semicolon, space).
        *   `Label`: Contains the ground truth label, either `0` (Normal) or `1` (Anomalous).
    *   **Update Paths:** Check the `data_path` variable in `train.py` and the `base_data_dir`, `test_data_filename`, `train_data_filename` variables in `eval.py` and adjust them if your dataset location differs.
        ```python
        # Example path in train.py - CHANGE IF NEEDED
        data_path = r'/path/to/your/dataset/train.csv'

        # Example paths in eval.py - CHANGE IF NEEDED
        base_data_dir = r'/path/to/your/dataset'
        test_data_filename = 'test.csv'
        train_data_filename = 'train.csv'
        ```

## Running the Code

1.  **Training:**
    *   Navigate to the repository directory in your terminal (with the virtual environment activated).
    *   Run the training script:
        ```bash
        python train.py
        ```
    *   This will:
        *   Load the training data (`train.csv`).
        *   Initialize the `LogLLM` model.
        *   Perform multi-phase training (Projector -> Classifier -> Proj+Cls -> All trainable).
        *   Print training progress (loss, accuracy).
        *   Save the fine-tuned projector (`projector.pt`), classifier (`classifier.pt`), and LoRA adapters (`Llama_ft/`, potentially `Bert_ft/`) into a directory named `ft_model_cls_{dataset_name}` (e.g., `ft_model_cls_BGL`).

2.  **Evaluation:**
    *   Ensure training is complete and the `ft_model_cls_{dataset_name}` directory exists with the saved components.
    *   Run the evaluation script:
        ```bash
        python eval.py
        ```
    *   This will:
        *   Load the test data (`test.csv` or fallback to `train.csv`).
        *   Load the fine-tuned `LogLLM` model components.
        *   Measure baseline resource usage.
        *   Run inference on the evaluation dataset, measuring performance and resource usage per batch.
        *   Print detailed classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
        *   Print average and peak/95th percentile resource usage statistics (CPU, GPU, RAM, Time), adjusted for baseline load.
        *   Save various performance and resource usage plots (confusion matrix, metrics bars, distribution comparison, time/resource graphs) into the `visualizations` directory.

## Hyperparameter Rationale

The choice of hyperparameters influences model performance, training stability, and resource consumption. Here's the reasoning behind some key parameters used in `train.py` and `eval.py`:

*   **Model Paths (`Bert_path`, `Llama_path`):** Point to the specific pre-trained models used as the foundation. `bert-base-uncased` is a standard choice for general text encoding. `Llama-3.2-3B` was chosen as a powerful LLM manageable on moderate hardware.
*   **8-bit Quantization (`BitsAndBytesConfig`):** **Crucial for memory.** Loading the 3.2 billion parameter Llama model in full precision (FP32/FP16) requires significant VRAM. 8-bit quantization drastically reduces the memory footprint, making training and inference feasible on GPUs like a 16GB VRAM card, with often minimal impact on performance for fine-tuning tasks.
*   **LoRA (`r=8`, `lora_alpha=16`, `target_modules=["q_proj", "v_proj"]`):** Parameter-Efficient Fine-Tuning (PEFT) method. Instead of training all 3.2B Llama parameters, LoRA introduces small, low-rank matrices ("adapters") into specific layers (here, query and value projections in attention). Only these adapters (~ millions of parameters) are trained.
    *   `r=8`: Rank of the decomposition, controls the number of trainable parameters in the adapters. Lower `r` means fewer parameters, faster training, less memory, but potentially less capacity to adapt. `r=8` is a common starting point.
    *   `lora_alpha=16`: Scaling factor for the LoRA adapters. Often set to `r` or `2*r`. It influences how much impact the adapters have relative to the original weights.
    *   `target_modules`: Specifies which layers within Llama receive the LoRA adapters. Targeting attention mechanism layers (`q_proj`, `v_proj`) is standard practice.
*   **Training Phases & Learning Rates (`n_epochs_phaseX`, `lr_phaseX`):** A staged approach is used:
    1.  **Projector Only:** Train the bridge between BERT and Llama first (LR `1e-4`). Allows the projector to learn the initial mapping without interference.
    2.  **Classifier Only:** Train the final decision layer (LR `5e-4`). Can use a higher LR as it's trained from scratch.
    3.  **Projector + Classifier:** Train both together (LR `7e-5`). Refines their interaction.
    4.  **All Trainable:** Fine-tune Projector, Classifier, and LoRA adapters (LR `1e-5`). A very low LR is used for fine-tuning the LLM adapters to avoid disrupting the pre-trained knowledge significantly.
    *   This phased approach aims for stability by allowing different components to learn progressively. The specific epoch counts and LRs are empirical starting points and may require tuning based on the dataset.
*   **Batch Sizes (`batch_size=8`, `micro_batch_size=4`):**
    *   `micro_batch_size=4`: The number of samples processed by the GPU in one forward/backward pass. Limited by VRAM. Chosen to fit the 3.2B quantized model + activations + gradients on ~16GB VRAM.
    *   `batch_size=8`: The effective batch size used for gradient updates.
    *   `gradient_accumulation_steps = batch_size // micro_batch_size = 2`: Gradients from 2 micro-batches are accumulated *before* the optimizer updates the weights. This simulates a larger batch size (`8`) using less memory, leading to potentially more stable gradients.
*   **Optimizer (`AdamW`):** A standard, effective optimizer for training deep learning models, particularly transformers. Includes weight decay correction. Betas (`0.9, 0.98`) and epsilon (`1e-9`) are common default values.
*   **Scheduler (`get_linear_schedule_with_warmup`):**
    *   **Warmup:** Gradually increases the learning rate from 0 to the target LR over the first few steps (5% of total steps). Prevents large, destabilizing updates early in training when weights are randomly initialized.
    *   **Decay:** Linearly decreases the learning rate from the target LR down to 0 over the remaining training steps. Helps the model converge more finely towards the end of training.
*   **Loss Function (`nn.CrossEntropyLoss`):** The standard loss function for multi-class (including binary) classification problems when the model outputs raw logits. It combines LogSoftmax and Negative Log-Likelihood Loss.
*   **Oversampling (`min_less_portion=0.5`):** Addresses class imbalance in the training data. If the minority class (e.g., Anomalous) makes up less than 50% of the data, samples from that class are duplicated during training batch creation until the target proportion (50%) is reached in the training batches. This prevents the model from becoming biased towards predicting only the majority class.
*   **Sequence Lengths (`max_content_len=100`, `max_seq_len=128`):**
    *   `max_content_len=100`: Maximum number of tokens BERT processes for a *single* log message. Truncates long messages.
    *   `max_seq_len=128`: Maximum number of *log messages* (after BERT encoding and projection) processed by Llama in a single sequence. Truncates long sequences.
    *   These values are a trade-off: longer sequences provide more context but increase computational cost and memory usage significantly, especially for Llama. These values were likely chosen to balance context with resource constraints.
*   **Evaluation Batch Size (`batch_size=16` in `eval.py`):** Can often be larger than the training `micro_batch_size` because gradients are not computed or stored during evaluation (`torch.no_grad()`), freeing up VRAM.

## Output and Visualizations

Upon successful execution, the scripts produce:

*   **Console Output:** Detailed logs during training (epoch, loss, accuracy, LR) and evaluation (metrics, resource usage stats).
*   **Fine-tuned Model (`ft_model_cls_{dataset_name}/`):** Contains the saved weights/adapters for the projector, classifier, and LoRA layers.
*   **Visualizations (`visualizations/` directory):**
    *   `confusion_matrix_*.png`: Heatmap showing True Positives, True Negatives, False Positives, False Negatives.
    *   `overall_metrics_*.png`: Bar chart of overall Accuracy, Precision, Recall, F1-Score.
    *   `per_class_metrics_*.png`: Grouped bar chart comparing Precision, Recall, F1-Score for Normal vs. Anomalous classes.
    *   `distribution_comparison_*.png`: Bar chart comparing the number of samples in the ground truth vs. predicted labels for each class.
    *   `inference_time_*.png`: Line plot showing inference time per batch.
    *   `resource_usage_*.png`: Line plots showing CPU/RAM and GPU Utilization/Memory usage per batch during inference.