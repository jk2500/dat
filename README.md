# Phase 1: DivPO Training for Diverse Noun Generation

This project implements the first phase of training a language model using Diversity-enhanced Preference Optimization (DivPO). The goal is to fine-tune a causal language model (e.g., Qwen, Phi) to generate diverse, common, single-word English nouns based on a simple prompt, avoiding proper nouns.

This script focuses specifically on the DivPO training process, where the model learns to prefer diverse and high-quality (common, non-proper noun) generations over less diverse or lower-quality ones.

## Features

*   **DivPO Implementation:** Uses a custom `DivPODPOTrainer` built on top of the TRL library's `DPOTrainer`.
*   **On-the-Fly Pair Generation:** Generates candidate words, scores them for quality and diversity, selects chosen/rejected pairs, and calculates the DPO loss within each training step, requiring only prompts as input dataset.
*   **Quality Filtering:** Uses spaCy for POS tagging and wordfreq for commonality checks to ensure generated words are common, single-word, non-proper nouns.
*   **Diversity Scoring:** Leverages sentence-transformers to calculate semantic diversity between generated candidates.
*   **Configurable:** Allows easy configuration of the base model, hyperparameters (K samples, DPO beta, learning rate, etc.), output directories, and embedding models.
*   **Device Compatibility:** Designed to run on CPU or Apple Silicon (MPS), with checks and logging for device selection. CUDA is possible but not the primary focus of the current setup.
*   **Automated NLP Data Download:** Includes checks and downloads for necessary NLTK and spaCy resources on the first run.

## Setup

1.  **Prerequisites:**
    *   Python 3.9+
    *   `pip` or `conda` for package management

2.  **Clone Repository (Optional):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    or using conda:
    ```bash
    conda create -n divpo_env python=3.10 # Or your preferred version
    conda activate divpo_env
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes: `torch`, `transformers`, `trl`, `datasets`, `sentence-transformers`, `spacy`, `nltk`, `wordfreq`, `accelerate`, `bitsandbytes` (optional))*

5.  **NLP Data Download:**
    The script will attempt to download required NLTK data (`words`, `averaged_perceptron_tagger`) and the spaCy model (`en_core_web_sm`) automatically when first run. Ensure you have an internet connection.

## Configuration

Key configuration options are located in the "Configuration" section (Cell [2]) of the `Phase_1_DivPO.py` script:

*   `MODEL_NAME`: The Hugging Face model identifier for the base language model (e.g., `"Qwen/Qwen2.5-0.5B-Instruct"`, `"microsoft/phi-3-mini-4k-instruct"`). You can also use a local path.
*   `OUTPUT_DIR`: Directory to save model checkpoints and the final trained model.
*   `K_SAMPLES`: Number of candidate words to generate per prompt for diversity calculation. Adjust based on available memory.
*   `DPO_BETA`: The beta parameter for DPO loss, controlling the trade-off between the policy and reference model.
*   `PER_DEVICE_TRAIN_BATCH_SIZE`: Batch size per GPU/MPS/CPU device.
*   `GRADIENT_ACCUMULATION_STEPS`: Accumulates gradients over multiple steps to simulate a larger effective batch size.
*   `LEARNING_RATE`: The learning rate for the AdamW optimizer.
*   `NUM_TRAIN_EPOCHS`: Number of training epochs.
*   `MAX_PROMPT_LENGTH`, `MAX_TARGET_LENGTH`: Maximum token lengths for prompts and generated words.
*   `EMBEDDING_MODEL_NAME`: Sentence Transformer model used for diversity calculation.
*   `EMBEDDING_DEVICE`: Device used for embedding calculations (`mps` or `cpu`).

## Usage

1.  **Modify Configuration:** Adjust the parameters in the "Configuration" section of the script as needed.
2.  **Run the Script:**
    *   **As a Python script:**
        ```bash
        python Phase_1_DivPO.py
        ```
    *   **In a Jupyter environment:** Open the `.ipynb` file (if you convert the script) and run the cells sequentially.

3.  **Output:**
    *   Training progress, loss, and metrics will be logged to the console.
    *   Model checkpoints will be saved periodically in subdirectories within the specified `OUTPUT_DIR`.
    *   The final trained model will be saved in `OUTPUT_DIR/final_checkpoint` upon successful completion.
    *   If training fails, an attempt will be made to save the current state to `OUTPUT_DIR/checkpoint_on_error`.

## Code Structure

*   **Imports and Setup:** Loads necessary libraries and performs initial setup (logging, warnings).
*   **Configuration:** Defines model names, hyperparameters, and paths.
*   **Helper Functions:** Contains functions for NLP data download, quality scoring (`calculate_quality`), embedding calculation (`get_embeddings`), and diversity scoring (`calculate_pairwise_semantic_diversity`).
*   **Prepare Dataset:** Creates a simple dataset containing only the prompts.
*   **Load Tokenizer and Model:** Loads the specified Hugging Face tokenizer and model, handling device placement (CPU/MPS).
*   **Custom DivPODPOTrainer:** Defines the `DivPODPOTrainer` class, overriding `compute_loss` to implement the DivPO logic (generation, scoring, pairing, loss calculation) and `_prepare_dataset` to handle prompt-only input.
*   **Custom Data Collator:** Defines `DivPODataCollator` to prepare batches containing only tokenized prompts.
*   **Training Arguments & Trainer Initialization:** Configures `DPOConfig` and initializes the `DivPODPOTrainer`.
*   **Train the Model:** Starts the training process using `divpo_trainer.train()`.
*   **Save Final Model:** Saves the final model, tokenizer, and configuration.
*   **(Optional) Test Generation:** A commented-out cell provides example code to load the trained model and test its generation capabilities.

## Dependencies

*   `torch`
*   `transformers`
*   `trl`
*   `datasets`
*   `sentence-transformers`
*   `spacy` (+ `en_core_web_sm` model)
*   `nltk` (+ `words`, `averaged_perceptron_tagger` data)
*   `wordfreq`
*   `accelerate`
*   `bitsandbytes` (Optional, for 4-bit quantization if enabled)

## License

MIT License

Copyright (c) 2025 jk2500

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 