# Phase 1: DivPO Training for Diverse Noun Generation

This project implements the first phase of training a language model using **Diverse Preference Optimization (DivPO)**, a method introduced in the paper **"Diverse Preference Optimization"** by Lanchantin et al. ([arXiv:2501.18101](https://arxiv.org/abs/2501.18101)). The goal is to fine-tune a causal language model (e.g., Qwen, Phi) to generate diverse, common, English nouns based on a simple prompt, avoiding proper nouns.

The approach of using semantic distance to encourage and evaluate diversity draws inspiration from psychometric research, particularly the [Divergent Association Task (DAT)](https://www.datcreativity.com) (Olson et al., PNAS; DOI: 10.1073/pnas.2022340118).

The DAT demonstrates that the ability to generate semantically distant words correlates with established measures of human divergent thinking and creativity. While this project applies the concept to AI fine-tuning rather than human assessment, the underlying principle of leveraging semantic distance as an indicator of diversity remains central.

**Note:** This project has been refactored. The original monolithic script `Phase_1_DivPO.py` is preserved but no longer the primary implementation. The current implementation uses a modular structure within the `divpo/` directory and is executed via `run_divpo.py`.

## Features

*   **DivPO Implementation:** Uses a custom `DivPODPOTrainer` built on top of the TRL library's `DPOTrainer`.
*   **On-the-Fly Pair Generation:** Generates candidate words, scores them for quality and diversity, selects chosen/rejected pairs, and calculates the DPO loss within each training step, requiring only prompts as input dataset.
*   **Quality Filtering:** Uses spaCy for POS tagging and wordfreq for commonality checks to ensure generated words are common, single-word, non-proper nouns.
*   **Diversity Scoring:** Leverages sentence-transformers to calculate semantic diversity between generated candidates.
*   **Configurable:** Allows easy configuration of the base model, hyperparameters (K samples, DPO beta, learning rate, etc.), output directories, and embedding models.
*   **Device Compatibility:** Designed to run on CPU or Apple Silicon (MPS), with checks and logging for device selection. CUDA is possible but not the primary focus of the current setup.
*   **Automated NLP Data Download:** Includes checks and downloads for necessary NLTK and spaCy resources on the first run via `divpo/utils.py`.

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
    The script (`run_divpo.py`) will attempt to download required NLTK data (`words`, `averaged_perceptron_tagger`) and the spaCy model (`en_core_web_sm`) automatically when first run. Ensure you have an internet connection.

## Configuration

Key configuration options are located in the `divpo/config.py` script:

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
*   `DEFAULT_PROMPT`: The prompt template used for generation.
*   `BASE_NUM_PROMPTS`: Target number of prompt instances in the training dataset.

## Usage

1.  **Modify Configuration:** Adjust the parameters in `divpo/config.py` as needed.
2.  **Run the Training Script:**
    ```