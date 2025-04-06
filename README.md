# DivPO Training for Diverse Noun Generation

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
    ```bash
    python run_divpo.py
    ```bash
3.  **Output:**
    *   Training progress, loss, and metrics will be logged to the console.
    *   Model checkpoints will be saved periodically in subdirectories within the specified `OUTPUT_DIR`.
    *   The final trained model will be saved in `OUTPUT_DIR/final_checkpoint` upon successful completion.
    *   If training fails, an attempt will be made to save the current state to `OUTPUT_DIR/checkpoint_on_error`.
    *   **Note:** Check `divpo/config.py` and `.gitignore` for the exact `OUTPUT_DIR` location and git tracking status.

## Code Structure

The codebase is organized into the following modules:

*   `divpo/__init__.py`: Package initialization.
*   `divpo/config.py`: Configuration parameters for training (model names, hyperparameters, paths, prompts).
*   `divpo/data.py`: Dataset preparation (prompt dataset creation) and data collation.
*   `divpo/model.py`: Loading and configuration of the language model and tokenizer.
*   `divpo/trainer.py`: Defines the custom `DivPODPOTrainer` class, overriding methods to implement the DivPO logic (generation, scoring, pairing, loss calculation).
*   `divpo/training.py`: High-level functions orchestrating the training pipeline (setup, load, train, save).
*   `divpo/utils.py`: Helper functions for NLP resource setup (NLTK, spaCy downloads), quality scoring (`calculate_quality`), embedding calculation (`get_embeddings`), and diversity scoring (`calculate_pairwise_semantic_diversity`).
*   `run_divpo.py`: The main script to execute the training process by calling functions from `divpo.training`.
*   `requirements.txt`: Lists project dependencies.
*   `Phase_1_DivPO.py`: (Deprecated) Original monolithic implementation script.
*   `setup_nlp_resources.py`: (Deprecated) Older script for NLP setup, now handled in `divpo/utils.py`.
*   `test.py`: Script for testing the trained model (may require updates).

## Example Test Results

A sample run comparing the baseline model (`Qwen/Qwen2.5-0.5B-Instruct`) against the model fine-tuned with DivPO yielded the following observations based on 200 generated samples:

*   **Vocabulary Shift & Collapse:** The trained model generated significantly fewer unique words (**9**) compared to the baseline model (**47**). Notably, the outputted words by the trained model ('silence', 'echo', 'peace', 'light', 'ethereal', 'elegance', 'effect', 'eternal', 'love') are **common nouns**, aligning with the project's quality goal, even though the overall variety decreased.
*   **Semantic Similarity/Diversity:**
    *   The *minimum* pairwise similarity between generated words increased significantly (0.033 baseline vs 0.139 trained), indicating that the closest pair of words in the trained set were more distinct than the closest pair in the baseline.
    *   The *average* pairwise similarity saw a slight increase (0.284 baseline vs 0.304 trained).
    *   The overall semantic diversity score (1 - max pairwise similarity) remained comparable, decreasing only slightly (0.716 baseline vs 0.696 trained).
*   **Interpretation:** While the training aimed for *diverse common nouns*, these results show a trade-off. The model converged to a much smaller vocabulary of valid common nouns, losing overall quantitative variety compared to the baseline. However, the words within this smaller set maintained a reasonable level of semantic distance from each other (especially avoiding very close pairs), and the overall diversity metric did not collapse. This might suggest a shift in sampling towards a constrained but relatively distinct set of common nouns, though further analysis is needed to determine if this outcome fully aligns with the desired goal of *broad* diversity within the target category.

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
