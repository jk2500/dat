"""
Configuration settings for DivPO training
"""
import torch
import logging
import os

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Base model path or name
OUTPUT_DIR = "./divpo_dat_model_phase1"    # Where to save checkpoints and final model

# DivPO Hyperparameters
K_SAMPLES = 16  # Number of samples per prompt (adjust based on memory)
DPO_BETA = 0.1  # DPO beta parameter (controls how much to trust the reference model)

# Training Hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # Adjust based on GPU/MPS memory
GRADIENT_ACCUMULATION_STEPS = 8  # Adjust effective batch size
LEARNING_RATE = 1e-6             # Tune LR (often lower for fine-tuning)
NUM_TRAIN_EPOCHS = 5            # Increased from 5 to 20 epochs for longer training
LOGGING_STEPS = 20               # Log less frequently for long training
SAVE_STEPS = 100                 # Save checkpoints less frequently
MAX_PROMPT_LENGTH = 64           # Max tokens for the prompt
MAX_TARGET_LENGTH = 10           # Max *new* tokens for the generated word

# Embedding Model for Diversity Calculation
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Prioritize CUDA over MPS
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

# Prompt template
DEFAULT_PROMPT = "Generate a english single word. Do not use proper nouns like names of people or places. Just generate a single word."

# Dataset configuration
EFFECTIVE_BATCH_SIZE = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
BASE_NUM_PROMPTS = 4096  # Increased from 1024 to 4096 for more training data

def calculate_num_prompts():
    """Calculate number of prompts, ensuring it's divisible by effective batch size."""
    num_prompts = (BASE_NUM_PROMPTS // EFFECTIVE_BATCH_SIZE) * EFFECTIVE_BATCH_SIZE
    if num_prompts == 0:
        num_prompts = EFFECTIVE_BATCH_SIZE
        logger.warning(
            f"Base num_prompts ({BASE_NUM_PROMPTS}) too small for effective batch size "
            f"({EFFECTIVE_BATCH_SIZE}), setting to {EFFECTIVE_BATCH_SIZE}"
        )
    return num_prompts