#!/usr/bin/env python
# coding: utf-8

# # Phase 1: DivPO Training for Diverse Noun Generation

# This notebook implements the first phase of training using Diversity-enhanced Preference Optimization (DivPO). The goal is to fine-tune a language model (like Phi-4-mini-instruct) to generate diverse, common, single-word English nouns based on a simple prompt.

# ## 1. Imports and Setup

# In[1]:


import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    BitsAndBytesConfig, # Optional
    PreTrainedModel,
    PreTrainedTokenizerBase
)
# Ensure PreTrainedModelWrapper is imported if using older TRL versions
# from trl.models import PreTrainedModelWrapper # Might be needed depending on TRL version
# For newer TRL, explicit wrapper might not be needed if model is nn.Module
from torch.nn.parallel import DistributedDataParallel # Import if needed for type hints
from collections import OrderedDict # Import for type hints


from datasets import Dataset
from trl import DPOTrainer
from sentence_transformers import SentenceTransformer
import spacy
import nltk
from wordfreq import top_n_list # Or other frequency source
import logging
import random
import os
import ssl
from typing import Dict, List, Optional, Tuple, Union, Any, Callable # Added missing types
import warnings


# --- Suppress specific DeprecationWarning ---
# Filter the specific warning about Trainer.tokenizer for both potential categories
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is now deprecated.*")


# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ## 2. Configuration

# In[2]:


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # Or other model
#MODEL_NAME = "./phi-4-mini-instruct-local" # Make sure this path is correct
OUTPUT_DIR = "./divpo_dat_model_phase1"      # Where to save checkpoints and the final model

# Optional: Quantization for large models (Keep False for MPS/CPU unless specifically needed and supported)
# USE_QUANTIZATION = False
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16 # Or float16
# )

# DivPO Hyperparameters
K_SAMPLES = 16 # Number of samples per prompt (adjust based on memory)
DPO_BETA = 0.1 # DPO beta parameter (controls how much to trust the reference model)

# Training Hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = 2 # Adjust based on GPU/MPS memory
GRADIENT_ACCUMULATION_STEPS = 8 # Adjust effective batch size (effective_batch_size = batch_size * num_gpus * grad_accum)
LEARNING_RATE = 1e-6 # Tune LR (often lower for fine-tuning)
NUM_TRAIN_EPOCHS = 1 # Start with 1 epoch
LOGGING_STEPS = 10 # Log more frequently initially
SAVE_STEPS = 100 # Save checkpoints periodically
MAX_PROMPT_LENGTH = 64 # Max tokens for the prompt
MAX_TARGET_LENGTH = 10 # Max *new* tokens for the generated word

# Embedding Model for Diversity Calculation
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu" # Use MPS if available, else CPU

logger.info(f"Using device for embedding model: {EMBEDDING_DEVICE}")


# ## 3. Helper Functions (Setup & Definitions)

# In[3]:


# --- NLTK & spaCy Download --- (Run once)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Checking/Downloading NLTK data...")
try:
    nltk.data.find('corpora/words')
    print("- NLTK 'words' already downloaded.")
except LookupError:
    print("- Downloading NLTK 'words'...")
    nltk.download('words', quiet=False)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    print("- NLTK 'averaged_perceptron_tagger' already downloaded.")
except LookupError:
    print("- Downloading NLTK 'averaged_perceptron_tagger'...")
    nltk.download('averaged_perceptron_tagger', quiet=False)
print("NLTK data download check complete.")

# --- spaCy Model Download ---
SPACY_MODEL = 'en_core_web_sm'
print(f"Checking/Downloading spaCy model '{SPACY_MODEL}'...")
try:
    nlp = spacy.load(SPACY_MODEL)
    print(f"- spaCy '{SPACY_MODEL}' model already installed.")
except OSError:
    print(f"- spaCy '{SPACY_MODEL}' model not found. Downloading...")
    spacy.cli.download(SPACY_MODEL)
    print(f"- spaCy model '{SPACY_MODEL}' downloaded.")
    nlp = spacy.load(SPACY_MODEL) # Load after downloading

print("NLP data/models check/download complete.")


# In[4]:


# --- Load NLP Tools & Define Helper Functions ---

logger.info("Loading NLP tools and embedding model...")
try:
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    common_vocab = set(top_n_list('en', 20000)) # Adjust N as needed
    # nlp = spacy.load(SPACY_MODEL) # Already loaded in previous cell
except LookupError:
    logger.error("NLTK resources failed to load even after download attempt. Please check installation.")
    raise # Re-raise error to stop execution
except Exception as e:
    logger.error(f"Error loading NLP resources: {e}", exc_info=True)
    raise

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    logger.info(f"Tools loaded. Embedding model '{EMBEDDING_MODEL_NAME}' on {EMBEDDING_DEVICE}.")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
    raise


def calculate_quality(word: str) -> float:
    """Checks if a single word meets the criteria (common, non-proper noun). Returns 1.0 or 0.0."""
    word = str(word).strip() # Ensure it's a string and stripped
    word_lower = word.lower()

    # 1. Format Check: Must be a single, non-empty, non-hyphenated word
    if not word or ' ' in word or '-' in word: # Also exclude hyphenated words for simplicity
        print(f"Quality Fail: '{word}' Invalid format (space/hyphen/empty).")
        return 0.0 # Not a single word or empty

    # 2. Commonality Check: Must be in the common vocabulary list
    is_common = word_lower in common_vocab # common_vocab loaded from wordfreq
    if not is_common:
        print(f"Quality Fail: '{word}' Not common enough.")
        return 0.0

    # 3. spaCy Processing & POS Tagging: Analyze the word's grammatical role
    doc = nlp(word) # Use the pre-loaded spaCy model
    if not doc or not doc[0]:
        print(f"Quality Fail: '{word}' SpaCy couldn't process.")
        return 0.0 # Spacy couldn't process

    token = doc[0]
    pos_tag = token.tag_ # Get the fine-grained Part-of-Speech tag

    # 4. Noun Type Check:
    #    - Must be a noun (NN: singular, NNS: plural)
    #    - Must NOT be a proper noun (NNP: singular proper, NNPS: plural proper)
    #    - Must NOT be unexpectedly capitalized (e.g., "Table" if not a proper noun)
    is_noun = pos_tag in ['NN', 'NNS']
    is_proper_noun = pos_tag in ['NNP', 'NNPS']
    is_unexpectedly_capitalized = len(word) > 1 and word[0].isupper() and not is_proper_noun

    # 5. Final Decision: All conditions must be met
    if is_noun and not is_proper_noun and not is_unexpectedly_capitalized and is_common:
        print(f"Quality OK: '{word}' Noun:{is_noun}, Proper:{is_proper_noun}, Common:{is_common}, Cap: {is_unexpectedly_capitalized}")
        return 1.0 # Passes all checks
    else:
        # Log why it failed
        print(f"Quality Fail: '{word}' Noun:{is_noun}, Proper:{is_proper_noun}, Common:{is_common}, Cap: {is_unexpectedly_capitalized}")
        return 0.0 # Fails one or more checks

def get_embeddings(words: List[str], embedding_model: SentenceTransformer, device: Union[str, torch.device]) -> torch.Tensor:
    """Get embeddings for a list of words."""
    if not words:
        return torch.empty((0, embedding_model.get_sentence_embedding_dimension()), device=device, dtype=torch.float32)
    with torch.no_grad():
        embeddings = embedding_model.encode(
            words,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
    # Sometimes encode might return float16 on MPS, ensure float32
    return embeddings.to(dtype=torch.float32)

def calculate_pairwise_semantic_diversity(idx: int, embeddings: torch.Tensor) -> float:
    """
    Calculates diversity for a word based on its maximum cosine similarity
    to any *other* word in the batch.
    Diversity = 1 - max_similarity. Higher score means more distinct.

    Args:
        idx: Index of the word in the embeddings tensor.
        embeddings: Tensor of shape (num_words, embedding_dim).

    Returns:
        Diversity score (float).
    """
    if embeddings.shape[0] <= 1:
        return 0.0 # No diversity if only one word

    current_embedding = embeddings[idx].unsqueeze(0) # Shape: (1, dim)
    other_embeddings = torch.cat([embeddings[:idx], embeddings[idx+1:]], dim=0) # Shape: (num_words-1, dim)

    # Calculate cosine similarities (using efficient matrix multiplication)
    # Normalize embeddings for cosine similarity calculation
    current_embedding_norm = F.normalize(current_embedding, p=2, dim=1)
    other_embeddings_norm = F.normalize(other_embeddings, p=2, dim=1)
    cosine_similarities = torch.mm(current_embedding_norm, other_embeddings_norm.t()).squeeze() # Shape: (num_words-1,)

    # Handle case where there's only one 'other' word (returns scalar tensor)
    if cosine_similarities.dim() == 0:
        max_similarity = cosine_similarities.item()
    else:
        max_similarity = torch.max(cosine_similarities).item()

    diversity = 1.0 - max_similarity
    return diversity


# ## 4. Prepare Dataset (Prompts Only)

# In[5]:


# For DivPO, we only need the prompts initially. The generations are done on-the-fly.
prompt = "Generate a english single word. Do not use proper nouns like names of people or places. Just generate a single word."

# Make num_prompts divisible by effective batch size for simplicity and efficiency
effective_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
base_num_prompts = 1024 # Desired approximate number of prompts
num_prompts_for_epoch = (base_num_prompts // effective_batch_size) * effective_batch_size
if num_prompts_for_epoch == 0:
    num_prompts_for_epoch = effective_batch_size # Ensure at least one full batch
    logger.warning(f"Base num_prompts ({base_num_prompts}) too small for effective batch size ({effective_batch_size}), setting num_prompts_for_epoch to {effective_batch_size}")


# Create a simple dataset dictionary
data = {"prompt": [prompt] * num_prompts_for_epoch}

# Convert to Hugging Face Dataset object
train_dataset = Dataset.from_dict(data)

logger.info(f"Created dataset with {len(train_dataset)} prompt instances (effective batch size: {effective_batch_size}).")
print(train_dataset)


# ## 5. Load Tokenizer and Model

# In[6]:


logger.info(f"Loading tokenizer for: {MODEL_NAME}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            padding_side='left' # <--- Add this line
        )
if tokenizer.pad_token is None:
    logger.info("Setting pad token to eos token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id # Ensure pad_token_id is also set
logger.info(f"Tokenizer loaded. Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

# Define model loading arguments - START BY LOADING ON CPU
model_kwargs = {
    "torch_dtype": torch.float32,  # Use float32 for MPS compatibility
    "device_map": "cpu",            # <--- Load onto CPU initially
    # Add trust_remote_code=True if required by the specific model (like Phi-3)
  "trust_remote_code": True,
}

# Check for MPS availability
mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
cuda_available = torch.cuda.is_available()
model_device = "cpu" # Default device

# Load the model onto CPU first
logger.info(f"Loading model '{MODEL_NAME}' onto CPU...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
logger.info("Model loaded on CPU.")

# Move model to MPS if available and intended
if mps_available:
    logger.info("Attempting to move model to Apple Silicon (MPS) backend...")
    try:
        model = model.to("mps")
        model_device = "mps" # Update model device
        logger.info("Model successfully moved to MPS.")
        # Optional: Run a small operation to confirm MPS is working
        # try:
        #     _ = torch.randn(1, device='mps')
        #     logger.info("MPS confirmed functional.")
        # except Exception as mps_test_e:
        #      logger.warning(f"MPS device detected but test operation failed: {mps_test_e}. Training might fallback to CPU.")
    except Exception as e:
        logger.error(f"Failed to move model to MPS: {e}. Keeping model on CPU.", exc_info=True)
        mps_available = False # Fallback if move fails
elif cuda_available:
    logger.info("CUDA available, but this setup prioritizes MPS/CPU. Keeping model on CPU.")
    # If CUDA support is desired, change device logic here and in TrainingArguments
else:
    logger.info("MPS not available. Keeping model on CPU.")


# Reference model will be created automatically by DPOTrainer if not provided
# DPOTrainer/Accelerate should handle placing the ref_model correctly based on the main model's device
logger.info(f"Model final compute device: {model.device}") # Check the primary device where computations will happen

# Optional: Log the device map if accelerate populated it (might be simple after .to())
# try:
#     logger.info(f"Model device map after move: {model.hf_device_map}")
# except AttributeError:
#     logger.info("Model does not have hf_device_map attribute after .to()")


# ## 6. Define Custom DivPODPOTrainer

# In[7]:


from trl import DPOConfig

# Type Hinting for clarity (adapt based on exact TRL version)
ModelType = Union[PreTrainedModel, torch.nn.Module, DistributedDataParallel]
# If using wrappers:
# ModelType = Union[PreTrainedModelWrapper, torch.nn.Module, DistributedDataParallel]


class DivPODPOTrainer(DPOTrainer):
    def __init__(
        self,
        # Custom arguments first
        k_samples: int = 16,
        quality_fn: Callable[[str], float] = None,
        diversity_fn: Callable[[int, torch.Tensor], float] = None,
        embedding_model: SentenceTransformer = None,
        embedding_device: Union[str, torch.device] = None,
        # Arguments for the parent DPOTrainer
        model: ModelType = None,
        ref_model: Optional[ModelType] = None,
        args: Optional[DPOConfig] = None,
        train_dataset: Optional[Dataset] = None,
        # Use processing_class instead of tokenizer
        processing_class=tokenizer,  # This replaces tokenizer
        data_collator=None,
        eval_dataset=None,
        # Other parameters with defaults
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        beta: float = 0.1,
        **kwargs
    ):
        # Store custom attributes first
        self.k_samples = k_samples
        self._quality_fn = quality_fn
        self._diversity_fn = diversity_fn
        self._embedding_model = embedding_model
        self._embedding_device = embedding_device
        
        # Store parameters that DPOTrainer expects
        self.beta = beta
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        
        # Call parent init with the parameters it expects in the current version
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,  # Use processing_class instead of tokenizer
            data_collator=data_collator,
            **kwargs
        )
        
        # Check that required custom args were provided
        if self._quality_fn is None or self._diversity_fn is None or self._embedding_model is None or self._embedding_device is None:
            raise ValueError("quality_fn, diversity_fn, embedding_model, and embedding_device must be provided")

        logger.info(f"DivPO Trainer Initialized with K={self.k_samples}, Beta={self.beta}")
        logger.info("Ensure processing_class (tokenizer) has pad_token set correctly.")
        if self.processing_class.pad_token is None:
            logger.warning("Processing class pad_token is None. Setting to eos_token.")
            self.processing_class.pad_token = self.processing_class.eos_token
            self.processing_class.pad_token_id = self.processing_class.eos_token_id

    def compute_loss(
        self,
        model: ModelType, # Use the ModelType hint
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # Ensure reference model is ready and on the correct device
        if self.ref_model is None:
             logger.error("Reference model not found/loaded by DPOTrainer!")
             raise ValueError("Reference model not loaded!")
        self.ref_model.to(self.accelerator.device)
        # Ensure main model is also on the accelerator device (should be handled by Trainer/Accelerator)
        # model.to(self.accelerator.device) # Usually not needed here

        # Get prompts from inputs. DPOTrainer should have tokenized the 'prompt' column.
        # Keys are usually "input_ids", "attention_mask", "labels" (which might be copies of input_ids)
        # We only need the prompt part for generation.
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]

        # Determine the actual prompt length (excluding padding)
        # This assumes right-padding. DPOTrainer usually handles padding.
        if self.processing_class.padding_side == "right":
            prompt_lens = torch.sum(prompt_attention_mask, dim=1)
        else: # Assuming left padding
             # Find first non-pad token index for each sequence
             prompt_lens = (prompt_input_ids != self.processing_class.pad_token_id).sum(-1)
             # Need careful handling if using left padding during generation & logprob calculation
             logger.warning("Left padding detected. Ensure generation and logprob logic handles it correctly.")


        batch_size = prompt_input_ids.size(0)

        all_policy_chosen_logps = []
        all_policy_rejected_logps = []
        all_ref_chosen_logps = []
        all_ref_rejected_logps = []
        valid_pairs_found_in_batch = 0
        batch_chosen_words = []
        batch_rejected_words = []


        # --- Generation Phase ---
        # Generation config - Use self attributes where possible
        generation_config = GenerationConfig(
            max_new_tokens=self.max_target_length, # Use trainer's config
            min_new_tokens=1,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            do_sample=True,
            temperature=1.5, # Slightly higher temp for more variety in generation
            top_k=75,
            # top_p=0.9 # Can also use top_p
        )

        # Prepare inputs for batch generation (repeat prompts K times)
        # Ensure they are on the correct device for the model
        model_compute_device = model.device # Device where the actual model parameters reside
        batch_prompt_ids = prompt_input_ids.repeat_interleave(self.k_samples, dim=0).to(model_compute_device)
        batch_prompt_mask = prompt_attention_mask.repeat_interleave(self.k_samples, dim=0).to(model_compute_device)
        prompt_lens_repeated = prompt_lens.repeat_interleave(self.k_samples) # Keep on CPU for indexing

        # Use the model passed to compute_loss (potentially wrapped by Accelerator)
        # No need to unwrap explicitly here, Trainer handles it.

        print(f"Generating {batch_prompt_ids.shape[0]} samples (Batch Size {batch_size} * K {self.k_samples}) on device {model_compute_device}...")
        with torch.no_grad():
            policy_outputs = model.generate(
                input_ids=batch_prompt_ids,
                attention_mask=batch_prompt_mask,
                generation_config=generation_config,
            )
        print("Generation complete.")

        # Move outputs back to CPU for decoding if they aren't already
        policy_outputs_cpu = policy_outputs.cpu()

        # Decode generated tokens, skipping prompt and special tokens
        all_candidate_words_flat = []
        for i in range(policy_outputs_cpu.shape[0]):
            original_prompt_len = prompt_lens_repeated[i].item() # Use CPU tensor here
            # Slice generated part: from end of prompt to end of sequence
            generated_tokens = policy_outputs_cpu[i, original_prompt_len:]
            decoded_text = self.processing_class.decode(generated_tokens, skip_special_tokens=True).strip()

            # --- MODIFIED CLEANUP ---
            # Take the *last* word and strip common punctuation, assuming it's the target noun.
            # Handle cases where the decoded text might be empty or only contain punctuation after splitting.
            words_in_decoded = decoded_text.split()
            if words_in_decoded:
                # Get the last element
                last_word_candidate = words_in_decoded[-1]
                # Strip common leading/trailing punctuation
                cleaned_word = last_word_candidate.strip(".,;:!?\"'()[]{}<>")
            else:
                # If splitting results in no words (e.g., empty string or just spaces/punctuation)
                cleaned_word = ""
            # --- END MODIFIED CLEANUP ---

            all_candidate_words_flat.append(cleaned_word) # Append the potentially empty cleaned word

        # --- ADDED PRINT STATEMENT ---
        # Print all generated words for this batch (after basic cleanup)
        print(f"--- Generated words for batch (Step {self.state.global_step if self.state else 'N/A'}): ---")
        print(all_candidate_words_flat) # This will now show words after the new cleanup
        print(f"--- End Generated words ---")
        # --- END ADDED PRINT STATEMENT ---


        # --- Scoring and Pair Selection Phase ---
        # Process each original prompt's K samples
        for i in range(batch_size):
            start_idx = i * self.k_samples
            end_idx = start_idx + self.k_samples
            candidate_words_for_prompt = all_candidate_words_flat[start_idx:end_idx]

            # Filter out empty strings resulting from cleanup (this step remains important)
            candidate_words = [w for w in candidate_words_for_prompt if w]
            # Lowercase after filtering empty strings
            candidate_words = [w.lower() for w in candidate_words]

            if not candidate_words:
                print(f"Batch item {i}: No valid words generated after initial cleanup.")
                continue

            # Remove duplicates before scoring
            unique_candidate_words = list(OrderedDict.fromkeys(candidate_words))
            num_candidates = len(unique_candidate_words)

            if num_candidates == 0:
                print(f"Batch item {i}: No unique non-empty words after cleanup.")
                continue

            print(f"Batch item {i}: Processing {num_candidates} unique candidates: {unique_candidate_words[:5]}...")

            # Score Candidates (Quality)
            qualities_list = [self._quality_fn(w) for w in unique_candidate_words]
            qualities = torch.tensor(qualities_list, device=self.accelerator.device) # Move scores to accelerator device

            # Score Candidates (Diversity) - only if more than one candidate
            diversities = torch.zeros(num_candidates, device=self.accelerator.device) # Default to zero
            if num_candidates > 1:
                 try:
                    # Calculate embeddings on the designated embedding device
                    embeddings = get_embeddings(unique_candidate_words, self._embedding_model, self._embedding_device)
                    # Ensure embeddings are on the accelerator device for diversity calculation
                    embeddings = embeddings.to(self.accelerator.device)
                    # Call the diversity function for each candidate
                    diversities_list = [self._diversity_fn(j, embeddings) for j in range(num_candidates)]
                    diversities = torch.tensor(diversities_list, device=self.accelerator.device)
                 except Exception as e:
                    logger.error(f"Error getting embeddings/diversity for batch item {i}: {e}", exc_info=True)
                    # Keep diversities as zeros

            # --- MODIFIED SELECTION LOGIC ---
            y_chosen_str, y_rejected_str = None, None # Initialize
            chosen_idx_in_unique, rejected_idx_in_unique = -1, -1 # Initialize indices for logging

            # Find indices of candidates with good quality (>= 1.0)
            qualities_tensor = torch.tensor(qualities_list, device=self._embedding_device) # Use list here
            idx_good_quality = torch.where(qualities_tensor >= 1.0)[0]
            # Find indices of candidates with bad quality (< 1.0)
            idx_bad_quality = torch.where(qualities_tensor < 1.0)[0]

            # --- Select Chosen Word (Highest Diversity among Good Quality) ---
            if len(idx_good_quality) >= 1: # Need at least one good word to be chosen
                good_diversities = diversities[idx_good_quality]
                max_diversity_val = torch.max(good_diversities)
                potential_chosen_indices_rel = torch.where(good_diversities == max_diversity_val)[0]
                # Randomly select one if there are ties for max diversity
                chosen_rel_idx = potential_chosen_indices_rel[torch.randint(len(potential_chosen_indices_rel), (1,)).item()]
                chosen_idx_in_unique = idx_good_quality[chosen_rel_idx].item()
                y_chosen_str = unique_candidate_words[chosen_idx_in_unique]
            else:
                 print(f"Batch item {i}: No good quality words found to select a chosen word.")


            # --- Select Rejected Word (Lowest Diversity among Bad Quality) ---
            if len(idx_bad_quality) >= 1: # Need at least one bad word to be rejected
                # Consider diversity only if there's more than one candidate overall (otherwise diversity is 0)
                if num_candidates > 1:
                    bad_diversities = diversities[idx_bad_quality]
                    min_diversity_val = torch.min(bad_diversities)
                    potential_rejected_indices_rel = torch.where(bad_diversities == min_diversity_val)[0]
                    # Randomly select one if there are ties for min diversity
                    rejected_rel_idx = potential_rejected_indices_rel[torch.randint(len(potential_rejected_indices_rel), (1,)).item()]
                    rejected_idx_in_unique = idx_bad_quality[rejected_rel_idx].item()
                    y_rejected_str = unique_candidate_words[rejected_idx_in_unique]
                else:
                    # If only one candidate overall, it must be bad quality here. Reject it.
                    rejected_idx_in_unique = idx_bad_quality[0].item()
                    y_rejected_str = unique_candidate_words[rejected_idx_in_unique]

            else:
                 # If there are no bad quality words, we might still have a chosen word,
                 # but we can't select a rejected word based on the new criteria.
                 print(f"Batch item {i}: No bad quality words found to select a rejected word.")


            # Note: The check `if chosen_idx_in_unique == rejected_idx_in_unique:` is removed
            # because chosen and rejected now come from disjoint quality sets (good vs bad).
            # The check `if y_chosen_str is not None and y_rejected_str is not None:` below
            # is now the primary guard to ensure a valid pair exists.
            # --- END MODIFIED SELECTION LOGIC ---

            # --- ADDED: Print the selected pair (or failure) ---
            if y_chosen_str is not None and y_rejected_str is not None:
                print(f"Batch item {i}: Selected Pair -> Chosen='{y_chosen_str}' (Good Q), Rejected='{y_rejected_str}' (Bad Q)")
            else:
                # More specific failure message
                fail_msg = []
                if y_chosen_str is None: fail_msg.append("No suitable chosen word found (good quality)")
                if y_rejected_str is None: fail_msg.append("No suitable rejected word found (bad quality)")
                print(f"Batch item {i}: Failed to select a distinct pair. Reason(s): {'; '.join(fail_msg)}")
            # --- END ADDED PRINT ---

            # --- Log Probability Calculation Phase ---
            # Get Log Probabilities if a valid pair was found
            if y_chosen_str is not None and y_rejected_str is not None:
                batch_chosen_words.append(y_chosen_str)
                batch_rejected_words.append(y_rejected_str)

                # Tokenize chosen/rejected words as target sequences
                chosen_tokens = self.processing_class(y_chosen_str, add_special_tokens=False).input_ids
                rejected_tokens = self.processing_class(y_rejected_str, add_special_tokens=False).input_ids

                # Max length check
                if len(chosen_tokens) > self.max_target_length or len(rejected_tokens) > self.max_target_length:
                    logger.warning(f"Batch item {i}: Skipping pair - generated token length exceeds max_target_length. Chosen: '{y_chosen_str}' ({len(chosen_tokens)}), Rejected: '{y_rejected_str}' ({len(rejected_tokens)}) ")
                    continue

                # Prepare dict for concatenated_forward (batch size of 1 for this pair)
                current_prompt_ids = prompt_input_ids[i:i+1, :prompt_lens[i].item()].to(self.accelerator.device)
                current_prompt_mask = prompt_attention_mask[i:i+1, :prompt_lens[i].item()].to(self.accelerator.device)

                # Convert token lists to tensors on the correct device
                chosen_input_ids_tensor = torch.tensor(chosen_tokens, dtype=torch.long).unsqueeze(0).to(self.accelerator.device)
                rejected_input_ids_tensor = torch.tensor(rejected_tokens, dtype=torch.long).unsqueeze(0).to(self.accelerator.device)

                # Create attention masks (all ones for the actual tokens)
                chosen_attention_mask_tensor = torch.ones_like(chosen_input_ids_tensor)
                rejected_attention_mask_tensor = torch.ones_like(rejected_input_ids_tensor)

                # Ensure the dictionary has the keys expected by concatenated_forward
                dpo_pair_batch = {
                    "prompt_input_ids": current_prompt_ids,
                    "prompt_attention_mask": current_prompt_mask,
                    "chosen_input_ids": chosen_input_ids_tensor,         # Correct key name
                    "chosen_attention_mask": chosen_attention_mask_tensor,    # Add mask
                    "rejected_input_ids": rejected_input_ids_tensor,       # Correct key name
                    "rejected_attention_mask": rejected_attention_mask_tensor, # Add mask
                    # Add pixel_values etc. if it were a vision model and needed
                }

                try:
                    # ----- CORRECTED FUNCTION CALLS -----
                    # Get logps using the policy model ('model')
                    # The model passed here might be wrapped by Accelerator
                    policy_outputs = self.concatenated_forward(
                        model, dpo_pair_batch
                    )
                    policy_chosen_logps_i = policy_outputs["chosen_logps"]
                    policy_rejected_logps_i = policy_outputs["rejected_logps"]

                    # Get logps using the reference model ('self.ref_model' or policy model with adapter disabled)
                    # Ensure ref_model is on the correct device (should be handled by Trainer/Accelerator)
                    if self.ref_model:
                         # Ensure ref_model is on the accelerator device if not already
                         self.ref_model.to(self.accelerator.device)
                         ref_outputs = self.concatenated_forward(
                             self.ref_model, dpo_pair_batch
                         )
                         ref_chosen_logps_i = ref_outputs["chosen_logps"]
                         ref_rejected_logps_i = ref_outputs["rejected_logps"]
                    elif self.is_peft_model: # PEFT case: use policy model with adapter disabled
                         # DPOTrainer provides a context manager for this
                         with self.null_ref_context():
                              # Need to use the unwrapped model inside the context
                              unwrapped_model = self.accelerator.unwrap_model(model)
                              ref_outputs = self.concatenated_forward(
                                  unwrapped_model, dpo_pair_batch
                              )
                              ref_chosen_logps_i = ref_outputs["chosen_logps"]
                              ref_rejected_logps_i = ref_outputs["rejected_logps"]
                    elif self.precompute_ref_log_probs and "ref_chosen_logps" in inputs:
                        # If ref logps were precomputed and passed in inputs (less likely in custom compute_loss)
                        # Note: This requires adapting how 'inputs' is handled if you go this route.
                        # For the single-pair approach, this branch is less relevant.
                        logger.warning("Using precomputed ref logps - ensure inputs are structured correctly for this.")
                        # You would need to fetch the correct precomputed logps for this specific pair (i)
                        # This requires careful indexing based on how precomputed logps are stored/passed.
                        # Placeholder:
                        # ref_chosen_logps_i = inputs["ref_chosen_logps"][i:i+1]
                        # ref_rejected_logps_i = inputs["ref_rejected_logps"][i:i+1]
                        raise NotImplementedError("Handling precomputed ref logps per-pair inside compute_loss needs careful implementation.")

                    else:
                         # Should not happen if trainer is initialized correctly
                         logger.error("Reference model is None, not a PEFT model, and ref logps not precomputed. Cannot compute reference logps.")
                         # Skip this pair or raise an error
                         continue
                    # ----- END OF CORRECTED AREA -----

                    # Store logps for the final batch loss calculation
                    all_policy_chosen_logps.append(policy_chosen_logps_i)
                    all_policy_rejected_logps.append(policy_rejected_logps_i)
                    all_ref_chosen_logps.append(ref_chosen_logps_i)
                    all_ref_rejected_logps.append(ref_rejected_logps_i)
                    valid_pairs_found_in_batch += 1
                except Exception as e:
                    # Log details including the specific pair that failed
                    logger.error(f"Error getting logps for batch item {i}: Chosen='{y_chosen_str}', Rejected='{y_rejected_str}'. Error: {e}", exc_info=True)

            else:
                # --- UPDATED DEBUG LOGGING REASONS ---
                failure_reason = []
                # Update reasons based on new logic
                if y_chosen_str is None:
                    failure_reason.append(f"Need at least 1 good quality word, found {len(idx_good_quality)}")
                if y_rejected_str is None:
                     failure_reason.append(f"Need at least 1 bad quality word, found {len(idx_bad_quality)}")
                if not failure_reason: # If both were found but somehow pair failed (shouldn't happen now)
                    failure_reason.append("Unknown reason for pair failure")

                # Prepare quality and diversity strings for logging
                qual_strs = [f"{w}({q:.1f})" for w, q in zip(unique_candidate_words, qualities_list)]
                div_strs = [f"{w}({d:.3f})" for w, d in zip(unique_candidate_words, diversities.tolist())] if num_candidates > 1 else ["N/A"]*num_candidates

                print( # Changed logger.debug to print to ensure visibility if logging level is INFO
                    f"Batch item {i}: No valid DivPO pair. Reason(s): {'; '.join(failure_reason)}. "
                    f"Candidates ({num_candidates}): {unique_candidate_words}. "
                    f"Qualities: [{', '.join(qual_strs)}]. "
                    f"Diversities: [{', '.join(div_strs)}]. "
                    f"Selected Chosen Idx: {chosen_idx_in_unique}, Selected Rejected Idx: {rejected_idx_in_unique}."
                )
                # --- END UPDATED DEBUG LOGGING ---
                pass # No valid pair found for this prompt instance


        # --- Loss Calculation ---
        loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
        logits = None # Initialize logits

        if valid_pairs_found_in_batch > 0:
            # Concatenate logps from all valid pairs in the batch
            policy_chosen_logps = torch.cat(all_policy_chosen_logps)
            policy_rejected_logps = torch.cat(all_policy_rejected_logps)
            ref_chosen_logps = torch.cat(all_ref_chosen_logps)
            ref_rejected_logps = torch.cat(all_ref_rejected_logps)

            # Calculate DPO loss components
            # Shape: (num_valid_pairs,)
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps

            logits = pi_logratios - ref_logratios

            # Calculate loss based on specified loss type (e.g., sigmoid, hinge)
            if self.loss_type == "sigmoid":
                loss = -F.logsigmoid(self.beta * logits).mean()
            elif self.loss_type == "hinge":
                loss = torch.relu(1 - self.beta * logits).mean()
            # Add other loss types if needed (e.g., IPO)
            # elif self.loss_type == "ipo":
            #     loss = (logits - 1/(2 * self.beta)) ** 2
            #     loss = loss.mean()
            else:
                raise ValueError(f"Unsupported loss_type: {self.loss_type}")


            print(f"Batch Loss Calculated: {loss.item():.4f} from {valid_pairs_found_in_batch} pairs.")

            # Store metrics for logging
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            margins = (chosen_rewards - rejected_rewards).mean().item()
            self.store_metrics({
                "loss": loss.item(),
                "rewards/chosen": chosen_rewards.mean().item(),
                "rewards/rejected": rejected_rewards.mean().item(),
                "rewards/accuracy": accuracy,
                "rewards/margins": margins,
                "logps/rejected": policy_rejected_logps.detach().mean().item(),
                "logps/chosen": policy_chosen_logps.detach().mean().item(),
                "logits": logits.detach().mean().item(),
                "valid_pairs_frac": valid_pairs_found_in_batch / batch_size # Avg valid pairs per prompt
            })

            # Log first few chosen/rejected words for debugging occasionally
            if self.state.global_step % (self.args.logging_steps * 5) == 0: # Log samples less frequently
                 log_n = min(3, len(batch_chosen_words))
                 if log_n > 0:
                      sample_pairs = list(zip(batch_chosen_words[:log_n], batch_rejected_words[:log_n]))
                      logger.info(f"Step {self.state.global_step} Sample Pairs (Chosen | Rejected): {sample_pairs}")
                      logger.info(f"Step {self.state.global_step} Reward Metrics: Acc={accuracy:.3f}, Margin={margins:.3f}")


        else:
            # No valid pairs in the entire batch
            # --- MODIFIED WARNING ---
            # Log the global step number along with the warning
            logger.warning(f"Step {self.state.global_step}: No valid DivPO pairs found in the entire batch (Size: {batch_size}). Returning zero loss.")
            # --- END MODIFIED WARNING ---
            # Return zero loss but requires grad
            loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
            # Store zero metrics
            zero_metrics = {
                "loss": 0.0, "rewards/chosen": 0.0, "rewards/rejected": 0.0,
                "rewards/accuracy": 0.0, "rewards/margins": 0.0, "logps/rejected": 0.0,
                "logps/chosen": 0.0, "logits": 0.0, "valid_pairs_frac": 0.0
            }
            self.store_metrics(zero_metrics)

        # Dummy outputs dict for compatibility if needed
        outputs = {}
        if return_outputs:
            outputs["logits"] = logits # Logits tensor (or None if no pairs)
            # Add other outputs if necessary, e.g., chosen/rejected tokens/logps


        if return_outputs:
            return loss, outputs
        return loss
    
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        """Override the _prepare_dataset method to handle our prompt-only dataset."""
        if dataset is None:
            return None
            
        if not hasattr(dataset, "column_names") or len(dataset.column_names) == 0:
            raise ValueError("Dataset must have at least one column")
            
        # Check if we have a prompt-only dataset (our case)
        is_prompt_only = "prompt" in dataset.column_names and "chosen" not in dataset.column_names
        
        if is_prompt_only:
            logger.info(f"Detected prompt-only dataset for {dataset_name}. Will generate pairs on-the-fly.")
            # For prompt-only datasets, we'll just tokenize the prompts
            # and generate completions during training
            
            # First, apply chat template if needed
            if hasattr(processing_class, "apply_chat_template") and callable(processing_class.apply_chat_template):
                # Extract prompts
                prompts = [example["prompt"] for example in dataset]
                
                # Apply chat template if available
                def apply_template(prompt):
                    try:
                        return processing_class.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    except:
                        # Fallback if chat template fails
                        return prompt
                
                # Apply template to all prompts
                if isinstance(dataset, Dataset):  # Regular dataset
                    dataset = dataset.map(
                        lambda x: {"formatted_prompt": apply_template(x["prompt"])},
                        desc=f"Applying chat template to {dataset_name} dataset"
                    )
                else:  # IterableDataset
                    dataset = dataset.map(lambda x: {"formatted_prompt": apply_template(x["prompt"])})
                
                # Tokenize the formatted prompts
                def tokenize_prompt(example):
                    prompt = example["formatted_prompt"]
                    tokenized = processing_class(
                        prompt, 
                        truncation=True,
                        padding="max_length",
                        max_length=args.max_prompt_length,
                        return_tensors=None,
                    )
                    # Rename the columns to match what DPOTrainer expects
                    return {
                        "prompt_input_ids": tokenized["input_ids"],
                        "prompt_attention_mask": tokenized["attention_mask"],
                    }
                
                # Apply tokenization
                remove_cols = ["prompt"]
                if "formatted_prompt" in dataset.column_names:
                    remove_cols.append("formatted_prompt")
                    
                dataset = dataset.map(
                    tokenize_prompt,
                    remove_columns=remove_cols,
                    desc=f"Tokenizing {dataset_name} dataset"
                )
            else:
                # Simple tokenization without chat template
                def tokenize_row(example):
                    tokenized = processing_class(
                        example["prompt"],
                        truncation=True,
                        padding="max_length",
                        max_length=args.max_prompt_length,
                        return_tensors=None,
                    )
                    # Rename the columns to match what DPOTrainer expects
                    return {
                        "prompt_input_ids": tokenized["input_ids"],
                        "prompt_attention_mask": tokenized["attention_mask"],
                    }
                
                dataset = dataset.map(
                    tokenize_row,
                    remove_columns=["prompt"],
                    desc=f"Tokenizing {dataset_name} dataset"
                )
                
            return dataset
        else:
            # If it's a standard DPO dataset with prompt/chosen/rejected,
            # use the parent implementation
            return super()._prepare_dataset(dataset, processing_class, args, dataset_name)


# In[8]:


class DivPODataCollator:
    """Custom data collator for DivPO that works with prompt-only datasets."""
    
    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        
    def __call__(self, features):
        # Only collect prompt inputs - we'll generate chosen/rejected on-the-fly
        prompt_input_ids = [torch.tensor(f["prompt_input_ids"]) for f in features]
        prompt_attention_mask = [torch.tensor(f["prompt_attention_mask"]) for f in features]
        
        # Pad the inputs
        prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        prompt_attention_mask = torch.nn.utils.rnn.pad_sequence(
            prompt_attention_mask, batch_first=True, padding_value=0
        )
        
        # Create dummy chosen/rejected tensors (will be replaced in compute_loss)
        batch_size = prompt_input_ids.shape[0]
        dummy_tensor = torch.ones((batch_size, 1), dtype=torch.long) * self.tokenizer.pad_token_id
        
        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            # Add dummy tensors that will be replaced in compute_loss
            "chosen_input_ids": dummy_tensor.clone(),
            "chosen_attention_mask": torch.ones_like(dummy_tensor),
            "rejected_input_ids": dummy_tensor.clone(),
            "rejected_attention_mask": torch.ones_like(dummy_tensor),
        }


# ## 7. Training Arguments & Trainer Initialization

# In[9]:


# Determine if MPS should be used for training based on availability check during model loading
use_mps_for_training = mps_available and model_device == "mps" # Check both availability and if model is actually on MPS

# Initialize TrainingArguments
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    remove_unused_columns=False, # Keep 'prompt' column for DPOTrainer's default processing
    report_to="none", # Or "wandb", "tensorboard"
    bf16=False,  # Disabled BF16 for MPS/CPU
    fp16=False,  # Disabled FP16 for MPS/CPU
    gradient_checkpointing=False, # Can cause issues with DPO/generation/MPS sometimes, disable initially
    # gradient_checkpointing_kwargs={"use_reentrant": False}, # If enabling GC
    optim="adamw_torch", # Use standard AdamW
    # Set use_mps_device explicitly if needed by older transformers/accelerate, otherwise auto-detection usually works
    # use_mps_device=use_mps_for_training, # Explicitly tell TrainingArguments about MPS - may be deprecated
    # Instead, rely on Accelerate to detect MPS if available and model is on MPS device
    seed=42, # Set seed for reproducibility
    save_total_limit=2, # Limit number of checkpoints saved
    logging_first_step=True,
    warmup_ratio=0.03, # Small warmup
    lr_scheduler_type="cosine", # Cosine learning rate schedule
    # Ensure Accelerate uses the correct device (it should auto-detect if model is on MPS)
    # no_cuda=not cuda_available, # Tell accelerate explicitly not to use CUDA if not available/intended

    beta=DPO_BETA,  # You can set beta here instead of passing it separately
    model_init_kwargs=None,  # This is what DPOTrainer is looking for
)

logger.info(f"DPO Config configured. Effective batch size: {effective_batch_size}. Using MPS for training: {use_mps_for_training}")


# Initialize the custom DivPO Trainer
divpo_trainer = DivPODPOTrainer(
    model=model, # The model, potentially already on MPS
    ref_model=None, # DPOTrainer creates ref model automatically if None
    args=training_args,
    train_dataset=train_dataset, # Dataset containing 'prompt' column
    processing_class=tokenizer,
    data_collator=DivPODataCollator(tokenizer, max_length=MAX_PROMPT_LENGTH + MAX_TARGET_LENGTH),  
    eval_dataset=None,
    # --- Custom DivPO args ---
    k_samples=K_SAMPLES,
    quality_fn=calculate_quality,
    diversity_fn=calculate_pairwise_semantic_diversity,
    embedding_model=embedding_model,
    embedding_device=EMBEDDING_DEVICE, # Device for SBERT calculation
    # --- DPO Trainer specific args ---
    max_length=MAX_PROMPT_LENGTH + MAX_TARGET_LENGTH, # Max length for prompt + generation for internal processing
    max_prompt_length=MAX_PROMPT_LENGTH,              # Max length of prompt part
    max_target_length=MAX_TARGET_LENGTH,
    beta=DPO_BETA,         # Max length of *generated* part (completion)
    # loss_type="sigmoid" # Default
    # label_pad_token_id = -100 # Default
    # padding_value = tokenizer.pad_token_id # DPOTrainer usually infers this
)

logger.info("DivPO Trainer initialized.")


# ## 8. Train the Model

# In[10]:


logger.info("Starting DivPO Training...")
try:
    train_result = divpo_trainer.train()
    logger.info("Training finished successfully.")
    # Log training metrics
    metrics = train_result.metrics
    divpo_trainer.log_metrics("train", metrics)
    divpo_trainer.save_metrics("train", metrics)
    divpo_trainer.save_state() # Save trainer state
except Exception as e:
    logger.error(f"Training failed with error: {e}", exc_info=True)
    # Potentially save model even if training failed partway
    error_save_dir = os.path.join(OUTPUT_DIR, "checkpoint_on_error")
    logger.info(f"Attempting to save model state to {error_save_dir} after error...")
    try:
        divpo_trainer.save_model(error_save_dir)
        logger.info(f"Model saved to {error_save_dir}")
    except Exception as save_e:
        logger.error(f"Could not save model after error: {save_e}", exc_info=True)
finally:
    logger.info("Training process concluded (check logs for success or failure details).")


# ## 9. Save Final Model

# In[11]:


# Note: If training completed successfully, the trainer likely saved checkpoints
# according to save_steps. This cell saves the final state explicitly.
# If training failed, the try/except block in the previous cell attempted a save.

logger.info("Saving final model checkpoint (if training was successful and configured)...")
final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")

# Use the trainer's save_model method which also saves tokenizer and config
# This will overwrite if the directory exists from a previous run or error save
try:
    divpo_trainer.save_model(final_checkpoint_dir)
    logger.info(f"Final model, tokenizer, and config saved to {final_checkpoint_dir}")
except Exception as e:
    logger.error(f"Failed to save final model checkpoint: {e}", exc_info=True)

# save_model should handle the tokenizer, but double-check or save explicitly if needed
# if not os.path.exists(os.path.join(final_checkpoint_dir, "tokenizer_config.json")):
#      logger.info("Saving tokenizer separately...")
#      tokenizer.save_pretrained(final_checkpoint_dir)

print("\nPhase 1 (DivPO Training) Script Section Complete.")


# ## 10. (Optional) Test Generation

# In[12]:


# Optional: Test Generation Cell (uncomment code block to run)
'''
from transformers import pipeline, TextGenerationPipeline, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import logging

logger = logging.getLogger(__name__)

final_checkpoint_dir_to_test = "./divpo_dat_model_phase1/final_checkpoint" # Or path to another checkpoint

logger.info(f"Loading trained model from {final_checkpoint_dir_to_test} for testing...")

if not os.path.exists(final_checkpoint_dir_to_test):
    logger.error(f"Checkpoint directory not found: {final_checkpoint_dir_to_test}")
else:
    # Determine device for inference
    test_mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    test_cuda_available = torch.cuda.is_available()
    inference_device_type = "mps" if test_mps_available else ("cuda" if test_cuda_available else "cpu")
    # Use device index 0 if multiple GPUs/MPS devices exist
    inference_device = f"{inference_device_type}:0" if inference_device_type != "cpu" else "cpu"

    logger.info(f"Setting inference device: {inference_device}")

    try:
        # Load the model and tokenizer again for a clean test pipeline
        test_model = AutoModelForCausalLM.from_pretrained(final_checkpoint_dir_to_test)
        test_tokenizer = AutoTokenizer.from_pretrained(final_checkpoint_dir_to_test)

        # Ensure pad token is set if needed (might be redundant if saved correctly)
        if test_tokenizer.pad_token is None:
            test_tokenizer.pad_token = test_tokenizer.eos_token
            test_tokenizer.pad_token_id = test_tokenizer.eos_token_id

        # Move model to the designated inference device
        test_model.to(inference_device)
        test_model.eval() # Set model to evaluation mode
        logger.info(f"Test model loaded onto {test_model.device}")

        # Use TextGenerationPipeline for more control if needed, or default pipeline
        # pipe = pipeline("text-generation", model=test_model, tokenizer=test_tokenizer, device=inference_device) # device arg might map model internally
        # Or manually ensure device placement if pipeline's device arg causes issues:
        pipe = TextGenerationPipeline(model=test_model, tokenizer=test_tokenizer, device=inference_device)


        test_prompt = "Generate a single word that is a common English noun (like a thing, object, or concept). Do not use proper nouns like names of people or places."
        # Use generation config consistent with training/expectations
        gen_config = GenerationConfig(
            max_new_tokens=MAX_TARGET_LENGTH, # Use config from training
            min_new_tokens=1,
            pad_token_id=test_tokenizer.pad_token_id,
            eos_token_id=test_tokenizer.eos_token_id,
            bos_token_id=test_tokenizer.bos_token_id,
            do_sample=True,
            temperature=0.7, # Can adjust temperature for testing
            top_k=50,
            repetition_penalty=1.1 # Slightly discourage repetition
        )

        num_sequences_to_generate = 10
        logger.info(f"Generating {num_sequences_to_generate} sequences...")
        with torch.no_grad(): # Ensure no gradients are calculated during inference
             generated_outputs = pipe(test_prompt, generation_config=gen_config, num_return_sequences=num_sequences_to_generate)

        print(f"\n--- Test Generation Results for prompt: '{test_prompt}' ---")
        generated_words = []
        for i, output in enumerate(generated_outputs):
            # Extract only the generated text part
            full_text = output['generated_text']
            # Find the end of the prompt to isolate the generated part
            # This assumes the prompt is present exactly at the beginning
            if full_text.startswith(test_prompt):
                 generated_part = full_text[len(test_prompt):].strip()
            else:
                 # Fallback if prompt isn't exactly at start (less reliable)
                 # Look for common instruction patterns if prompt is modified by model
                 output_marker = "Output:"
                 if output_marker in full_text:
                     generated_part = full_text.split(output_marker, 1)[-1].strip()
                 else:
                     generated_part = full_text.replace(test_prompt, "").strip() # Basic replacement

            # Further cleanup: take the first word, handle potential extra text
            first_word = generated_part.split()[0].strip(".,;:!?\"'") if generated_part.split() else "[empty]"
            generated_words.append(first_word.lower())
            print(f"{i+1}: '{first_word}' (Full Raw: '{generated_part}')")

        # Analyze results
        unique_words = set(generated_words) - set(['[empty]'])
        num_unique = len(unique_words)
        print(f"\nAnalysis: Generated {len(generated_words)} words, {num_unique} unique words.")
        print(f"Unique words: {sorted(list(unique_words))}")
        print("---------------------------------------------------------")

    except Exception as e:
        logger.error(f"Failed to run test generation: {e}", exc_info=True)
'''

