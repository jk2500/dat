from transformers import pipeline, TextGenerationPipeline, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import torch.nn.functional as F
import logging
import os
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Set, Tuple

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
MAX_TARGET_LENGTH = 10
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # The original, untrained model
TRAINED_MODEL_PATH = "./divpo_dat_model_phase1/final_checkpoint"
BASE_PROMPT = "Generate a single english common noun. Do not use proper nouns like names of people or places. Just generate a single common noun."
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Same as used in training

def calculate_semantic_metrics(words: List[str], embedding_model: SentenceTransformer) -> Tuple[float, float]:
    """Calculate average and minimum pairwise semantic similarities."""
    if len(words) < 2:
        return 0.0, 0.0
    
    # Get embeddings
    embeddings = embedding_model.encode(words, convert_to_tensor=True)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Calculate pairwise similarities
    similarities = torch.mm(embeddings, embeddings.t())
    
    # Mask out self-similarities
    mask = torch.ones_like(similarities) - torch.eye(similarities.shape[0], device=similarities.device)
    masked_similarities = similarities * mask
    
    # Calculate metrics
    avg_similarity = masked_similarities.sum() / (len(words) * (len(words) - 1))
    min_similarity = masked_similarities[masked_similarities > 0].min()
    
    return avg_similarity.item(), min_similarity.item()

def analyze_diversity(words: List[str], model_name: str, embedding_model: SentenceTransformer):
    """Analyze semantic diversity within a set of generated words."""
    print(f"\n=== Semantic Diversity Analysis for {model_name} ===")
    
    if len(words) < 2:
        print("Not enough words for diversity analysis")
        return
    
    # Get embeddings
    embeddings = embedding_model.encode(words, convert_to_tensor=True)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Calculate pairwise similarities
    similarities = torch.mm(embeddings, embeddings.t())
    
    # Mask out self-similarities
    mask = torch.ones_like(similarities) - torch.eye(similarities.shape[0], device=similarities.device)
    masked_similarities = similarities * mask
    
    # Calculate metrics
    avg_similarity = masked_similarities.sum() / (len(words) * (len(words) - 1))
    min_similarity = masked_similarities[masked_similarities > 0].min()
    
    print(f"Number of unique words: {len(words)}")
    print(f"Average pairwise similarity: {avg_similarity.item():.3f}")
    print(f"Minimum pairwise similarity: {min_similarity.item():.3f}")
    print(f"Semantic diversity score: {1 - avg_similarity.item():.3f}")
    print("-" * 80)

def setup_model_and_tokenizer(model_path_or_name):
    """Set up model and tokenizer for testing."""
    logger.info(f"Loading model from {model_path_or_name}...")
    
    # Determine device for inference
    test_mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    test_cuda_available = torch.cuda.is_available()
    inference_device_type = "cuda" if test_cuda_available else ("mps" if test_mps_available else "cpu")
    inference_device = f"{inference_device_type}:0" if inference_device_type != "cpu" else "cpu"
    
    logger.info(f"Setting inference device: {inference_device}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path_or_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model.to(inference_device)
        model.eval()
        logger.info(f"Model loaded onto {model.device}")
        
        return model, tokenizer, inference_device
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return None, None, None

def run_generation_test(model, tokenizer, device, model_name="unnamed_model"):
    """Run generation test with given model and tokenizer."""
    try:
        pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
        
        # Format prompt with chat template
        try:
            messages = [{"role": "user", "content": BASE_PROMPT}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.info("Applied chat template to prompt.")
        except Exception as e:
            logger.warning(f"Could not apply chat template: {e}. Using raw prompt.")
            formatted_prompt = BASE_PROMPT
        
        gen_config = GenerationConfig(
            max_new_tokens=MAX_TARGET_LENGTH,
            min_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            repetition_penalty=1.1
        )
        
        num_sequences = 200
        logger.info(f"Generating {num_sequences} sequences using {model_name}...")
        
        with torch.no_grad():
            outputs = pipe(formatted_prompt, generation_config=gen_config, num_return_sequences=num_sequences)
        
        print(f"\n--- Generation Results for {model_name} ---")
        generated_words = []
        
        for i, output in enumerate(outputs):
            full_text = output['generated_text']
            
            if full_text.startswith(formatted_prompt):
                generated_part = full_text[len(formatted_prompt):].strip()
            else:
                generated_part = full_text.replace(formatted_prompt, "").strip()
            
            words_in_generated = generated_part.split()
            if words_in_generated:
                last_word = words_in_generated[-1].strip(".,;:!?\"'()[]{}<>")
            else:
                last_word = "[empty]"
            
            generated_words.append(last_word.lower())
            print(f"{i+1}: '{last_word}' (Full: '{generated_part}')")
        
        unique_words = set(w for w in generated_words if w != '[empty]')
        print(f"\nAnalysis for {model_name}:")
        print(f"Total words generated: {len(generated_words)}")
        print(f"Unique words: {len(unique_words)}")
        print(f"All unique words: {sorted(list(unique_words))}")
        print("-" * 80)
        
        return generated_words, unique_words
    
    except Exception as e:
        logger.error(f"Failed to run generation test: {e}", exc_info=True)
        return [], set()

def main():
    """Run baseline and trained model tests."""
    # Load embedding model
    logger.info(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    embedding_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=embedding_device)
    
    # Test baseline model
    logger.info("Testing baseline (untrained) model...")
    base_model, base_tokenizer, base_device = setup_model_and_tokenizer(BASE_MODEL_NAME)
    base_words, base_unique = [], set()
    if base_model is not None:
        base_words, base_unique = run_generation_test(
            base_model, base_tokenizer, base_device, "baseline_model"
        )
        analyze_diversity(list(base_unique), "Baseline Model", embedding_model)
    
    # Test trained model if available
    trained_words, trained_unique = [], set()
    if os.path.exists(TRAINED_MODEL_PATH):
        logger.info("Testing trained model...")
        trained_model, trained_tokenizer, trained_device = setup_model_and_tokenizer(TRAINED_MODEL_PATH)
        if trained_model is not None:
            trained_words, trained_unique = run_generation_test(
                trained_model, trained_tokenizer, trained_device, "trained_model"
            )
            
            # Compare results
            print("\n=== Comparison ===")
            print(f"Baseline unique words: {len(base_unique)}")
            print(f"Trained unique words: {len(trained_unique)}")
            print(f"Words only in baseline: {sorted(base_unique - trained_unique)}")
            print(f"Words only in trained: {sorted(trained_unique - base_unique)}")
            print(f"Words in both: {sorted(base_unique & trained_unique)}")
            
            # Analyze semantic diversity
            analyze_diversity(list(trained_unique), "Trained Model", embedding_model)
    else:
        logger.warning(f"Trained model not found at {TRAINED_MODEL_PATH}")

if __name__ == "__main__":
    main()
