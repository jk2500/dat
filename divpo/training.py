"""
Training orchestration for DivPO
"""
import torch
import os
import logging
import nltk
from wordfreq import top_n_list
from sentence_transformers import SentenceTransformer
from trl import DPOConfig
import warnings
from typing import Dict, Any, Optional, Set

from divpo.config import (
    MODEL_NAME, OUTPUT_DIR, K_SAMPLES, DPO_BETA, 
    PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, NUM_TRAIN_EPOCHS, LOGGING_STEPS, SAVE_STEPS,
    MAX_PROMPT_LENGTH, MAX_TARGET_LENGTH, 
    EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, DEFAULT_PROMPT,
    calculate_num_prompts, EFFECTIVE_BATCH_SIZE
)
from divpo.model import load_tokenizer, load_model
from divpo.data import create_prompt_dataset, DivPODataCollator
from divpo.utils import calculate_quality, calculate_pairwise_semantic_diversity
from divpo.trainer import DivPODPOTrainer

# Set up logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is now deprecated.*")


def load_resources():
    """
    Load NLP resources and embedding model.
    
    Returns:
        Dictionary with loaded resources
    """
    logger.info("Loading NLP tools and embedding model...")
    
    try:
        # Load word lists - using entire English vocabulary instead of common words
        english_vocab = set(w.lower() for w in nltk.corpus.words.words())
        
        # Load embedding model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
        logger.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded on {EMBEDDING_DEVICE}")
        
        return {
            'english_vocab': english_vocab,
            'common_vocab': english_vocab,  # Use entire English vocab for quality check
            'embedding_model': embedding_model
        }
    
    except Exception as e:
        logger.error(f"Failed to load resources: {e}", exc_info=True)
        raise


def setup_training(resources: Dict[str, Any], name_list: Set[str]) -> Dict[str, Any]:
    """
    Set up all components needed for training.
    
    Args:
        resources: Dictionary of NLP resources
        name_list: Set of known lowercased names
        
    Returns:
        Dictionary with training components
    """
    # Load tokenizer and model
    tokenizer = load_tokenizer(MODEL_NAME)
    model, model_device = load_model(MODEL_NAME)
    
    # Prepare dataset
    num_prompts = calculate_num_prompts()
    train_dataset = create_prompt_dataset(DEFAULT_PROMPT, num_prompts)
    
    # Create quality function with bound parameters
    def quality_fn(word: str) -> float:
        return calculate_quality(word, name_list, resources['common_vocab'])
    
    # Initialize training arguments
    use_mps_for_training = model_device == "mps"
    
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        remove_unused_columns=False,
        report_to="none",
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        seed=42,
        save_total_limit=2,
        logging_first_step=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        beta=DPO_BETA,
        model_init_kwargs=None,
    )
    
    logger.info(f"DPO Config configured. Effective batch size: {EFFECTIVE_BATCH_SIZE}. Using MPS for training: {use_mps_for_training}")
    
    # Initialize trainer
    trainer = DivPODPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DivPODataCollator(tokenizer, max_length=MAX_PROMPT_LENGTH + MAX_TARGET_LENGTH),  
        eval_dataset=None,
        # Custom DivPO args
        k_samples=K_SAMPLES,
        quality_fn=quality_fn,
        diversity_fn=calculate_pairwise_semantic_diversity,
        embedding_model=resources['embedding_model'],
        embedding_device=EMBEDDING_DEVICE,
        # DPO Trainer specific args
        max_length=MAX_PROMPT_LENGTH + MAX_TARGET_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
        beta=DPO_BETA,
    )
    
    logger.info("DivPO Trainer initialized successfully")
    
    return {
        'tokenizer': tokenizer,
        'model': model,
        'trainer': trainer,
        'training_args': training_args,
        'model_device': model_device
    }


def run_training(training_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the DivPO training process.
    
    Args:
        training_components: Dictionary with training components
        
    Returns:
        Training results and metrics
    """
    trainer = training_components['trainer']
    
    logger.info("Starting DivPO Training...")
    try:
        # Run training
        train_result = trainer.train()
        logger.info("Training finished successfully.")
        
        # Log training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        return {
            'success': True,
            'metrics': metrics
        }
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        
        # Try to save model even if training failed
        error_save_dir = os.path.join(OUTPUT_DIR, "checkpoint_on_error")
        logger.info(f"Attempting to save model state to {error_save_dir} after error...")
        
        try:
            trainer.save_model(error_save_dir)
            logger.info(f"Model saved to {error_save_dir}")
        except Exception as save_e:
            logger.error(f"Could not save model after error: {save_e}", exc_info=True)
        
        return {
            'success': False,
            'error': str(e)
        }


def save_final_model(training_components: Dict[str, Any], 
                    training_result: Dict[str, Any]) -> None:
    """
    Save the final model after training.
    
    Args:
        training_components: Dictionary with training components
        training_result: Results from training
    """
    trainer = training_components['trainer']
    
    logger.info("Saving final model checkpoint...")
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    
    try:
        trainer.save_model(final_checkpoint_dir)
        logger.info(f"Final model, tokenizer, and config saved to {final_checkpoint_dir}")
    except Exception as e:
        logger.error(f"Failed to save final model checkpoint: {e}", exc_info=True)
        
    # Print completion message
    if training_result.get('success', False):
        logger.info("\nDivPO Training completed successfully.")
    else:
        logger.warning("\nDivPO Training ended with issues. See logs for details.") 