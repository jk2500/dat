"""
Model handling functionality for DivPO
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer
)
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load and configure the tokenizer for DivPO training.
    
    Args:
        model_name: Name or path of the model/tokenizer
        
    Returns:
        Configured tokenizer
    """
    logger.info(f"Loading tokenizer for: {model_name}")
    
    # Load tokenizer with left padding for generation
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='left'
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        logger.info("Setting pad token to eos token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    logger.info(f"Tokenizer loaded. Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    return tokenizer

def load_model(model_name: str) -> Tuple[torch.nn.Module, str]:
    """
    Load model and move it to the appropriate device.
    
    Args:
        model_name: Name or path of the model
        
    Returns:
        Tuple of (model, device name)
    """
    # Define model loading arguments - start by loading on CPU
    model_kwargs = {
        "torch_dtype": torch.float32,  # Use float32 for MPS compatibility
        "device_map": "cpu",           # Load onto CPU initially
        "trust_remote_code": True,     # Required for some models
    }

    # Check for available compute devices
    mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    cuda_available = torch.cuda.is_available()
    model_device = "cpu"  # Default device

    # Load the model
    logger.info(f"Loading model '{model_name}' onto CPU...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info("Model loaded on CPU.")

    # Move model to MPS if available
    if mps_available:
        logger.info("Attempting to move model to Apple Silicon (MPS) backend...")
        try:
            model = model.to("mps")
            model_device = "mps"
            logger.info("Model successfully moved to MPS.")
        except Exception as e:
            logger.error(f"Failed to move model to MPS: {e}. Keeping model on CPU.", exc_info=True)
            mps_available = False
    elif cuda_available:
        logger.info("CUDA available, but this setup prioritizes MPS/CPU. Keeping model on CPU.")
    else:
        logger.info("MPS not available. Keeping model on CPU.")

    logger.info(f"Model final compute device: {model.device}")
    return model, model_device 