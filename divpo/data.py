"""
Data preparation and collation for DivPO
"""
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def create_prompt_dataset(prompt: str, num_prompts: int) -> Dataset:
    """
    Create a dataset with repeated prompts for DivPO training.
    
    Args:
        prompt: The text prompt to repeat
        num_prompts: Number of times to repeat the prompt
        
    Returns:
        A HuggingFace Dataset containing the prompts
    """
    data = {"prompt": [prompt] * num_prompts}
    dataset = Dataset.from_dict(data)
    logger.info(f"Created dataset with {len(dataset)} prompt instances")
    return dataset


class DivPODataCollator:
    """
    Custom data collator for DivPO that works with prompt-only datasets.
    Prepares batches for the trainer by handling prompt tokenization and padding.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, padding: bool = True, max_length: int = None):
        """
        Initialize the data collator.
        
        Args:
            tokenizer: The tokenizer to use for encoding/decoding
            padding: Whether to pad sequences
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features into a format suitable for DivPO training.
        
        Args:
            features: List of features dictionaries from the dataset
            
        Returns:
            Dictionary with batched tensors
        """
        # Collect prompt inputs - we'll generate chosen/rejected on-the-fly
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