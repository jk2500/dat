"""
Utility functions for DivPO
"""
import torch
import torch.nn.functional as F
import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def get_embeddings(words: List[str], embedding_model: SentenceTransformer, device: str) -> torch.Tensor:
    """
    Get embeddings for a list of words.
    
    Args:
        words: List of words to embed
        embedding_model: Sentence transformer model for embeddings
        device: Device to run embedding on ('cpu', 'cuda', 'mps')
        
    Returns:
        Tensor of embeddings with shape (len(words), embedding_dim)
    """
    if not words:
        return torch.empty((0, embedding_model.get_sentence_embedding_dimension()), 
                          device=device, dtype=torch.float32)
    
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
    Calculate diversity for a word based on its maximum cosine similarity
    to any *other* word in the batch.
    Diversity = 1 - max_similarity. Higher score means more distinct.

    Args:
        idx: Index of the word in the embeddings tensor
        embeddings: Tensor of shape (num_words, embedding_dim)

    Returns:
        Diversity score (float) in range [0, 1]
    """
    if embeddings.shape[0] <= 1:
        return 0.0  # No diversity if only one word

    current_embedding = embeddings[idx].unsqueeze(0)  # Shape: (1, dim)
    other_embeddings = torch.cat([embeddings[:idx], embeddings[idx+1:]], dim=0)  # Shape: (num_words-1, dim)

    # Calculate cosine similarities
    current_embedding_norm = F.normalize(current_embedding, p=2, dim=1)
    other_embeddings_norm = F.normalize(other_embeddings, p=2, dim=1)
    cosine_similarities = torch.mm(current_embedding_norm, other_embeddings_norm.t()).squeeze()

    # Handle case where there's only one 'other' word (returns scalar tensor)
    if cosine_similarities.dim() == 0:
        max_similarity = cosine_similarities.item()
    else:
        max_similarity = torch.max(cosine_similarities).item()

    diversity = 1.0 - max_similarity
    return diversity

def calculate_quality(word: str, nlp, common_vocab) -> float:
    """
    Checks if a single word meets the criteria (common, non-proper noun).
    
    Args:
        word: The word to evaluate
        nlp: Loaded spaCy model
        common_vocab: Set of common vocabulary words
        
    Returns:
        1.0 if word passes quality checks, 0.0 otherwise
    """
    word = str(word).strip()  # Ensure it's a string and stripped
    word_lower = word.lower()

    # 1. Format Check: Must be a single, non-empty, non-hyphenated word
    if not word or ' ' in word or '-' in word:
        print(f"Quality Fail: '{word}' Invalid format (space/hyphen/empty).")
        return 0.0

    # 2. Commonality Check: Must be in the common vocabulary list
    is_common = word_lower in common_vocab
    if not is_common:
        print(f"Quality Fail: '{word}' Not common enough.")
        return 0.0

    # 3. spaCy Processing & POS Tagging
    doc = nlp(word)
    if not doc or not doc[0]:
        print(f"Quality Fail: '{word}' SpaCy couldn't process.")
        return 0.0

    token = doc[0]
    pos_tag = token.tag_

    # 4. Noun Type Check
    is_noun = pos_tag in ['NN']
    is_proper_noun = pos_tag in ['NNP']
    is_unexpectedly_capitalized = len(word) > 1 and word[0].isupper() and not is_proper_noun

    # 5. Final Decision
    if is_noun and not is_proper_noun and not is_unexpectedly_capitalized and is_common:
        print(f"Quality OK: '{word}' Noun:{is_noun}, Proper:{is_proper_noun}, Common:{is_common}, Cap: {is_unexpectedly_capitalized}")
        return 1.0
    else:
        print(f"Quality Fail: '{word}' Noun:{is_noun}, Proper:{is_proper_noun}, Common:{is_common}, Cap: {is_unexpectedly_capitalized}")
        return 0.0 