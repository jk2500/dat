"""
Utility functions for DivPO
"""
import torch
import torch.nn.functional as F
import logging
from typing import List, Set
from sentence_transformers import SentenceTransformer
# Removed spacy import as it's no longer used for quality check
import nltk
import ssl
import os

# Define logger at the top
logger = logging.getLogger(__name__)

# Remove global loading of name_list - will be loaded in setup_nltk_only

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

def calculate_quality(word: str, name_list: Set[str], common_vocab=None, nlp=None) -> float:
    """
    Checks if a single word is likely a common noun using WordNet.
    It must exist in WordNet primarily as a NOUN and not be in the passed name_list.
    Args:
        word: The word to evaluate
        name_list: Pre-loaded set of known lowercased names.
        nlp: spaCy model (UNUSED)
        common_vocab: Set of common vocabulary words (UNUSED)
    Returns:
        1.0 if word passes quality checks, 0.0 otherwise
    """
    word = str(word).strip()
    word_lower = word.lower()

    # 1. Format Check
    if not word or ' ' in word or '-' in word:
        print(f"Quality Fail: '{word}' Invalid format (space/hyphen/empty).")
        return 0.0

    # 2. WordNet Lookup and Noun Check
    is_common_noun_in_wordnet = False
    try:
        from nltk.corpus import wordnet as wn # Import here to ensure it's available after download
        syns = wn.synsets(word_lower)
        if syns and syns[0].pos() == wn.NOUN:
             is_common_noun_in_wordnet = True
    except ImportError:
        print(f"Quality Check Skip: NLTK/WordNet not available for '{word}'")
        return 0.0 # Cannot perform check, fail
    except LookupError:
        print(f"Quality Check Skip: WordNet corpus not downloaded for '{word}'")
        return 0.0 # Cannot perform check, fail
    except Exception as e:
        logger.error(f"WordNet error processing word '{word}': {e}", exc_info=False)
        return 0.0 # Fail on error

    # 3. Gazetteer Check (Is it in our list of known first names?)
    # Use the name_list passed as argument
    is_known_name = word_lower in name_list

    # 4. Final Decision
    if is_common_noun_in_wordnet and not is_known_name:
        print(f"Quality OK: '{word}' WordNetNoun:{is_common_noun_in_wordnet}, KnownName:{is_known_name}")
        return 1.0
    else:
        print(f"Quality Fail: '{word}' WordNetNoun:{is_common_noun_in_wordnet}, KnownName:{is_known_name}")
        return 0.0


# --- NLP Resource Setup Functions (Moved from setup_nlp_resources.py) ---

def setup_ssl_context():
    """Fix SSL certificate verification issues for NLTK downloads."""
    # logger.info("Setting up SSL context for downloads...") # Less verbose
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # logger.info("SSL context modification not needed on this system.") # Less verbose
        return
    else:
        # logger.info("Modified SSL context to allow downloads.") # Less verbose
        ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    logger.info("Checking/Downloading NLTK data...")
    
    # List of required NLTK resources
    nltk_resources = [
        ('corpora/words', 'words'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/names', 'names'),
        ('corpora/wordnet', 'wordnet') # Added wordnet corpus
    ]
    
    all_downloaded = True
    for resource_path, resource_name in nltk_resources:
        try:
            nltk.data.find(resource_path)
            logger.info(f"- NLTK '{resource_name}' already downloaded.")
        except LookupError:
            all_downloaded = False
            logger.info(f"- Downloading NLTK '{resource_name}'...")
            try:
                nltk.download(resource_name, quiet=False)
            except Exception as e:
                logger.error(f"Failed to download NLTK resource '{resource_name}': {e}")
                # Consider raising an error or exiting if essential resources fail
    
    if all_downloaded:
        logger.info("NLTK data download check complete (all resources present).")
    else:
        logger.info("NLTK data download check complete.")

def setup_nltk_only() -> Set[str]:
    """Runs setup for NLTK resources and loads/returns the name list."""
    logger.info("Starting NLTK resources setup...")
    setup_ssl_context()
    download_nltk_resources()
    logger.info("NLTK resource download check complete.")

    # Load name list after ensuring download
    name_list_loaded = set()
    try:
        from nltk.corpus import names
        name_list_loaded = set(names.words('male.txt') + names.words('female.txt'))
        name_list_loaded = {name.lower() for name in name_list_loaded}
        logger.info(f"Loaded NLTK name list with {len(name_list_loaded)} names.")
    except LookupError:
        logger.error("NLTK 'names' corpus failed to load even after download attempt. Name check will be skipped.")
    except Exception as e:
        logger.error(f"Error loading NLTK name list after download: {e}", exc_info=True)

    return name_list_loaded 