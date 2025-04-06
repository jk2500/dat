import nltk
import spacy
import ssl
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_ssl_context():
    """Fix SSL certificate verification issues for NLTK downloads."""
    logger.info("Setting up SSL context for downloads...")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        logger.info("SSL context modification not needed on this system.")
        return
    else:
        logger.info("Modified SSL context to allow downloads.")
        ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    logger.info("Checking/Downloading NLTK data...")
    
    # List of required NLTK resources
    nltk_resources = [
        ('corpora/words', 'words'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    for resource_path, resource_name in nltk_resources:
        try:
            nltk.data.find(resource_path)
            logger.info(f"- NLTK '{resource_name}' already downloaded.")
        except LookupError:
            logger.info(f"- Downloading NLTK '{resource_name}'...")
            nltk.download(resource_name, quiet=False)
    
    logger.info("NLTK data download check complete.")

def download_spacy_model(model_name='en_core_web_sm'):
    """Download and load the specified spaCy model if not already present."""
    logger.info(f"Checking/Downloading spaCy model '{model_name}'...")
    try:
        nlp = spacy.load(model_name)
        logger.info(f"- spaCy '{model_name}' model already installed.")
        return nlp
    except OSError:
        logger.info(f"- spaCy '{model_name}' model not found. Downloading...")
        spacy.cli.download(model_name)
        logger.info(f"- spaCy model '{model_name}' downloaded.")
        nlp = spacy.load(model_name)  # Load after downloading
        return nlp

def setup_all_resources():
    """Run all setup steps."""
    logger.info("Starting NLP resources setup...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Setup steps
    setup_ssl_context()
    download_nltk_resources()
    nlp = download_spacy_model()
    
    logger.info("NLP resources setup complete. All required models and data are downloaded.")
    return nlp

if __name__ == "__main__":
    setup_all_resources()
    print("\nSetup complete! You can now run the main DivPO training script.") 