#!/usr/bin/env python3
"""
Main script to run DivPO training
"""
import logging
import sys
import os
# Changed import to use the NLTK-only setup function
from divpo.utils import setup_nltk_only

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('divpo_training.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to orchestrate DivPO training"""
    logger.info("=== Starting DivPO Training Process ===")
    
    # Setup directory for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Set up NLTK resources and get name list
    logger.info("Setting up NLTK resources...")
    name_list = setup_nltk_only() # Now returns the loaded name list
    logger.info("NLTK resources ready.")
    
    # Step 2: Import after NLP setup to avoid circular imports
    from divpo.training import load_resources, setup_training, run_training, save_final_model
    
    # Step 3: Load other resources
    resources = load_resources()
    
    # Step 4: Set up training components, passing the name list
    training_components = setup_training(resources, name_list) # Pass name_list
    
    # Step 5: Run training
    training_result = run_training(training_components)
    
    # Step 6: Save final model
    save_final_model(training_components, training_result)
    
    logger.info("=== DivPO Training Process Complete ===")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1) 