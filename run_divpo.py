#!/usr/bin/env python3
"""
Main script to run DivPO training
"""
import logging
import sys
import os
from setup_nlp_resources import setup_all_resources

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
    
    # Step 1: Set up NLP resources
    logger.info("Setting up NLP resources...")
    nlp = setup_all_resources()
    logger.info("NLP resources ready.")
    
    # Step 2: Import after NLP setup to avoid circular imports
    from divpo.training import load_resources, setup_training, run_training, save_final_model
    
    # Step 3: Load resources
    resources = load_resources()
    
    # Step 4: Set up training components
    training_components = setup_training(resources, nlp)
    
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