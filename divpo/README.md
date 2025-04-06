# DivPO: Diversity Preference Optimization

This is a refactored implementation of the DivPO (Diversity Preference Optimization) technique for language model fine-tuning. The code has been reorganized into a modular structure for better maintainability and extensibility.

## Code Structure

The codebase is organized into the following modules:

- `divpo/__init__.py` - Package initialization
- `divpo/config.py` - Configuration parameters for training
- `divpo/data.py` - Dataset preparation and data collation
- `divpo/model.py` - Model loading and configuration
- `divpo/trainer.py` - The core DivPODPOTrainer implementation
- `divpo/utils.py` - Utility functions for word quality and diversity
- `divpo/training.py` - High-level training orchestration functions
- `run_divpo.py` - Main script to execute the training process

## How to Use

1. Ensure all dependencies are installed:
```
pip install -r requirements.txt
```

2. Run the main script:
```
python run_divpo.py
```

This will:
- Set up NLP resources (NLTK, spaCy)
- Load the language model and tokenizer
- Prepare the dataset
- Initialize the DivPO trainer
- Run the training process
- Save the final model

## Customization

To customize the training process:

1. Modify parameters in `divpo/config.py` to change:
   - Model selection
   - Training hyperparameters
   - DivPO hyperparameters
   - Dataset size

2. Edit prompt templates in `divpo/config.py`:
   - Change `DEFAULT_PROMPT` to use different instructions

3. Adjust quality criteria in `divpo/utils.py`:
   - Modify `calculate_quality()` to change what makes a word "good"

## Implementation Details

The DivPO technique extends Direct Preference Optimization (DPO) by:

1. Generating K candidate outputs for each input prompt
2. Evaluating each candidate for:
   - Quality (is it a proper noun, common, etc.)
   - Diversity (how different is it from other candidates)
3. Selecting chosen/rejected pairs based on these metrics
4. Computing DPO loss on these pairs to train the model

The goal is to guide the model toward generating high-quality, diverse responses.

## Original Implementation

This is a refactored version of the original implementation. The original monolithic code is preserved in `Phase_1_DivPO.py`. 