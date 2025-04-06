from transformers import pipeline, TextGenerationPipeline, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import logging
import os
import sys # Added for path manipulation if needed, though not strictly required here

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants (copied from Phase_1_DivPO.py) ---
# You might want to load these from a config file or pass as arguments in a real application
MAX_TARGET_LENGTH = 10 # Max *new* tokens for the generated word (used in GenerationConfig)

# --- Main Script ---
final_checkpoint_dir_to_test = "./divpo_dat_model_phase1/final_checkpoint" # Or path to another checkpoint

logger.info(f"Loading trained model from {final_checkpoint_dir_to_test} for testing...")

if not os.path.exists(final_checkpoint_dir_to_test):
    logger.error(f"Checkpoint directory not found: {final_checkpoint_dir_to_test}")
else:
    # Determine device for inference
    test_mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    test_cuda_available = torch.cuda.is_available()
    inference_device_type = "mps" if test_mps_available else ("cuda" if test_cuda_available else "cpu")
    # Use device index 0 if multiple GPUs/MPS devices exist
    inference_device = f"{inference_device_type}:0" if inference_device_type != "cpu" else "cpu"

    logger.info(f"Setting inference device: {inference_device}")

    try:
        # Load the model and tokenizer again for a clean test pipeline
        test_model = AutoModelForCausalLM.from_pretrained(final_checkpoint_dir_to_test)
        test_tokenizer = AutoTokenizer.from_pretrained(final_checkpoint_dir_to_test)

        # Ensure pad token is set if needed (might be redundant if saved correctly)
        if test_tokenizer.pad_token is None:
            test_tokenizer.pad_token = test_tokenizer.eos_token
            test_tokenizer.pad_token_id = test_tokenizer.eos_token_id

        # Move model to the designated inference device
        test_model.to(inference_device)
        test_model.eval() # Set model to evaluation mode
        logger.info(f"Test model loaded onto {test_model.device}")

        # Use TextGenerationPipeline for more control if needed, or default pipeline
        pipe = TextGenerationPipeline(model=test_model, tokenizer=test_tokenizer, device=inference_device)


        base_prompt_text = "Generate a english single word. Do not use proper nouns like names of people or places. Just generate a single word."

        # --- Apply Chat Template ---
        # Create the message structure expected by the template
        messages = [
            {"role": "user", "content": base_prompt_text}
            # Add system prompt if needed/used during training, e.g.:
            # {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": base_prompt_text}
        ]
        # Apply the template. Ensure add_generation_prompt=True to get the prompt ending with the assistant turn starter.
        try:
            test_prompt_formatted = test_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # Crucial to prepare for generation
            )
            logger.info("Applied chat template to test prompt.")
            print(f"--- Formatted Prompt for Testing ---\n{test_prompt_formatted}\n---------------------------------")
        except Exception as e:
            logger.warning(f"Could not apply chat template automatically: {e}. Using raw prompt string. Output might be suboptimal.")
            test_prompt_formatted = base_prompt_text # Fallback to raw prompt
        # --- End Apply Chat Template ---


        # Use generation config consistent with training/expectations
        # Consider aligning temperature/top_k with training if template doesn't fix it
        gen_config = GenerationConfig(
            max_new_tokens=MAX_TARGET_LENGTH, # Use config from training
            min_new_tokens=1,
            pad_token_id=test_tokenizer.pad_token_id,
            # Ensure EOS token handling matches the model's template/fine-tuning
            # Qwen often uses <|im_end|> (ID: 151645) as a stop token.
            # Use the tokenizer's configured eos_token_id if possible.
            eos_token_id=test_tokenizer.eos_token_id, # Trust the loaded tokenizer's EOS ID
            bos_token_id=test_tokenizer.bos_token_id,
            do_sample=True,
            temperature=0.7, # Keep test temperature for now
            top_k=50,        # Keep test top_k for now
            repetition_penalty=1.1
        )

        num_sequences_to_generate = 200
        logger.info(f"Generating {num_sequences_to_generate} sequences using formatted prompt...")
        with torch.no_grad(): # Ensure no gradients are calculated during inference
             # --- Use formatted prompt ---
             generated_outputs = pipe(test_prompt_formatted, generation_config=gen_config, num_return_sequences=num_sequences_to_generate)
             # --- End Use formatted prompt ---

        print(f"\n--- Test Generation Results for prompt: '{base_prompt_text}' ---") # Print original base prompt for clarity
        generated_words = []
        for i, output in enumerate(generated_outputs):
            # Extract only the generated text part
            full_text = output['generated_text']
            # Find the end of the *formatted* prompt to isolate the generated part
            # This assumes the formatted prompt is present exactly at the beginning
            if full_text.startswith(test_prompt_formatted):
                 generated_part = full_text[len(test_prompt_formatted):].strip()
            else:
                 # Fallback: Try removing the base prompt text if formatted prompt isn't exact match
                 # This might happen if pipeline adds/removes spaces etc.
                 if base_prompt_text in full_text:
                      # Find the last occurrence of the base prompt text and take text after it
                      start_index = full_text.rfind(base_prompt_text) + len(base_prompt_text)
                      # Also need to account for potential template tokens like <|im_end|><|im_start|>assistant\n
                      # A simpler, though less robust, fallback is just basic replacement again
                      generated_part = full_text.replace(test_prompt_formatted, "").strip() # Less reliable fallback
                      logger.warning(f"Formatted prompt not found exactly at start of output {i+1}. Using basic replacement fallback.")

                 else:
                      generated_part = full_text # Cannot reliably remove prompt
                      logger.warning(f"Could not reliably remove prompt from output {i+1}.")


            # --- MODIFIED CLEANUP (like training) ---
            # Take the *last* word and strip common punctuation
            words_in_generated = generated_part.split()
            if words_in_generated:
                # Get the last element
                last_word_candidate = words_in_generated[-1]
                # Strip common leading/trailing punctuation
                cleaned_word = last_word_candidate.strip(".,;:!?\"'()[]{}<>")
            else:
                # If splitting results in no words
                cleaned_word = "[empty]"
            # --- END MODIFIED CLEANUP ---

            # Use the cleaned_word (last word) for analysis
            generated_words.append(cleaned_word.lower())
            print(f"{i+1}: '{cleaned_word}' (Full Raw: '{generated_part}')") # Print the cleaned word

        # Analyze results based on the extracted last words
        unique_words = set(w for w in generated_words if w != '[empty]') # Filter empty results
        num_unique = len(unique_words)
        print(f"\nAnalysis: Generated {len(generated_words)} words, {num_unique} unique words.")
        print(f"Unique words: {sorted(list(unique_words))}")
        print("---------------------------------------------------------")

    except Exception as e:
        logger.error(f"Failed to run test generation: {e}", exc_info=True)
