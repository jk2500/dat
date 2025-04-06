"""
DivPO Trainer implementation
"""
import torch
import torch.nn.functional as F
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from trl import DPOTrainer, DPOConfig
from torch.nn.parallel import DistributedDataParallel
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from divpo.utils import get_embeddings

logger = logging.getLogger(__name__)

# Type Hint for clarity
ModelType = Union[PreTrainedModel, torch.nn.Module, DistributedDataParallel]

class DivPODPOTrainer(DPOTrainer):
    """
    DivPO Trainer - extends DPOTrainer to implement Diversity Preference Optimization
    """
    
    def __init__(
        self,
        # Custom arguments first
        k_samples: int = 16,
        quality_fn: Callable[[str], float] = None,
        diversity_fn: Callable[[int, torch.Tensor], float] = None,
        embedding_model: SentenceTransformer = None,
        embedding_device: Union[str, torch.device] = None,
        # Arguments for the parent DPOTrainer
        model: ModelType = None,
        ref_model: Optional[ModelType] = None,
        args: Optional[DPOConfig] = None,
        train_dataset: Optional[Dataset] = None,
        # Use processing_class instead of tokenizer
        processing_class=None,  # This replaces tokenizer
        data_collator=None,
        eval_dataset=None,
        # Other parameters with defaults
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        beta: float = 0.1,
        **kwargs
    ):
        # Store custom attributes first
        self.k_samples = k_samples
        self._quality_fn = quality_fn
        self._diversity_fn = diversity_fn
        self._embedding_model = embedding_model
        self._embedding_device = embedding_device
        
        # Store parameters that DPOTrainer expects
        self.beta = beta
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        
        # Call parent init with the parameters it expects
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,  # Use processing_class instead of tokenizer
            data_collator=data_collator,
            **kwargs
        )
        
        # Check that required custom args were provided
        if self._quality_fn is None or self._diversity_fn is None or self._embedding_model is None or self._embedding_device is None:
            raise ValueError("quality_fn, diversity_fn, embedding_model, and embedding_device must be provided")

        logger.info(f"DivPO Trainer Initialized with K={self.k_samples}, Beta={self.beta}")
        logger.info("Ensure processing_class (tokenizer) has pad_token set correctly.")
        if self.processing_class.pad_token is None:
            logger.warning("Processing class pad_token is None. Setting to eos_token.")
            self.processing_class.pad_token = self.processing_class.eos_token
            self.processing_class.pad_token_id = self.processing_class.eos_token_id

    def compute_loss(
        self,
        model: ModelType,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the DivPO loss by:
        1. Generate K samples for each prompt in the batch
        2. Score each sample for quality and diversity
        3. Select chosen/rejected pairs based on scores
        4. Compute DPO loss on these pairs
        
        Args:
            model: The model to compute loss for
            inputs: The model inputs (prompt_input_ids, prompt_attention_mask)
            return_outputs: Whether to return outputs in addition to loss
            num_items_in_batch: Optional batch size hint
            
        Returns:
            The loss tensor or tuple of (loss, outputs)
        """
        # Ensure reference model is ready
        if self.ref_model is None:
             logger.error("Reference model not found/loaded by DPOTrainer!")
             raise ValueError("Reference model not loaded!")
        self.ref_model.to(self.accelerator.device)

        # Get prompts from inputs
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]

        # Determine the actual prompt length (excluding padding)
        if self.processing_class.padding_side == "right":
            prompt_lens = torch.sum(prompt_attention_mask, dim=1)
        else: # Left padding
             prompt_lens = (prompt_input_ids != self.processing_class.pad_token_id).sum(-1)
             logger.warning("Left padding detected. Ensure generation and logprob logic handles it correctly.")

        batch_size = prompt_input_ids.size(0)

        all_policy_chosen_logps = []
        all_policy_rejected_logps = []
        all_ref_chosen_logps = []
        all_ref_rejected_logps = []
        valid_pairs_found_in_batch = 0
        batch_chosen_words = []
        batch_rejected_words = []

        # --- Generation Phase ---
        # Generation config
        generation_config = GenerationConfig(
            max_new_tokens=self.max_target_length,
            min_new_tokens=1,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            do_sample=True,
            temperature=1.5,
            top_k=75,
        )

        # Prepare inputs for batch generation (repeat prompts K times)
        model_compute_device = model.device
        batch_prompt_ids = prompt_input_ids.repeat_interleave(self.k_samples, dim=0).to(model_compute_device)
        batch_prompt_mask = prompt_attention_mask.repeat_interleave(self.k_samples, dim=0).to(model_compute_device)
        prompt_lens_repeated = prompt_lens.repeat_interleave(self.k_samples)

        print(f"Generating {batch_prompt_ids.shape[0]} samples (Batch Size {batch_size} * K {self.k_samples}) on device {model_compute_device}...")
        with torch.no_grad():
            policy_outputs = model.generate(
                input_ids=batch_prompt_ids,
                attention_mask=batch_prompt_mask,
                generation_config=generation_config,
            )
        print("Generation complete.")

        # Move outputs to CPU for decoding
        policy_outputs_cpu = policy_outputs.cpu()

        # Decode generated tokens, extracting the last word from each generation
        all_candidate_words_flat = []
        for i in range(policy_outputs_cpu.shape[0]):
            original_prompt_len = prompt_lens_repeated[i].item()
            generated_tokens = policy_outputs_cpu[i, original_prompt_len:]
            decoded_text = self.processing_class.decode(generated_tokens, skip_special_tokens=True).strip()

            # Extract the last word and clean it
            words_in_decoded = decoded_text.split()
            if words_in_decoded:
                last_word_candidate = words_in_decoded[-1]
                cleaned_word = last_word_candidate.strip(".,;:!?\"'()[]{}<>")
            else:
                cleaned_word = ""
                
            all_candidate_words_flat.append(cleaned_word)

        # Print generated words
        print(f"--- Generated words for batch (Step {self.state.global_step if self.state else 'N/A'}): ---")
        print(all_candidate_words_flat)
        print(f"--- End Generated words ---")

        # --- Scoring and Pair Selection Phase ---
        # Process each original prompt's K samples
        for i in range(batch_size):
            start_idx = i * self.k_samples
            end_idx = start_idx + self.k_samples
            candidate_words_for_prompt = all_candidate_words_flat[start_idx:end_idx]

            # Filter out empty strings and lowercase
            candidate_words = [w for w in candidate_words_for_prompt if w]
            candidate_words = [w.lower() for w in candidate_words]

            if not candidate_words:
                print(f"Batch item {i}: No valid words generated after initial cleanup.")
                continue

            # Remove duplicates
            unique_candidate_words = list(OrderedDict.fromkeys(candidate_words))
            num_candidates = len(unique_candidate_words)

            if num_candidates == 0:
                print(f"Batch item {i}: No unique non-empty words after cleanup.")
                continue

            print(f"Batch item {i}: Processing {num_candidates} unique candidates: {unique_candidate_words[:5]}...")

            # Score Candidates (Quality)
            qualities_list = [self._quality_fn(w) for w in unique_candidate_words]
            qualities = torch.tensor(qualities_list, device=self.accelerator.device)

            # Score Candidates (Diversity)
            diversities = torch.zeros(num_candidates, device=self.accelerator.device)
            if num_candidates > 1:
                 try:
                    # Calculate embeddings
                    embeddings = get_embeddings(unique_candidate_words, self._embedding_model, self._embedding_device)
                    embeddings = embeddings.to(self.accelerator.device)
                    diversities_list = [self._diversity_fn(j, embeddings) for j in range(num_candidates)]
                    diversities = torch.tensor(diversities_list, device=self.accelerator.device)
                 except Exception as e:
                    logger.error(f"Error getting embeddings/diversity for batch item {i}: {e}", exc_info=True)

            # Selection Logic
            y_chosen_str, y_rejected_str = None, None
            chosen_idx_in_unique, rejected_idx_in_unique = -1, -1

            # Find good quality candidates (>= 1.0) and bad quality candidates (< 1.0)
            qualities_tensor = torch.tensor(qualities_list, device=self._embedding_device)
            idx_good_quality = torch.where(qualities_tensor >= 1.0)[0]
            idx_bad_quality = torch.where(qualities_tensor < 1.0)[0]

            # Select chosen word (highest diversity among good quality)
            if len(idx_good_quality) >= 1:
                good_diversities = diversities[idx_good_quality]
                max_diversity_val = torch.max(good_diversities)
                potential_chosen_indices_rel = torch.where(good_diversities == max_diversity_val)[0]
                # Randomly select one if there are ties
                chosen_rel_idx = potential_chosen_indices_rel[torch.randint(len(potential_chosen_indices_rel), (1,)).item()]
                chosen_idx_in_unique = idx_good_quality[chosen_rel_idx].item()
                y_chosen_str = unique_candidate_words[chosen_idx_in_unique]
            else:
                 print(f"Batch item {i}: No good quality words found to select a chosen word.")

            # Select rejected word (lowest diversity among bad quality)
            if len(idx_bad_quality) >= 1:
                if num_candidates > 1:
                    bad_diversities = diversities[idx_bad_quality]
                    min_diversity_val = torch.min(bad_diversities)
                    potential_rejected_indices_rel = torch.where(bad_diversities == min_diversity_val)[0]
                    rejected_rel_idx = potential_rejected_indices_rel[torch.randint(len(potential_rejected_indices_rel), (1,)).item()]
                    rejected_idx_in_unique = idx_bad_quality[rejected_rel_idx].item()
                    y_rejected_str = unique_candidate_words[rejected_idx_in_unique]
                else:
                    rejected_idx_in_unique = idx_bad_quality[0].item()
                    y_rejected_str = unique_candidate_words[rejected_idx_in_unique]
            else:
                 print(f"Batch item {i}: No bad quality words found to select a rejected word.")

            # Print selected pair
            if y_chosen_str is not None and y_rejected_str is not None:
                print(f"Batch item {i}: Selected Pair -> Chosen='{y_chosen_str}' (Good Q), Rejected='{y_rejected_str}' (Bad Q)")
            else:
                fail_msg = []
                if y_chosen_str is None: fail_msg.append("No suitable chosen word found (good quality)")
                if y_rejected_str is None: fail_msg.append("No suitable rejected word found (bad quality)")
                print(f"Batch item {i}: Failed to select a distinct pair. Reason(s): {'; '.join(fail_msg)}")

            # --- Log Probability Calculation Phase ---
            if y_chosen_str is not None and y_rejected_str is not None:
                batch_chosen_words.append(y_chosen_str)
                batch_rejected_words.append(y_rejected_str)

                # Tokenize chosen/rejected words
                chosen_tokens = self.processing_class(y_chosen_str, add_special_tokens=False).input_ids
                rejected_tokens = self.processing_class(y_rejected_str, add_special_tokens=False).input_ids

                # Max length check
                if len(chosen_tokens) > self.max_target_length or len(rejected_tokens) > self.max_target_length:
                    logger.warning(f"Batch item {i}: Skipping pair - generated token length exceeds max_target_length.")
                    continue

                # Prepare for forward pass
                current_prompt_ids = prompt_input_ids[i:i+1, :prompt_lens[i].item()].to(self.accelerator.device)
                current_prompt_mask = prompt_attention_mask[i:i+1, :prompt_lens[i].item()].to(self.accelerator.device)

                # Convert token lists to tensors
                chosen_input_ids_tensor = torch.tensor(chosen_tokens, dtype=torch.long).unsqueeze(0).to(self.accelerator.device)
                rejected_input_ids_tensor = torch.tensor(rejected_tokens, dtype=torch.long).unsqueeze(0).to(self.accelerator.device)

                # Create attention masks
                chosen_attention_mask_tensor = torch.ones_like(chosen_input_ids_tensor)
                rejected_attention_mask_tensor = torch.ones_like(rejected_input_ids_tensor)

                # Create pair batch
                dpo_pair_batch = {
                    "prompt_input_ids": current_prompt_ids,
                    "prompt_attention_mask": current_prompt_mask,
                    "chosen_input_ids": chosen_input_ids_tensor,
                    "chosen_attention_mask": chosen_attention_mask_tensor,
                    "rejected_input_ids": rejected_input_ids_tensor,
                    "rejected_attention_mask": rejected_attention_mask_tensor,
                }

                try:
                    # Get logps using the policy model
                    policy_outputs = self.concatenated_forward(model, dpo_pair_batch)
                    policy_chosen_logps_i = policy_outputs["chosen_logps"]
                    policy_rejected_logps_i = policy_outputs["rejected_logps"]

                    # Get logps using the reference model
                    if self.ref_model:
                         self.ref_model.to(self.accelerator.device)
                         ref_outputs = self.concatenated_forward(self.ref_model, dpo_pair_batch)
                         ref_chosen_logps_i = ref_outputs["chosen_logps"]
                         ref_rejected_logps_i = ref_outputs["rejected_logps"]
                    elif self.is_peft_model:
                         with self.null_ref_context():
                              unwrapped_model = self.accelerator.unwrap_model(model)
                              ref_outputs = self.concatenated_forward(unwrapped_model, dpo_pair_batch)
                              ref_chosen_logps_i = ref_outputs["chosen_logps"]
                              ref_rejected_logps_i = ref_outputs["rejected_logps"]
                    elif self.precompute_ref_log_probs and "ref_chosen_logps" in inputs:
                        raise NotImplementedError("Handling precomputed ref logps per-pair inside compute_loss needs careful implementation.")
                    else:
                         logger.error("Reference model is None, not a PEFT model, and ref logps not precomputed.")
                         continue

                    # Store logps for batch loss calculation
                    all_policy_chosen_logps.append(policy_chosen_logps_i)
                    all_policy_rejected_logps.append(policy_rejected_logps_i)
                    all_ref_chosen_logps.append(ref_chosen_logps_i)
                    all_ref_rejected_logps.append(ref_rejected_logps_i)
                    valid_pairs_found_in_batch += 1
                except Exception as e:
                    logger.error(f"Error getting logps for batch item {i}: {e}", exc_info=True)
            else:
                # Log failure reasons
                failure_reason = []
                if y_chosen_str is None:
                    failure_reason.append(f"Need at least 1 good quality word, found {len(idx_good_quality)}")
                if y_rejected_str is None:
                     failure_reason.append(f"Need at least 1 bad quality word, found {len(idx_bad_quality)}")
                if not failure_reason:
                    failure_reason.append("Unknown reason for pair failure")

                # Log quality and diversity scores
                qual_strs = [f"{w}({q:.1f})" for w, q in zip(unique_candidate_words, qualities_list)]
                div_strs = [f"{w}({d:.3f})" for w, d in zip(unique_candidate_words, diversities.tolist())] if num_candidates > 1 else ["N/A"]*num_candidates

                print(
                    f"Batch item {i}: No valid DivPO pair. Reason(s): {'; '.join(failure_reason)}. "
                    f"Candidates ({num_candidates}): {unique_candidate_words}. "
                    f"Qualities: [{', '.join(qual_strs)}]. "
                    f"Diversities: [{', '.join(div_strs)}]. "
                    f"Selected Chosen Idx: {chosen_idx_in_unique}, Selected Rejected Idx: {rejected_idx_in_unique}."
                )

        # --- Loss Calculation ---
        loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
        logits = None

        if valid_pairs_found_in_batch > 0:
            # Concatenate logps from all valid pairs
            policy_chosen_logps = torch.cat(all_policy_chosen_logps)
            policy_rejected_logps = torch.cat(all_policy_rejected_logps)
            ref_chosen_logps = torch.cat(all_ref_chosen_logps)
            ref_rejected_logps = torch.cat(all_ref_rejected_logps)

            # Calculate DPO loss components
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = pi_logratios - ref_logratios

            # Calculate loss based on specified loss type
            if self.loss_type == "sigmoid":
                loss = -F.logsigmoid(self.beta * logits).mean()
            elif self.loss_type == "hinge":
                loss = torch.relu(1 - self.beta * logits).mean()
            else:
                raise ValueError(f"Unsupported loss_type: {self.loss_type}")

            print(f"Batch Loss Calculated: {loss.item():.4f} from {valid_pairs_found_in_batch} pairs.")

            # Store metrics for logging
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            margins = (chosen_rewards - rejected_rewards).mean().item()
            self.store_metrics({
                "loss": loss.item(),
                "rewards/chosen": chosen_rewards.mean().item(),
                "rewards/rejected": rejected_rewards.mean().item(),
                "rewards/accuracy": accuracy,
                "rewards/margins": margins,
                "logps/rejected": policy_rejected_logps.detach().mean().item(),
                "logps/chosen": policy_chosen_logps.detach().mean().item(),
                "logits": logits.detach().mean().item(),
                "valid_pairs_frac": valid_pairs_found_in_batch / batch_size
            })

            # Log some chosen/rejected words periodically
            if self.state.global_step % (self.args.logging_steps * 5) == 0:
                 log_n = min(3, len(batch_chosen_words))
                 if log_n > 0:
                      sample_pairs = list(zip(batch_chosen_words[:log_n], batch_rejected_words[:log_n]))
                      logger.info(f"Step {self.state.global_step} Sample Pairs (Chosen | Rejected): {sample_pairs}")
                      logger.info(f"Step {self.state.global_step} Reward Metrics: Acc={accuracy:.3f}, Margin={margins:.3f}")
        else:
            # No valid pairs in the batch
            logger.warning(f"Step {self.state.global_step}: No valid DivPO pairs found in the entire batch.")
            # Return zero loss but requires grad
            loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
            # Store zero metrics
            zero_metrics = {
                "loss": 0.0, "rewards/chosen": 0.0, "rewards/rejected": 0.0,
                "rewards/accuracy": 0.0, "rewards/margins": 0.0, "logps/rejected": 0.0,
                "logps/chosen": 0.0, "logits": 0.0, "valid_pairs_frac": 0.0
            }
            self.store_metrics(zero_metrics)

        # Prepare outputs dict for return
        outputs = {}
        if return_outputs:
            outputs["logits"] = logits

        if return_outputs:
            return loss, outputs
        return loss
    
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        """
        Override the _prepare_dataset method to handle prompt-only dataset.
        
        Args:
            dataset: The dataset to prepare
            processing_class: The tokenizer or processor
            args: Training arguments
            dataset_name: Name of the dataset (train/eval)
            
        Returns:
            Prepared dataset
        """
        if dataset is None:
            return None
            
        if not hasattr(dataset, "column_names") or len(dataset.column_names) == 0:
            raise ValueError("Dataset must have at least one column")
            
        # Check if we have a prompt-only dataset
        is_prompt_only = "prompt" in dataset.column_names and "chosen" not in dataset.column_names
        
        if is_prompt_only:
            logger.info(f"Detected prompt-only dataset for {dataset_name}. Will generate pairs on-the-fly.")
            
            # Apply chat template if available
            if hasattr(processing_class, "apply_chat_template") and callable(processing_class.apply_chat_template):
                # Extract prompts
                prompts = [example["prompt"] for example in dataset]
                
                # Apply chat template 
                def apply_template(prompt):
                    try:
                        return processing_class.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    except:
                        # Fallback if chat template fails
                        return prompt
                
                # Apply template to all prompts
                if isinstance(dataset, Dataset):
                    dataset = dataset.map(
                        lambda x: {"formatted_prompt": apply_template(x["prompt"])},
                        desc=f"Applying chat template to {dataset_name} dataset"
                    )
                else:  # IterableDataset
                    dataset = dataset.map(lambda x: {"formatted_prompt": apply_template(x["prompt"])})
                
                # Tokenize the formatted prompts
                def tokenize_prompt(example):
                    prompt = example["formatted_prompt"]
                    tokenized = processing_class(
                        prompt, 
                        truncation=True,
                        padding="max_length",
                        max_length=args.max_prompt_length,
                        return_tensors=None,
                    )
                    return {
                        "prompt_input_ids": tokenized["input_ids"],
                        "prompt_attention_mask": tokenized["attention_mask"],
                    }
                
                # Apply tokenization
                remove_cols = ["prompt"]
                if "formatted_prompt" in dataset.column_names:
                    remove_cols.append("formatted_prompt")
                    
                dataset = dataset.map(
                    tokenize_prompt,
                    remove_columns=remove_cols,
                    desc=f"Tokenizing {dataset_name} dataset"
                )
            else:
                # Simple tokenization without chat template
                def tokenize_row(example):
                    tokenized = processing_class(
                        example["prompt"],
                        truncation=True,
                        padding="max_length",
                        max_length=args.max_prompt_length,
                        return_tensors=None,
                    )
                    return {
                        "prompt_input_ids": tokenized["input_ids"],
                        "prompt_attention_mask": tokenized["attention_mask"],
                    }
                
                dataset = dataset.map(
                    tokenize_row,
                    remove_columns=["prompt"],
                    desc=f"Tokenizing {dataset_name} dataset"
                )
                
            return dataset
        else:
            # If it's a standard DPO dataset with prompt/chosen/rejected, use parent implementation
            return super()._prepare_dataset(dataset, processing_class, args, dataset_name) 