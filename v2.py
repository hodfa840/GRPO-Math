import sys
# Block vLLM if it causes issues with Unsloth (keep this if needed)
# sys.modules["vllm"] = None # Keep commented unless you know you need it

# === Make sure Unsloth is first ===
try:
    import unsloth
except ImportError:
    print("Unsloth not found. Please install it: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
    sys.exit(1)

from unsloth import FastLanguageModel, PatchFastRL

# Fix compatibility between Unsloth and GRPOTrainer (Important!)
PatchFastRL("GRPO", FastLanguageModel)

import os
import json
import random
import torch
import datetime
import re
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import TrainingArguments, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
# Use standard tqdm if not in notebook
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
import numpy as np
import evaluate as hf_evaluate # For potential future metrics like BLEU, ROUGE if needed
import wandb # Import wandb explicitly

# ========== CONFIGURATION ==========
# --- Model & Paths ---
MODEL_NAME = "unsloth/zephyr-sft-bnb-4bit" # Base model
# MODEL_NAME = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit" # Alternative
SAVE_DIR = "results/grpo_zephyr_gsm8k"       # Directory to save the final model
LOG_DIR = "results/logs"                   # Directory for log files
LOG_FILE = os.path.join(LOG_DIR, f"grpo_zephyr_gsm8k_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
WANDB_PROJECT_NAME = "grpo_gsm8k_tuning"   # Project name for Weights & Biases

# --- Data & Debugging ---
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
DEBUG = True                      # Use smaller dataset subset for quick testing
DEBUG_TRAIN_SIZE = 100            # Number of training samples in debug mode
DEBUG_VAL_SIZE = 20               # Number of validation samples in debug mode
DEBUG_TEST_SIZE = 20              # Number of test samples in debug mode
VAL_SET_SIZE = 0.10               # Proportion of training data to use for validation (if not in debug)

# --- Training Hyperparameters ---
SEED = 42
LEARNING_RATE = 5e-6              # Learning rate for AdamW
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
WEIGHT_DECAY = 0.01               # Weight decay
WARMUP_RATIO = 0.1                # Portion of training steps for warmup
LR_SCHEDULER_TYPE = "cosine"
OPTIMIZER = "paged_adamw_8bit"    # Efficient optimizer for 4-bit models
LOGGING_STEPS = 10                # Log metrics every N steps (Good for debug)
PER_DEVICE_TRAIN_BATCH_SIZE = 1   # Keep low for large models/sequences
GRADIENT_ACCUMULATION_STEPS = 16  # Increase for larger effective batch size (Eff Batch = N_GPU * BS * Accum)
NUM_GENERATIONS = 4               # Number of completions per prompt in GRPO
MAX_PROMPT_LENGTH = 256           # Max tokens for the prompt portion
MAX_COMPLETION_LENGTH = 512       # Max tokens for the generated completion
MAX_SEQ_LENGTH = MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH # Should match model loading
MAX_STEPS = 500 if not DEBUG else 100 # Total training steps (increase significantly for real runs)
EVAL_STEPS = 50 if not DEBUG else 20  # Evaluate roughly during training (Note: GRPO doesn't use this directly like Trainer)
SAVE_STEPS = 100 if not DEBUG else 50 # Save checkpoints every N steps
MAX_GRAD_NORM = 1.0               # Gradient clipping value (adjusted from 0.1)

# --- Reward Weights ---
# Higher value means stronger incentive
REWARD_CORRECTNESS = 10.0
REWARD_STRICT_FORMAT = 2.0
REWARD_INT_ANSWER = 1.0
REWARD_SOFT_FORMAT = 0.5 # Lower weight for just having the tags somewhere

# --- Generation Parameters (for Evaluation) ---
EVAL_MAX_NEW_TOKENS = MAX_COMPLETION_LENGTH # Match training completion length
EVAL_TEMPERATURE = 0.7
EVAL_TOP_P = 0.95

# --- Output Verbosity ---
PRINT_REWARD_FREQ = 0.2 # Print reward details for ~20% of calls

# ========== Seed Everything ==========
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ========== Define system prompt and XML format ==========
SYSTEM_PROMPT = """\
You are a helpful assistant that solves math problems.
Think step-by-step to solve the problem.
Respond in the following strict format, placing your reasoning within <reasoning> tags and your final numerical answer within <answer> tags. Do not include anything outside these tags or extra explanations after the final tag.

<reasoning>
Step 1: Identify the key information and the question.
Step 2: Break down the problem into smaller steps.
Step 3: Perform calculations for each step.
Step 4: Arrive at the final answer.
</reasoning>
<answer>
FINAL_NUMERICAL_ANSWER
</answer>"""

# (We don't use XML_COT_FORMAT for input, only define the expected output structure via SYSTEM_PROMPT)

# ========== Helper functions for extraction ==========
def extract_xml_tags(text: str, tag: str) -> str:
    """Extracts content between the first occurrence of <tag> and </tag>."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def extract_xml_answer(text: str) -> str:
    """Extracts the content within the <answer> tags."""
    return extract_xml_tags(text, "answer")

def extract_hash_answer(text: str) -> str:
    """Extracts the final answer from the original GSM8K format."""
    if "####" not in text:
        return ""
    # Extract the numerical part after ####, removing potential commas
    answer = text.split("####")[-1].strip()
    answer = re.sub(r'[^\d.-]', '', answer) # Keep digits, dots, and minus signs
    return answer

# ========== LOAD & PREP DATA ==========
def format_gsm8k_prompt(example):
    # Renaming columns to 'prompt' and 'reference' to align with TRL defaults
    # when data_keys is not specified, based on previous error debugging.
    return {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': example['question']}
        ],
        'reference': extract_hash_answer(example['answer']) # This name MUST match the reward func kwarg
    }


def load_and_prepare_data(debug=DEBUG):
    print("Loading and preparing dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # Split train into train and validation
    train_val_split = dataset['train'].train_test_split(test_size=VAL_SET_SIZE, seed=SEED)
    train_data = train_val_split['train']
    val_data = train_val_split['test']
    test_data = dataset['test']

    # Apply formatting
    train_data = train_data.map(format_gsm8k_prompt, remove_columns=train_data.column_names)
    val_data = val_data.map(format_gsm8k_prompt, remove_columns=val_data.column_names)
    test_data = test_data.map(format_gsm8k_prompt, remove_columns=test_data.column_names)

    if debug:
        print("DEBUG MODE: reducing sizes...")
        train_data = train_data.select(range(min(DEBUG_TRAIN_SIZE, len(train_data))))
        val_data = val_data.select(range(min(DEBUG_VAL_SIZE, len(val_data))))
        test_data = test_data.select(range(min(DEBUG_TEST_SIZE, len(test_data))))

    print(f"Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

# ========== REWARD FUNCTIONS ==========

def correctness_reward_func(prompts, completions, reference, **kwargs) -> list[float]:
    """Rewards correctness if the extracted answer matches the ground truth."""
    rewards = []
    # Assumes prompts is a list containing one chat list for the current item
    question = prompts[0][-1]['content'] if prompts and prompts[0] else "Unknown Question"

    # Get the ground truth answer for this specific item (passed as a list)
    ground_truth_answer = reference[0] if reference else "Unknown Answer"

    for gen_idx, gen_list in enumerate(completions): # completions is List[List[Dict]]
        # Assumes one message per generation attempt in the inner list
        response_text = gen_list[0]['content'] if gen_list else ""
        extracted_ans = extract_xml_answer(response_text)

        is_correct = (extracted_ans == ground_truth_answer) and (extracted_ans != "")
        current_reward = REWARD_CORRECTNESS if is_correct else 0.0
        rewards.append(current_reward)

        # --- Added Print Statement ---
        # Print details occasionally for inspection
        # Only print for the first generation attempt (gen_idx == 0) for brevity
        if gen_idx == 0 and random.random() < PRINT_REWARD_FREQ:
            print("\n" + "="*20 + " Reward Calculation Sample " + "="*20)
            print(f"Question:\n{question}")
            print(f"Ground Truth Answer: {ground_truth_answer}")
            print(f"Model Response:\n{response_text}")
            print(f"Extracted Answer: {extracted_ans}")
            print(f"Correct? {is_correct} -> Correctness Reward: {current_reward}")
            print("="*60 + "\n")
        # --- End Added Print Statement ---

    return rewards


def int_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Rewards if the extracted answer is purely an integer (or float)."""
    rewards = []
    for gen_list in completions:
        response_text = gen_list[0]['content'] if gen_list else ""
        extracted_ans = extract_xml_answer(response_text)
        is_numeric = False
        if extracted_ans:
            try:
                float(extracted_ans)
                is_numeric = True
            except ValueError:
                is_numeric = False
        rewards.append(REWARD_INT_ANSWER if is_numeric else 0.0)
    return rewards

def strict_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Rewards if the completion strictly matches the <reasoning>...</reasoning>\n<answer>...</answer> format."""
    pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$" # Made reasoning non-greedy
    rewards = []
    for gen_list in completions:
        response_text = gen_list[0]['content'].strip() if gen_list else ""
        match = re.match(pattern, response_text, re.DOTALL | re.IGNORECASE)
        rewards.append(REWARD_STRICT_FORMAT if match else 0.0)
    return rewards

def soft_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Rewards if <reasoning> and <answer> tags are present anywhere."""
    pattern_reasoning = r"<reasoning>.*?</reasoning>"
    pattern_answer = r"<answer>.*?</answer>"
    rewards = []
    for gen_list in completions:
        response_text = gen_list[0]['content'] if gen_list else ""
        has_reasoning = re.search(pattern_reasoning, response_text, re.DOTALL | re.IGNORECASE) is not None
        has_answer = re.search(pattern_answer, response_text, re.DOTALL | re.IGNORECASE) is not None
        rewards.append(REWARD_SOFT_FORMAT if (has_reasoning and has_answer) else 0.0)
    return rewards


# ========== SAVE LOG ==========
def save_log(log_data, file_path=LOG_FILE):
    """Saves the log data dictionary to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(log_data, f, indent=4, default=str)
        print(f"Log saved to {file_path}")
    except Exception as e:
        print(f"Error saving log file: {e}")


# ========== EVALUATION FUNCTION ==========
@torch.no_grad()
def evaluate_model(model, tokenizer, dataset, device, generation_config):
    """Evaluates the model on a given dataset."""
    model.eval()
    correct_answers = 0
    correct_format = 0
    total = len(dataset)
    all_results = []

    print(f"Starting evaluation on {total} samples...")
    # Use standard tqdm if available
    eval_iterator = tqdm(dataset, desc="Evaluating", total=total)

    for example in eval_iterator:
        prompt_data = example['prompt']           # Use corrected key 'prompt'
        ground_truth_answer = example['reference']  # Use corrected key 'reference'

        input_text = tokenizer.apply_chat_template(prompt_data, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH).to(device)

        outputs = model.generate(**inputs, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        extracted_ans = extract_xml_answer(response_text)
        is_correct = (extracted_ans == ground_truth_answer) and (extracted_ans != "")

        # Updated pattern for stricter format check during eval
        pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
        is_strict_format = re.match(pattern, response_text, re.DOTALL | re.IGNORECASE) is not None

        if is_correct: correct_answers += 1
        if is_strict_format: correct_format += 1

        all_results.append({
            "question": prompt_data[-1]['content'], # Assumes user query is last
            "ground_truth": ground_truth_answer,
            "generated_text": response_text,
            "extracted_answer": extracted_ans,
            "is_correct": is_correct,
            "is_strict_format": is_strict_format,
        })

    accuracy = (correct_answers / total) * 100 if total > 0 else 0
    format_accuracy = (correct_format / total) * 100 if total > 0 else 0

    print(f"Evaluation Complete: Accuracy = {accuracy:.2f}%, Format Accuracy = {format_accuracy:.2f}%")
    return {"accuracy": accuracy, "format_accuracy": format_accuracy, "results": all_results}


# ========== MAIN TRAINING FUNCTION ==========
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_dataset, val_dataset, test_dataset = load_and_prepare_data(debug=DEBUG)

    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        # token = "hf_...", # Add your token if model is private
        # device_map = "auto", # Unsloth handles this well
        fast_inference=False, # Set to False as we blocked vLLM
    )
    print("Base model loaded.")

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth", # Use Unsloth's specific GC
        random_state=SEED,
    )
    print("LoRA adapters added.")
    model.print_trainable_parameters()

    # Define training arguments using GRPOConfig
    training_args = GRPOConfig(
        output_dir=os.path.join(SAVE_DIR, "checkpoints"),
        num_train_epochs=-1,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        max_grad_norm=MAX_GRAD_NORM,
        remove_unused_columns=False, # Important: Keep columns for reward functions
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        seed=SEED,
        # --- Enable W&B always ---
        report_to="wandb",
        # report_to="wandb" if not DEBUG else "none", # Original line
        # --- End W&B change ---
        run_name=f"grpo-zephyr-gsm8k-debug_{DEBUG}-{datetime.datetime.now().strftime('%Y%m%d_%H%M')}", # Add debug status to name
    )

    # --- Initialize W&B always if report_to is set ---
    if training_args.report_to == "wandb":
        try:
            # Ensure wandb is imported
            import wandb
            wandb.init(project=WANDB_PROJECT_NAME, config=training_args.to_dict())
            print("Weights & Biases initialized.")
        except ImportError:
            print("Wandb not installed or import failed. Skipping W&B logging. `pip install wandb`")
            training_args.report_to = "none"
    # --- End W&B init change ---

    # Initialize GRPOTrainer - Using the setup derived from debugging
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer, # Use processing_class
        train_dataset=train_dataset,
        reward_funcs=[            # Use reward_funcs
            correctness_reward_func, # Expects 'reference' kwarg now
            strict_format_reward_func,
            int_reward_func,
            soft_format_reward_func,
        ],
        # REMOVED data_keys - Relying on column name matching for reward funcs
    )
    print("Trainer initialized.")

    print("Starting training...")
    # Wrap training in try/except to potentially catch errors earlier
    # And ensure wandb finishes
    try:
        train_result = trainer.train()
        print("Training finished.")

        metrics = train_result.metrics
        # Ensure log history exists before accessing
        log_history = trainer.state.log_history if hasattr(trainer.state, 'log_history') else []
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        print(f"Saving final model to {SAVE_DIR}...")
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print("Model and tokenizer saved.")

        # === Evaluation ===
        print("\n=== Starting Post-Training Evaluation ===")
        # Use Unsloth's GenerationConfig if needed, otherwise standard Transformers works
        # Note: Unsloth's might be needed if using specific optimized generation features
        eval_generation_config = transformers.GenerationConfig( # Using standard transformers GC
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            temperature=EVAL_TEMPERATURE,
            top_p=EVAL_TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        device = model.device

        print("\n--- Evaluating on Validation Set ---")
        val_metrics = evaluate_model(
            model=trainer.model,
            tokenizer=tokenizer,
            dataset=val_dataset,
            device=device,
            generation_config=eval_generation_config
        )
        print(f"Validation Metrics: {val_metrics['accuracy']:.2f}% Accuracy, {val_metrics['format_accuracy']:.2f}% Format Accuracy")

        print("\n--- Evaluating on Test Set ---")
        test_metrics = evaluate_model(
            model=trainer.model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            device=device,
            generation_config=eval_generation_config
        )
        print(f"Test Metrics: {test_metrics['accuracy']:.2f}% Accuracy, {test_metrics['format_accuracy']:.2f}% Format Accuracy")

        # === Save Logs ===
        print("\nSaving training and evaluation logs...")
        final_log = {
            "run_timestamp": datetime.datetime.now().isoformat(),
            "status": "COMPLETED",
            "config": {
                "model_name": MODEL_NAME, "save_dir": SAVE_DIR, "dataset_name": DATASET_NAME,
                "debug_mode": DEBUG, "seed": SEED,
                "reward_weights": {
                    "correctness": REWARD_CORRECTNESS, "strict_format": REWARD_STRICT_FORMAT,
                    "int_answer": REWARD_INT_ANSWER, "soft_format": REWARD_SOFT_FORMAT,
                },
                "training_args": training_args.to_dict()
            },
            "training_metrics": metrics,
            "training_log_history": log_history,
            "evaluation": {
                "validation": {k: v for k, v in val_metrics.items() if k != 'results'},
                "test": {k: v for k, v in test_metrics.items() if k != 'results'}
            }
        }
        save_log(final_log, LOG_FILE)

        val_results_file = os.path.join(LOG_DIR, f"grpo_eval_val_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        test_results_file = os.path.join(LOG_DIR, f"grpo_eval_test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_log({"results": val_metrics["results"]}, val_results_file)
        save_log({"results": test_metrics["results"]}, test_results_file)

        # === Test Inference Example ===
        print("\n=== Test Inference Example ===")
        test_question = "Farmer John has 3 cows. His neighbor has twice as many cows. How many cows do they have in total?"
        # Use the 'prompt' format for consistency
        prompt_data = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": test_question}]
        input_text = tokenizer.apply_chat_template(prompt_data, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH).to(device)

        outputs = trainer.model.generate(**inputs, generation_config=eval_generation_config, pad_token_id=tokenizer.eos_token_id)
        generated_part = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        print(f"\n--- Test Question ---")
        print(test_question)
        print(f"\n--- Model Output ---")
        print(generated_part)
        print(f"\nExtracted Answer: {extract_xml_answer(generated_part)}")

    except Exception as e:
        print("\n--- TRAINING FAILED ---")
        import traceback
        traceback.print_exc()
        # Optionally save basic log even on failure
        # Ensure log_history is defined even if training failed early
        log_history_on_fail = trainer.state.log_history if 'trainer' in locals() and hasattr(trainer.state, 'log_history') else []
        error_log = {
            "run_timestamp": datetime.datetime.now().isoformat(),
            "status": "FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
             "training_log_history": log_history_on_fail, # Log history if available
            "config": { # Save config for debugging
                "model_name": MODEL_NAME, "save_dir": SAVE_DIR, "dataset_name": DATASET_NAME,
                "debug_mode": DEBUG, "seed": SEED,
                 "training_args": training_args.to_dict() if 'training_args' in locals() else "Not Initialized"
            }
        }
        error_log_file = os.path.join(LOG_DIR, f"grpo_FAILED_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_log(error_log, error_log_file)

    finally:
        # Ensure wandb run finishes even if training fails or was skipped
        current_wandb_run = wandb.run if 'wandb' in sys.modules and hasattr(sys.modules['wandb'], 'run') else None
        if training_args.report_to == "wandb" and current_wandb_run is not None:
            print("Finishing W&B run...")
            wandb.finish()
            print("W&B run finished.")
        else:
            print("W&B logging was not enabled or run did not initialize.")


if __name__ == "__main__":


    main()