import argparse
import json
import math
import os
import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from .sft_helper import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def load_data_from_path(data_path: str, max_samples: Optional[int] = None) -> list:
    """
    Load data from JSON/JSONL file or directory.
    
    Args:
        data_path: Path to JSON file, JSONL file, or directory containing JSON/JSONL files.
        max_samples: Maximum number of samples to load (None for all).
    
    Returns:
        List of data examples.
    """
    data = []
    
    # Handle both file and directory paths
    if os.path.isdir(data_path):
        # Look for JSON or JSONL files
        json_files = sorted([f for f in os.listdir(data_path) if f.endswith(('.json', '.jsonl'))])
        file_paths = [os.path.join(data_path, f) for f in json_files]
    else:
        file_paths = [data_path]
    
    for file_path in file_paths:
        print(f"Loading data from {file_path}")
        
        try:
            with open(file_path, encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    if max_samples and len(data) >= max_samples:
                        break
                    try:
                        example = json.loads(line.strip())
                        data.append(example)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line in {file_path}: {e}")
        except UnicodeDecodeError:
            print(f"Warning: Skipping {file_path} - not a valid UTF-8 text file")
            continue
        
        if max_samples and len(data) >= max_samples:
            break

    print(f"Loaded {len(data)} samples")
    return data


class SFTDataset(Dataset):
    """Dataset for SFT training with prompt-output pairs."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to JSONL file, JSON file, or directory containing JSON/JSONL files.
            tokenizer: HuggingFace tokenizer.
            max_samples: Maximum number of samples to load (None for all).
        """
        self.data = load_data_from_path(data_path, max_samples=max_samples)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        # Handle both formats: (1) prompt/output fields and (2) messages/ground_truth fields
        if "prompt" in example and "output" in example:
            prompt = example.get("prompt", "")
            output = example.get("output", "")
        elif "messages" in example:
            # Extract prompt from system+user messages, output from assistant message
            messages = example.get("messages", [])
            prompt_parts = []
            output = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ["system", "user"]:
                    prompt_parts.append(content)
                elif role == "assistant":
                    output = content
            prompt = "\n".join(prompt_parts)
        else:
            # Fallback to empty if format is not recognized
            prompt = ""
            output = ""
            print(f"Warning: Unrecognized format for example at index {idx}: {example}")
        return {"prompt": prompt, "output": output}


def collate_fn(batch, tokenizer: PreTrainedTokenizerBase):
    """Collate batch of examples."""
    prompts = [example["prompt"] for example in batch]
    outputs = [example["output"] for example in batch]

    tokenized = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=outputs,
        tokenizer=tokenizer,
    )
    return tokenized


def evaluate_on_math(
    policy: PreTrainedModel,
    llm: LLM,
    eval_dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_eval_samples: Optional[int] = None,
    num_generations: int = 1,
    generation_config: Optional[dict] = None,
) -> dict:
    """
    Evaluate policy on MATH validation set.
    
    Args:
        policy: Policy model.
        llm: vLLM instance for inference.
        eval_dataset_path: Path to evaluation dataset (JSON, JSONL, or directory).
        tokenizer: HuggingFace tokenizer.
        max_eval_samples: Max samples to evaluate.
        num_generations: Number of generations per prompt.
        generation_config: Config for vLLM sampling.
    
    Returns:
        Dict with evaluation metrics.
    """
    if generation_config is None:
        generation_config = {
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
        }

    # Load evaluation data
    eval_data = load_data_from_path(eval_dataset_path, max_samples=max_eval_samples)

    # Load policy into vLLM
    load_policy_into_vllm_instance(policy, llm)

    # Generate and evaluate
    correct = 0
    total = len(eval_data)

    sampling_params = SamplingParams(**generation_config)

    for example in eval_data:
        prompt = example.get("prompt", "")
        target = example.get("output", "")

        # Generate
        outputs = llm.generate(
            [prompt],
            sampling_params=sampling_params,
        )
        generated_text = outputs[0].outputs[0].text

        # Simple exact match evaluation (can be improved)
        if generated_text.strip() == target.strip():
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
def train_sft(
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    train_data_path: str = "data/train.jsonl",
    eval_data_path: str = "data/eval.jsonl",
    output_dir: str = "output/sft",
    num_train_epochs: int = 3,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    learning_rate: float = 1e-4,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    eval_steps: int = 100,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    device: str = "cuda:0",
    eval_device: str = "cuda:1",
    use_wandb: bool = True,
    run_name: str = "sft-run",
):
    """
    Train SFT policy on MATH/Prime Intellect data.
    
    Args:
        model_id: HuggingFace model ID.
        train_data_path: Path to training JSONL.
        eval_data_path: Path to evaluation JSONL.
        output_dir: Base directory to save outputs.
        num_train_epochs: Number of training epochs.
        train_batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        learning_rate: Learning rate.
        max_train_samples: Max training samples (None for all).
        max_eval_samples: Max eval samples (None for all).
        eval_steps: Evaluate every N training steps.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_grad_norm: Gradient clipping value.
        seed: Random seed.
        device: Device for policy training.
        eval_device: Device for vLLM evaluation.
        use_wandb: Whether to use Weights & Biases.
        run_name: Name of this training run (used in output directory path).
    """
    # Setup - append run_name to output directory
    torch.manual_seed(seed)
    output_dir = f"{output_dir}/{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")

    if use_wandb:
        wandb.init(project="sft-math", name=run_name, config=locals())
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # Load model and tokenizer
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    # Enable gradient checkpointing to reduce memory usage
    policy.gradient_checkpointing_enable()
    policy.train()

    # Setup optimizer
    optimizer = optim.AdamW(policy.parameters(), lr=learning_rate)

    # Load datasets
    print(f"Loading training data from {train_data_path}...")
    train_dataset = SFTDataset(
        train_data_path,
        tokenizer,
        max_samples=max_train_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    # Initialize vLLM for evaluation
    print(f"Initializing vLLM on {eval_device}...")
    llm = init_vllm(model_id, device=eval_device, seed=seed)

    # Training loop
    global_step = 0
    best_eval_accuracy = 0.0

    for epoch in range(num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{num_train_epochs}")
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            # Get log probs
            log_probs_output = get_response_log_probs(
                policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            log_probs = log_probs_output["log_probs"]

            # Training step
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=1.0,
            )

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            # Log metrics
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train_step": global_step,
                })

            if batch_idx % 10 == 0:
                print(f"  Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            # Periodic evaluation
            if global_step % eval_steps == 0:
                print(f"\nEvaluating at step {global_step}...")
                policy.eval()
                
                eval_metrics = evaluate_on_math(
                    policy=policy,
                    llm=llm,
                    eval_dataset_path=eval_data_path,
                    tokenizer=tokenizer,
                    max_eval_samples=max_eval_samples,
                )

                if use_wandb:
                    wandb.log({
                        "eval/accuracy": eval_metrics["accuracy"],
                        "eval_step": global_step,
                    })

                print(f"Eval Accuracy: {eval_metrics['accuracy']:.4f}")

                # Save best model
                if eval_metrics["accuracy"] > best_eval_accuracy:
                    best_eval_accuracy = eval_metrics["accuracy"]
                    policy.save_pretrained(f"{output_dir}/best_model")
                    tokenizer.save_pretrained(f"{output_dir}/best_model")
                    print(f"Saved best model with accuracy {best_eval_accuracy:.4f}")

                policy.train()

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        policy.save_pretrained(f"{output_dir}/checkpoint-epoch-{epoch + 1}")
        tokenizer.save_pretrained(f"{output_dir}/checkpoint-epoch-{epoch + 1}")

    # Save final model
    policy.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    print(f"\nTraining complete. Best accuracy: {best_eval_accuracy:.4f}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/sft")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_device", type=str, default="cuda:1")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=f"sft-math-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    args = parser.parse_args()

    train_sft(
        model_id=args.model_id,
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        device=args.device,
        eval_device=args.eval_device,
        use_wandb=args.use_wandb,
        run_name=args.run_name,
    )