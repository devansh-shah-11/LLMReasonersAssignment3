#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) trainer for reasoning tasks.

Trains a policy on reasoning datasets (e.g., Countdown) using GRPO algorithm:
- Rollout batch of responses using the policy
- Score responses with a reward function
- Compute group-normalized advantages
- Apply policy gradient loss with clipping
- Update policy with gradient accumulation
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import wandb

from student.sft_helper import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step,
)
from student.drgrpo_grader import question_only_reward_fn


class ReasoningDataset(Dataset):
    """Dataset for reasoning tasks (Countdown, etc.)"""
    
    def __init__(self, jsonl_path: str, max_samples: int | None = None):
        """Load JSONL dataset.
        
        Expected format per line:
        {"prompt": "...", "ground_truth": "..." or "answer": "..."}
        """
        self.examples = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                example = json.loads(line)
                self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def init_vllm(model_path: str, num_gpus: int = 1) -> LLM:
    """Initialize vLLM instance for fast rollout generation."""
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
    )
    return llm


def generate_rollouts(
    llm: LLM,
    prompts: list[str],
    group_size: int,
    max_tokens: int = 512,
) -> list[str]:
    """Generate multiple rollouts per prompt using vLLM.
    
    Args:
        llm: vLLM instance
        prompts: list of prompt strings
        group_size: number of rollouts per prompt
        max_tokens: max tokens to generate
    
    Returns:
        list of all rollout responses (length = len(prompts) * group_size)
    """
    # Repeat each prompt group_size times
    repeated_prompts = [p for p in prompts for _ in range(group_size)]
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    
    outputs = llm.generate(repeated_prompts, sampling_params)
    rollout_responses = [output.outputs[0].text for output in outputs]
    
    return rollout_responses


def compute_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    ground_truths: list[str],
    group_size: int,
) -> tuple[list[float], dict]:
    """Score all rollouts using reward function.
    
    Args:
        reward_fn: Callable[[response, ground_truth], dict[str, float]]
        rollout_responses: all rollout responses
        ground_truths: ground truths, repeated group_size times
        group_size: rollouts per prompt
    
    Returns:
        tuple[list[float], dict]: (raw_rewards, metadata)
    """
    raw_rewards = []
    for response, ground_truth in zip(rollout_responses, ground_truths):
        result = reward_fn(response, ground_truth)
        reward_val = result.get("reward", 0.0)
        raw_rewards.append(reward_val)
    
    # Compute metadata
    rewards_tensor = torch.tensor(raw_rewards)
    metadata = {
        "mean_reward": rewards_tensor.mean().item(),
        "std_reward": rewards_tensor.std().item(),
        "max_reward": rewards_tensor.max().item(),
        "min_reward": rewards_tensor.min().item(),
    }
    
    return raw_rewards, metadata


def train_grpo(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    dataset_path: str = "tests/fixtures/sft_sample.jsonl",
    output_dir: str = "/tmp/grpo_output",
    run_name: str = None,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    rollout_batch_size: int = 32,
    group_size: int = 4,
    num_train_steps: int = 100,
    validation_interval: int = 10,
    gradient_accumulation_steps: int = 1,
    cliprange: float = 0.5,
    advantage_eps: float = 1e-4,
    normalize_by_std: bool = True,
    max_samples: int = None,
    seed: int = 42,
):
    """Train GRPO policy on reasoning dataset.
    
    Args:
        model_name: HuggingFace model name
        dataset_path: path to training JSONL
        output_dir: where to save models
        run_name: name for this run (auto-generated if None)
        learning_rate: learning rate for optimizer
        batch_size: batch size for policy updates
        rollout_batch_size: number of prompts per rollout batch
        group_size: rollouts per prompt
        num_train_steps: total training steps
        validation_interval: steps between validation
        gradient_accumulation_steps: gradient accumulation
        cliprange: clip range for GRPO
        advantage_eps: epsilon for reward normalization
        normalize_by_std: whether to normalize by std
        max_samples: max training examples (None = all)
        seed: random seed
    """
    
    # Setup
    torch.manual_seed(seed)
    if run_name is None:
        run_name = f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = Path(output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="llm-reasoners-grpo",
        name=run_name,
        config={
            "model_name": model_name,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "rollout_batch_size": rollout_batch_size,
            "group_size": group_size,
            "num_train_steps": num_train_steps,
            "cliprange": cliprange,
        }
    )
    
    # Load model and tokenizer
    print(f"Loading model {model_name}...")
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize vLLM for rollout generation (separate GPU)
    print("Initializing vLLM for rollout generation...")
    try:
        rollout_llm = init_vllm(model_name, num_gpus=1)
    except Exception as e:
        print(f"Warning: Could not initialize vLLM: {e}. Falling back to single-GPU setup.")
        rollout_llm = None
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = ReasoningDataset(dataset_path, max_samples=max_samples)
    print(f"Loaded {len(dataset)} examples")
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=rollout_batch_size,
        shuffle=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    global_step = 0
    validation_rewards_history = []
    
    for epoch in range((num_train_steps + len(train_loader) - 1) // len(train_loader)):
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= num_train_steps:
                break
            
            # Extract prompts and ground truths
            prompts = [ex.get("prompt") for ex in batch]
            ground_truths = [
                ex.get("ground_truth") or ex.get("answer") 
                for ex in batch
            ]
            
            # Generate rollouts
            print(f"Step {global_step}: Generating {len(prompts) * group_size} rollouts...")
            if rollout_llm is not None:
                rollout_responses = generate_rollouts(
                    rollout_llm,
                    prompts,
                    group_size,
                )
            else:
                # Fallback: use policy model for rollouts (slower)
                rollout_responses = []
                model.eval()
                with torch.no_grad():
                    for prompt in prompts:
                        for _ in range(group_size):
                            inputs = tokenizer([prompt], return_tensors="pt").to(device)
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=512,
                                temperature=0.8,
                                do_sample=True,
                            )
                            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            response = response[len(prompt):].strip()
                            rollout_responses.append(response)
                model.train()
            
            # Repeat ground truths for each rollout
            repeated_gts = [gt for gt in ground_truths for _ in range(group_size)]
            
            # Compute rewards
            print(f"Step {global_step}: Computing rewards...")
            raw_rewards_list, reward_metadata = compute_rewards(
                question_only_reward_fn,
                rollout_responses,
                repeated_gts,
                group_size,
            )
            
            # Compute group-normalized advantages
            raw_rewards_tensor = torch.tensor(raw_rewards_list, dtype=torch.float32)
            normalized_rewards, _, norm_metadata = compute_group_normalized_rewards(
                reward_fn=question_only_reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_gts,
                group_size=group_size,
                advantage_eps=advantage_eps,
                normalize_by_std=normalize_by_std,
            )
            
            # Prepare advantages for training
            # Shape: (batch_size * group_size, 1)
            advantages = normalized_rewards.unsqueeze(1)
            raw_rewards = raw_rewards_tensor.unsqueeze(1)
            
            # Tokenize prompts and responses for training
            full_responses = []
            for prompt, response in zip(prompts, rollout_responses):
                full_responses.append(response)
            
            # Tokenize each (prompt, response) pair
            policy_inputs = []
            old_log_probs_list = []
            response_masks_list = []
            
            model.eval()
            with torch.no_grad():
                for i, (prompt, response) in enumerate(zip(prompts * group_size, rollout_responses)):
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    response_ids = tokenizer.encode(response, add_special_tokens=False)
                    full_ids = prompt_ids + response_ids
                    
                    # Tokenize
                    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long).to(device)
                    labels = torch.tensor([full_ids[1:]], dtype=torch.long).to(device)
                    
                    # Get old log probs
                    with torch.no_grad():
                        old_logits = model(input_ids).logits
                        old_log_probs = F.log_softmax(old_logits, dim=-1)
                        old_log_probs = old_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                    
                    old_log_probs_list.append(old_log_probs.squeeze(0))
                    
                    # Create response mask
                    prompt_len = len(prompt_ids)
                    response_mask = torch.zeros(len(full_ids) - 1, dtype=torch.bool)
                    response_mask[prompt_len:] = True
                    response_masks_list.append(response_mask)
                    
                    policy_inputs.append(input_ids.squeeze(0))
            
            # Pad and stack tensors
            max_len = max(len(inp) for inp in policy_inputs)
            padded_inputs = torch.zeros(len(policy_inputs), max_len, dtype=torch.long).to(device)
            padded_masks = torch.zeros(len(policy_inputs), max_len, dtype=torch.bool).to(device)
            padded_old_log_probs = torch.zeros(len(policy_inputs), max_len, dtype=torch.float32).to(device)
            
            for i, (inp, mask, old_lp) in enumerate(zip(policy_inputs, response_masks_list, old_log_probs_list)):
                seq_len = len(inp)
                padded_inputs[i, :seq_len] = inp
                padded_masks[i, :seq_len] = mask
                padded_old_log_probs[i, :seq_len] = old_lp
            
            # Forward pass
            model.train()
            optimizer.zero_grad()
            
            batch_loss = 0.0
            num_microbatches = (len(padded_inputs) + batch_size - 1) // batch_size
            
            for microbatch_idx in range(num_microbatches):
                start_idx = microbatch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(padded_inputs))
                
                microbatch_inputs = padded_inputs[start_idx:end_idx]
                microbatch_masks = padded_masks[start_idx:end_idx]
                microbatch_old_log_probs = padded_old_log_probs[start_idx:end_idx]
                microbatch_advantages = advantages[start_idx:end_idx]
                
                # Get policy log probs
                policy_output = model(microbatch_inputs)
                policy_logits = policy_output.logits
                policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                
                # Get log probs of labels
                labels = microbatch_inputs[:, 1:]  # Shift for causal LM
                labels_padded = torch.full_like(microbatch_inputs, -100)
                labels_padded[:, :-1] = microbatch_inputs[:, 1:]
                policy_log_probs_per_token = policy_log_probs.gather(dim=-1, index=labels_padded.unsqueeze(-1)).squeeze(-1)
                
                # Compute loss with gradient accumulation
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs_per_token,
                    response_mask=microbatch_masks,
                    gradient_accumulation_steps=gradient_accumulation_steps * num_microbatches,
                    loss_type="grpo_clip",
                    advantages=microbatch_advantages,
                    old_log_probs=microbatch_old_log_probs,
                    cliprange=cliprange,
                )
                
                batch_loss += loss.item()
            
            # Update
            if (microbatch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Logging
            wandb.log({
                "train/loss": batch_loss / num_microbatches,
                "train/mean_reward": reward_metadata["mean_reward"],
                "train/std_reward": reward_metadata["std_reward"],
                "step": global_step,
            })
            
            # Validation
            if global_step % validation_interval == 0:
                print(f"\nStep {global_step}: Validation")
                model.eval()
                
                # Sample a few examples for validation
                val_prompts = prompts[:min(2, len(prompts))]
                
                with torch.no_grad():
                    val_rollouts = []
                    val_rewards = []
                    
                    for prompt in val_prompts:
                        inputs = tokenizer([prompt], return_tensors="pt").to(device)
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.0,  # Greedy
                            do_sample=False,
                        )
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = response[len(prompt):].strip()
                        val_rollouts.append(response)
                        
                        # Get reward
                        gt = ground_truths[val_prompts.index(prompt)]
                        reward_result = question_only_reward_fn(response, gt)
                        reward = reward_result.get("reward", 0.0)
                        val_rewards.append(reward)
                    
                    val_mean_reward = sum(val_rewards) / len(val_rewards) if val_rewards else 0.0
                    validation_rewards_history.append({
                        "step": global_step,
                        "reward": val_mean_reward,
                    })
                    
                    wandb.log({
                        "val/mean_reward": val_mean_reward,
                        "step": global_step,
                    })
                    
                    print(f"Validation reward: {val_mean_reward:.4f}")
                    for i, (prompt, rollout, reward) in enumerate(zip(val_prompts, val_rollouts, val_rewards)):
                        print(f"\nExample {i+1}:")
                        print(f"Prompt: {prompt[:100]}...")
                        print(f"Response: {rollout[:200]}...")
                        print(f"Reward: {reward:.4f}")
                
                model.train()
                
                # Checkpoint
                checkpoint_dir = output_dir / f"checkpoint-step-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir / "model")
                tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
            
            global_step += 1
        
        if global_step >= num_train_steps:
            break
    
    # Save final model
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_model_dir / "model")
    tokenizer.save_pretrained(final_model_dir / "tokenizer")
    
    # Save validation history
    with open(output_dir / "validation_rewards.json", "w") as f:
        json.dump(validation_rewards_history, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {final_model_dir}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO trainer for reasoning tasks")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset_path", type=str, default="tests/fixtures/sft_sample.jsonl")
    parser.add_argument("--output_dir", type=str, default="/tmp/grpo_output")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--num_train_steps", type=int, default=100)
    parser.add_argument("--validation_interval", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--cliprange", type=float, default=0.5)
    parser.add_argument("--advantage_eps", type=float, default=1e-4)
    parser.add_argument("--normalize_by_std", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_grpo(**vars(args))
