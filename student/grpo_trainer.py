"""
GRPO Train Loop for Countdown dataset.

Usage:
    python grpo_train.py --data_path /scratch/dns5508/dataset/countdown \
                         --prompt_file student/prompts/countdown.prompt

Dataset formats supported (tried in order):
  1. HuggingFace Arrow:  <data_path>/dataset/{train,dev,test}/
  2. Parquet:            <data_path>/{split}.parquet
  3. 10k parquet:        <data_path>/train_10k.parquet  (train only)
"""

import argparse
import os
import re
import random
from typing import Literal
from unittest.mock import patch

import torch
import torch.nn as nn
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from student.sft_helper import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step,
    tokenize_prompt_and_output,
)

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_countdown_prompt(prompt_file: str) -> str:
    """Load the countdown prompt template from disk."""
    with open(prompt_file, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Dataset loading — handles both Arrow (HF) and Parquet formats
# ---------------------------------------------------------------------------

def load_countdown_dataset(data_path: str, split: str) -> list[dict]:
    """
    Load Countdown dataset.  Tries in order:
      1. HuggingFace Arrow dataset at <data_path>/dataset/<split>/
      2. Parquet at <data_path>/<split>.parquet
      3. Parquet at <data_path>/train_10k.parquet  (train split only)

    Returns list of dicts with keys: "target" (int), "numbers" (list[int]).
    """
    # 1. HuggingFace Arrow format
    arrow_path = os.path.join(data_path, "dataset", split)
    if os.path.isdir(arrow_path):
        try:
            from datasets import load_from_disk
            ds = load_from_disk(arrow_path)
            print(f"[data] Arrow dataset from {arrow_path}, cols={ds.column_names}")
            return _normalise_hf_dataset(ds)
        except Exception as e:
            print(f"[data] Arrow load failed ({e}), trying parquet...")

    # 2. Parquet
    parquet_path = os.path.join(data_path, f"{split}.parquet")
    if not os.path.exists(parquet_path) and split == "train":
        parquet_path = os.path.join(data_path, "train_10k.parquet")

    if os.path.exists(parquet_path):
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            print(f"[data] Parquet {parquet_path}, cols={list(df.columns)}")
            return _normalise_df(df)
        except Exception as e:
            raise RuntimeError(f"[data] Could not load {parquet_path}: {e}")

    raise FileNotFoundError(
        f"No Countdown {split} split found at {data_path}. "
        "Expected dataset/<split>/ (Arrow) or <split>.parquet"
    )


def _normalise_hf_dataset(ds) -> list[dict]:
    return [{"target": int(ds["target"][i]), "numbers": list(ds["nums"][i])}
            for i in range(len(ds))]


def _normalise_df(df) -> list[dict]:
    return [{"target": int(row["target"]), "numbers": list(row["nums"])}
            for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(example: dict, prompt_template: str) -> str:
    """
    Fill the countdown prompt template.
    Supports {numbers}/{target} placeholders, or appends the problem to a
    static preamble if no placeholders are found.
    """
    numbers_str = str(example["numbers"])
    target_str = str(example["target"])

    if "{numbers}" in prompt_template and "{target}" in prompt_template:
        return prompt_template.format(numbers=numbers_str, target=target_str)

    problem = (
        f"\nUsing the numbers in the list {numbers_str}, "
        f"create an equation that equals {target_str}.\n"
        "You can use basic arithmetic operations (+, -, *, /) "
        "and each number can only be used once.\n"
    )
    return prompt_template.rstrip() + problem


def build_ground_truth(example: dict) -> str:
    """'target|n1,n2,...' string consumed by the reward function."""
    nums = ",".join(str(n) for n in example["numbers"])
    return f"{example['target']}|{nums}"


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def countdown_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    """
    Parse <answer>…</answer>, evaluate arithmetic, compare to target.
    ground_truth = "target|n1,n2,..."
    """
    format_reward = answer_reward = 0.0

    try:
        target_str, _ = ground_truth.split("|", 1)
        target = int(target_str.strip())
    except Exception:
        return {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}

    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        format_reward = 1.0
        answer_text = m.group(1).strip()
        try:
            eq_matches = re.findall(r"=\s*([\-\d\.]+)\s*$", answer_text, re.MULTILINE)
            if eq_matches:
                result = float(eq_matches[-1])
            else:
                expr = re.sub(r"Step\s*\d+\s*:", "", answer_text)
                lines = [l.strip() for l in expr.splitlines() if l.strip()]
                last = lines[-1] if lines else expr
                if "=" in last:
                    last = last.split("=")[0].strip()
                result = float(eval(last, {"__builtins__": {}}))
            if abs(result - target) < 1e-6:
                answer_reward = 1.0
        except Exception:
            answer_reward = 0.0

    reward = 0.1 * format_reward + 0.9 * answer_reward
    return {"reward": reward, "format_reward": format_reward, "answer_reward": answer_reward}


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------

def init_vllm(model_id: str, device: str, seed: int,
              gpu_memory_utilization: float = 0.85):
    """
    Initialise a vLLM LLM on one specific GPU.
    Monkeypatches from TRL bypass world-size / profiling checks that break
    single-GPU setups.
    """
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,  # single GPU
            disable_log_stats=True,  # quieter on HPC
        )
    return llm


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm) -> None:
    """Sync policy weights into the vLLM executor in-place (no extra VRAM)."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ---------------------------------------------------------------------------
# Rollout + evaluation
# ---------------------------------------------------------------------------

def generate_rollouts(
    llm,
    prompts: list[str],
    group_size: int,
    temperature: float,
    max_tokens: int,
    min_tokens: int,
) -> list[str]:
    """Generate group_size completions per prompt; returns flat list."""
    from vllm import SamplingParams

    repeated = [p for p in prompts for _ in range(group_size)]
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(repeated, sampling_params=params)
    return [out.outputs[0].text for out in outputs]


@torch.no_grad()
def evaluate(llm, examples: list[dict], prompt_template: str,
             n_eval: int, max_tokens: int) -> dict[str, float]:
    """Greedy evaluation on first n_eval validation examples."""
    from vllm import SamplingParams

    subset = examples[:n_eval]
    prompts = [build_prompt(ex, prompt_template) for ex in subset]
    gts = [build_ground_truth(ex) for ex in subset]

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            stop=["</answer>"], include_stop_str_in_output=True)
    outputs = llm.generate(prompts, sampling_params=params)
    responses = [out.outputs[0].text for out in outputs]

    totals = {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
    for resp, gt in zip(responses, gts):
        r = countdown_reward_fn(resp, gt)
        for k in totals:
            totals[k] += r[k]

    n = len(subset)
    return {f"eval/{k}": v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main GRPO train loop
# ---------------------------------------------------------------------------

def grpo_train(
    data_path: str,
    prompt_file: str,
    model_id: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    output_dir: str = "./grpo_output",
    policy_device: str = "cuda:0",
    vllm_device: str = "cuda:1",
    seed: int = 42,
    # Algorithm
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 16,
    group_size: int = 8,
    sampling_temperature: float = 0.7,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 64,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.80,
    loss_type: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip"
    ] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    # Eval / logging
    eval_every: int = 10,
    n_eval_examples: int = 256,
    wandb_project: str = "grpo-countdown",
    n_sample_rollouts: int = 3,
):
    torch.manual_seed(seed)
    random.seed(seed)

    # ---- sanity checks -------------------------------------------------- #
    assert train_batch_size % gradient_accumulation_steps == 0, \
        "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_bs = train_batch_size // gradient_accumulation_steps

    assert rollout_batch_size % group_size == 0, \
        "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout = rollout_batch_size // group_size
    assert train_batch_size >= group_size

    if epochs_per_rollout_batch > 1 or train_batch_size > rollout_batch_size:
        assert loss_type == "grpo_clip", \
            "Off-policy training requires loss_type='grpo_clip'"

    os.makedirs(output_dir, exist_ok=True)

    # ---- Prompt ---------------------------------------------------------- #
    prompt_template = load_countdown_prompt(prompt_file)
    print(f"[prompt] Loaded ({len(prompt_template)} chars) from {prompt_file}")

    # ---- W&B ------------------------------------------------------------- #
    wandb.init(
        project=wandb_project,
        config=dict(
            model_id=model_id, n_grpo_steps=n_grpo_steps,
            learning_rate=learning_rate, rollout_batch_size=rollout_batch_size,
            group_size=group_size, train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            loss_type=loss_type, use_std_normalization=use_std_normalization,
            epochs_per_rollout_batch=epochs_per_rollout_batch,
        ),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="eval_step")

    # ---- Datasets -------------------------------------------------------- #
    print("[data] Loading datasets...")
    train_examples = load_countdown_dataset(data_path, "train")
    val_examples   = load_countdown_dataset(data_path, "dev")
    print(f"[data] train={len(train_examples)}, dev={len(val_examples)}")

    # ---- Policy model ---------------------------------------------------- #
    print(f"[model] Loading: {model_id}")
    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(policy_device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- vLLM ------------------------------------------------------------ #
    print(f"[vllm] Initialising on {vllm_device} "
          f"(gpu_memory_utilization={gpu_memory_utilization}) ...")
    llm = init_vllm(model_id, device=vllm_device, seed=seed,
                    gpu_memory_utilization=gpu_memory_utilization)
    print("[vllm] Ready.")

    # ---- Optimiser ------------------------------------------------------- #
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate,
        weight_decay=0.0, betas=(0.9, 0.95),
    )

    # ---- Training loop --------------------------------------------------- #
    data_idx = train_step = eval_step = 0
    print(f"\n[train] Starting GRPO — {n_grpo_steps} steps\n")

    for grpo_step in range(1, n_grpo_steps + 1):

        # ---- Phase 1: Rollout ------------------------------------------- #
        policy.eval()
        load_policy_into_vllm_instance(policy, llm)

        batch_examples = [
            train_examples[(data_idx + i) % len(train_examples)]
            for i in range(n_prompts_per_rollout)
        ]
        data_idx = (data_idx + n_prompts_per_rollout) % len(train_examples)

        prompts = [build_prompt(ex, prompt_template) for ex in batch_examples]
        ground_truths = [build_ground_truth(ex) for ex in batch_examples]

        rollout_responses = generate_rollouts(
            llm=llm, prompts=prompts, group_size=group_size,
            temperature=sampling_temperature,
            max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens,
        )

        repeated_prompts = [p for p in prompts for _ in range(group_size)]
        repeated_gts = [g for g in ground_truths for _ in range(group_size)]

        # ---- Phase 2: Advantages ---------------------------------------- #
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=countdown_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        # ---- Phase 3: Tokenise ------------------------------------------ #
        tokenized = tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        all_input_ids = tokenized["input_ids"]      # (rollout_bs, T-1)
        all_labels    = tokenized["labels"]         # (rollout_bs, T-1)
        all_resp_mask = tokenized["response_mask"]  # (rollout_bs, T-1)

        # (rollout_bs, 1) for broadcasting over sequence length
        adv_col = advantages.unsqueeze(1).to(policy_device)
        rwd_col = raw_rewards.unsqueeze(1).to(policy_device)

        # ---- Phase 4: Old log-probs (grpo_clip only) -------------------- #
        old_log_probs_all = None
        if loss_type == "grpo_clip":
            policy.eval()
            with torch.inference_mode():
                old_lp = get_response_log_probs(
                    model=policy,
                    input_ids=all_input_ids.to(policy_device),
                    labels=all_labels.to(policy_device),
                    return_token_entropy=False,
                )
            old_log_probs_all = old_lp["log_probs"].detach()

        # ---- Phase 5: Inner gradient steps ------------------------------ #
        policy.train()

        for _epoch in range(epochs_per_rollout_batch):
            perm    = torch.randperm(rollout_batch_size)
            n_micro = rollout_batch_size // micro_bs

            optimizer.zero_grad()
            agg_loss = agg_entropy = agg_clip = 0.0

            for mi in range(n_micro):
                idx = perm[mi * micro_bs : (mi + 1) * micro_bs]

                mb_ids = all_input_ids[idx].to(policy_device)
                mb_labels = all_labels[idx].to(policy_device)
                mb_mask = all_resp_mask[idx].to(policy_device)
                mb_adv = adv_col[idx]
                mb_rwd = rwd_col[idx]
                mb_old_lp = (old_log_probs_all[idx]
                             if old_log_probs_all is not None else None)

                lp_out = get_response_log_probs(
                    model=policy, input_ids=mb_ids, labels=mb_labels,
                    return_token_entropy=True,
                )
                policy_lp = lp_out["log_probs"]
                tok_entropy = lp_out["token_entropy"]

                scaled_loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_lp,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=n_micro,
                    loss_type=loss_type,
                    raw_rewards=mb_rwd,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    cliprange=cliprange,
                )
                agg_loss += scaled_loss.item()

                with torch.no_grad():
                    mask_f = mb_mask.float()
                    n_tok = mask_f.sum().clamp(min=1)
                    agg_entropy += ((tok_entropy * mask_f).sum() / n_tok).item()
                if "clip_fraction" in meta:
                    agg_clip += meta["clip_fraction"].item()

            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            train_step += 1
            log = {
                "train/loss": agg_loss,
                "train/grad_norm": grad_norm.item(),
                "train/token_entropy": agg_entropy / n_micro,
                "train/mean_reward": reward_meta["mean_raw_reward"],
                "train/max_reward": reward_meta["max_raw_reward"],
                "train_step": train_step,
            }
            if loss_type == "grpo_clip":
                log["train/clip_fraction"] = agg_clip / n_micro
            wandb.log(log)

        # ---- Phase 6: Evaluation ---------------------------------------- #
        if grpo_step % eval_every == 0:
            print(f"\n[step {grpo_step:4d}] Evaluating ({n_eval_examples} val examples)...")
            load_policy_into_vllm_instance(policy, llm)

            eval_metrics = evaluate(
                llm=llm, examples=val_examples,
                prompt_template=prompt_template,
                n_eval=n_eval_examples, max_tokens=sampling_max_tokens,
            )
            eval_metrics["eval_step"] = eval_step
            wandb.log(eval_metrics)
            eval_step += 1

            print(f"  answer_reward={eval_metrics['eval/answer_reward']:.3f}  "
                  f"format_reward={eval_metrics['eval/format_reward']:.3f}")

            print(f"\n  === Sample rollouts (step {grpo_step}) ===")
            for i in range(min(n_sample_rollouts, len(rollout_responses))):
                gt = repeated_gts[i]
                resp = rollout_responses[i]
                rwd = countdown_reward_fn(resp, gt)
                print(f"  [{i}] GT={gt}")
                print(f"       resp[:300]: {resp[:300]!r}")
                print(f"       reward={rwd}\n")

        print(f"[step {grpo_step:4d}/{n_grpo_steps}] "
              f"loss={agg_loss:.4f}  "
              f"mean_rwd={reward_meta['mean_raw_reward']:.3f}  "
              f"grad={grad_norm.item():.3f}")

    # ---- Save ------------------------------------------------------------ #
    print(f"\n[done] Saving to {output_dir}")
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.finish()
    print("[done] Complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO training on Countdown")

    # Paths
    parser.add_argument("--data_path", type=str,
                        default="/scratch/dns5508/dataset/countdown")
    parser.add_argument("--prompt_file", type=str,
                        default="student/prompts/countdown.prompt")
    parser.add_argument("--model_id", type=str,
                        default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./grpo_output")

    # Devices
    parser.add_argument("--policy_device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:1")

    # Algorithm
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--rollout_batch_size", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--sampling_temperature", type=float, default=0.7)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--loss_type", type=str,
                        default="reinforce_with_baseline",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--use_std_normalization", action="store_true", default=True)
    parser.add_argument("--no_std_normalization", dest="use_std_normalization",
                        action="store_false")
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)

    # Eval / logging
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--n_eval_examples", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    parser.add_argument("--wandb_project", type=str, default="grpo-countdown")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    grpo_train(**vars(args))