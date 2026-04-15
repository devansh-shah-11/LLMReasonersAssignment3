import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from vllm import LLM, SamplingParams
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from student.sft_helper import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)

def parse_args():
    p = argparse.ArgumentParser()

    # Paths (match sbatch script exactly)
    p.add_argument("--train_data_path", type=str, required=True)
    p.add_argument("--eval_data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default="sft_run")

    # Training
    p.add_argument("--num_epochs", type=int,   default=3)
    p.add_argument("--train_batch_size", type=int,   default=2,
                   help="Physical per-step batch size (DataLoader batch_size).")
    p.add_argument("--grad_accum_steps", type=int,   default=16,
                   help="Gradient accumulation steps.")
    p.add_argument("--learning_rate",    type=float, default=2e-5)
    p.add_argument("--max_train_samples",type=int,   default=None,
                   help="Cap on unique training examples. Omit for full dataset.")
    p.add_argument("--max_seq_len",  type=int,   default=1024)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_every",   type=int,   default=50,
                   help="Evaluate every N optimizer steps.")
    p.add_argument("--seed", type=int,   default=42)

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")

    # Eval
    p.add_argument("--max_new_tokens",type=int, default=1024)

    # Devices
    p.add_argument("--device",       type=str, default="cuda:0",
                   help="Device for policy model.")
    p.add_argument("--eval_device",  type=str, default="cuda:1",
                   help="Device for vLLM engine.")

    # Logging
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--wandb_project",  type=str, default="sft_math3")

    return p.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt_and_output(record: dict) -> tuple[str, str]:
    """Extract (prompt_str, output_str) from a messages record."""
    system_content = user_content = assistant_content = ""
    for msg in record["messages"]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_content = content
        elif role == "user":
            user_content = content
        elif role == "assistant":
            assistant_content = content

    if system_content:
        prompt = f"System: {system_content}\nUser: {user_content}\nAssistant:"
    else:
        prompt = f"User: {user_content}\nAssistant:"

    return prompt, assistant_content


class MathSFTDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        prompt, output = build_prompt_and_output(record)
        return {
            "prompt":       prompt,
            "output":       output,
            "ground_truth": record.get("ground_truth", ""),
        }


def make_collate_fn(tokenizer, max_seq_len: int):
    def _collate(batch):
        result = tokenize_prompt_and_output(
            [b["prompt"] for b in batch],
            [b["output"] for b in batch],
            tokenizer,
        )
        for k in ("input_ids", "labels", "response_mask"):
            result[k] = result[k][:, :max_seq_len]
        result["ground_truths"] = [b["ground_truth"] for b in batch]
        return result
    return _collate


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_LAST_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} content, handling nested braces."""
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        return None
    start = idx + len(r'\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else None


def extract_answer(text: str) -> str:
    m = _extract_boxed(text)
    if m is not None:
        return m.strip()
    n = _LAST_NUMBER_RE.findall(text)
    return n[-1] if n else (text.strip().split()[-1] if text.strip() else "")


def is_correct(pred: str, gold: str) -> bool:
    pred = pred.strip().lower().replace(",", "")
    gold = gold.strip().lower().replace(",", "")
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

def init_vllm(model_name: str, dtype: str = "bfloat16") -> LLM:
    print(f"[vLLM] Starting engine on cuda:1 (remapped) …")
    vllm_set_random_seed(42)
    # Monkeypatch from TRL: patch world_size so vLLM doesn't think it's in
    # a multi-process context, and patch out the memory-profiling assertion
    # that fires when PyTorch has already touched the GPU before vLLM init.
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_name,
            device="cuda:1",
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.45,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
    print("[vLLM] Ready.")
    return llm


def sync_weights_to_vllm(llm: LLM, policy_model):
    """Copy policy weights directly into the live vLLM engine (in-memory, no disk I/O)."""
    state_dict = policy_model.state_dict()
    llm_model = (
        llm.llm_engine
           .model_executor
           .driver_worker
           .model_runner
           .model
    )
    llm_model.load_weights(state_dict.items())


def vllm_accuracy(
    llm: LLM,
    records: list[dict],
    max_new_tokens: int,
    eos_token: str | None = None,
) -> float:
    prompts = [build_prompt_and_output(r)[0] for r in records]
    golds   = [r.get("ground_truth", "") for r in records]

    stop = ["\n\n\n"]
    if eos_token:
        stop.append(eos_token)

    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop=stop),
    )

    correct = 0
    n_fallback = 0
    for o, g in zip(outputs, golds):
        text = o.outputs[0].text
        if _extract_boxed(text) is None:
            n_fallback += 1
        correct += is_correct(extract_answer(text), g)

    if n_fallback > 0:
        print(f"  [vllm_accuracy] used last-number fallback on {n_fallback}/{len(records)} examples")

    return correct / len(records)


# ---------------------------------------------------------------------------
# Val loss + entropy (on policy GPU, no generation needed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_val_metrics(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean_val_loss, mean_response_token_entropy)."""
    model.eval()
    total_loss = total_entropy = total_tokens = 0.0

    for batch in loader:
        input_ids     = batch["input_ids"].to(device)
        labels        = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device)

        out       = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
        log_probs = out["log_probs"]
        entropy   = out["token_entropy"]
        mask_f    = response_mask.float()
        n         = mask_f.sum().item()

        if n > 0:
            total_loss    += -(log_probs * mask_f).sum().item()
            total_entropy +=  (entropy   * mask_f).sum().item()
            total_tokens  += n

    if total_tokens == 0:
        return float("inf"), 0.0
    return total_loss / total_tokens, total_entropy / total_tokens


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set CUDA_VISIBLE_DEVICES BEFORE any CUDA initialization
    # Parse device indices (e.g., args.device="cuda:0", args.eval_device="cuda:1" → GPUs 0,1)
    policy_gpu_idx = int(args.device.split(":")[-1])
    eval_gpu_idx = int(args.eval_device.split(":")[-1])
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{policy_gpu_idx},{eval_gpu_idx}"
    print(f"[CUDA] Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # After setting CUDA_VISIBLE_DEVICES, use remapped indices (0 for policy, 1 for eval)
    policy_device = torch.device("cuda:0")
    eval_device = "cuda:1"
    
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print(f"Loading train: {args.train_data_path}")
    train_records = load_jsonl(args.train_data_path)
    print(f"Loading eval:  {args.eval_data_path}")
    eval_records  = load_jsonl(args.eval_data_path)

    random.shuffle(train_records)
    if args.max_train_samples:
        train_records = train_records[:args.max_train_samples]

    # filter records with empty assistant outputs
    n_before = len(train_records)
    train_records = [r for r in train_records if build_prompt_and_output(r)[1].strip()]
    n_skipped = n_before - len(train_records)
    if n_skipped:
        print(f"  Skipped {n_skipped} train records with empty outputs")

    print(f"Train: {len(train_records)}  |  Eval: {len(eval_records)}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- vLLM engine (must init BEFORE policy model for a clean memory baseline) ----
    torch.cuda.empty_cache()
    llm = init_vllm(args.model_name)

    # ---- Policy model (loaded after vLLM so its memory profiling baseline is clean) ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(policy_device)

    # ---- DataLoaders ----
    collate = make_collate_fn(tokenizer, args.max_seq_len)
    grad_accum   = max(1, args.grad_accum_steps)
    train_loader = DataLoader(
        MathSFTDataset(train_records),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )
    eval_loader = DataLoader(
        MathSFTDataset(eval_records),
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    # ---- Optimizer & scheduler ----
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps     = steps_per_epoch * args.num_epochs
    warmup_steps    = max(1, int(total_steps * args.warmup_ratio))
    
    eval_steps = args.eval_every

    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      weight_decay=args.weight_decay,
                      betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(
        f"\nRun: {args.run_name}\n"
        f"  train_samples={len(train_records)}  grad_accum={grad_accum}\n"
        f"  steps_per_epoch={steps_per_epoch}  total_steps={total_steps}\n"
        f"  eval_every={eval_steps} optimizer steps\n"
    )

    # ---- wandb ----
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    global_step    = 0
    best_val_loss  = float("inf")
    running_loss   = 0.0
    running_count  = 0

    # ---- Training loop ----
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        for micro_step, batch in enumerate(train_loader):
            input_ids     = batch["input_ids"].to(policy_device)
            labels        = batch["labels"].to(policy_device)
            response_mask = batch["response_mask"].to(policy_device)

            out       = get_response_log_probs(model, input_ids, labels)
            log_probs = out["log_probs"]

            scaled_loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=grad_accum,
            )
            # Unscale to get the true per-step loss for logging
            running_loss  += scaled_loss.item() * grad_accum
            running_count += 1

            if (micro_step + 1) % grad_accum != 0:
                continue

            # ---- Optimizer step ----
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            train_loss    = running_loss / running_count
            running_loss  = 0.0
            running_count = 0

            if args.use_wandb:
                wandb.log({"train/loss": train_loss,
                           "train/lr":   scheduler.get_last_lr()[0]},
                          step=global_step)

            # ---- Eval ----
            if global_step % eval_steps == 0 or global_step == total_steps:
                sync_weights_to_vllm(llm, model)
                val_loss, val_entropy = evaluate_val_metrics(
                    model, eval_loader, policy_device
                )
                val_acc = vllm_accuracy(
                    llm, eval_records, args.max_new_tokens,
                    eos_token=tokenizer.eos_token,
                )

                print(
                    f"[epoch {epoch} | step {global_step:4d}]  "
                    f"train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  "
                    f"val_entropy={val_entropy:.4f}  "
                    f"val_acc={val_acc:.3f}"
                )

                if args.use_wandb:
                    wandb.log({
                        "eval/val_loss":    val_loss,
                        "eval/val_entropy": val_entropy,
                        "eval/val_acc":     val_acc,
                    }, step=global_step)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(output_dir / "best")
                    tokenizer.save_pretrained(output_dir / "best")
                    print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

                model.train()

        if len(train_loader) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            train_loss    = running_loss / max(running_count, 1)
            running_loss  = 0.0
            running_count = 0
            if args.use_wandb:
                wandb.log({"train/loss": train_loss,
                           "train/lr":   scheduler.get_last_lr()[0]},
                          step=global_step)

        print(f"Epoch {epoch}/{args.num_epochs} done.")

    # ---- Save final model ----
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    print(f"\nFinal model saved → {output_dir / 'final'}")

    # ---- Train loss drop summary ----
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train(parse_args())