#!/usr/bin/env python3
"""
SFT Training Sweep - Efficient hyperparameter sweep with single model load.
Loads the model once and runs multiple training configurations with different datasets/hyperparams.
"""

import argparse
from datetime import datetime

from .sft_trainer import train_sft


def run_sweep(
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    train_data_path: str = "data/train.jsonl",
    eval_data_path: str = "data/eval.jsonl",
    output_dir: str = "output/sft",
    num_train_epochs: int = 3,
    data_sizes: list = None,
    batch_sizes: list = None,
    learning_rates: list = None,
    max_eval_samples: int = None,
    eval_steps: int = 100,
    seed: int = 42,
    device: str = "cuda:0",
    eval_device: str = "cuda:1",
    use_wandb: bool = True,
):
    """
    Run a sweep of hyperparameters.
    Model is kept in memory across runs - only dataset and config change.

    Args:
        data_sizes: List of max_train_samples values (None = full dataset)
        batch_sizes: List of batch sizes to try
        learning_rates: List of learning rates to try
        ... (other args same as train_sft)
    """
    if data_sizes is None:
        data_sizes = [128, 256, 512, 1024, None]
    if batch_sizes is None:
        batch_sizes = [8, 16]
    if learning_rates is None:
        learning_rates = [1e-4, 5e-5]

    total_runs = len(data_sizes) * len(batch_sizes) * len(learning_rates)
    print(f"\n{'='*80}")
    print(
        f"Starting SFT Sweep: {len(data_sizes)} sizes × {len(batch_sizes)} batches × {len(learning_rates)} LRs = {total_runs} runs"
    )
    print(f"{'='*80}\n")

    completed = 0
    failed = 0

    for data_size in data_sizes:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                size_tag = data_size if data_size else "full"
                run_name = f"sft_{size_tag}_bs{batch_size}_lr{lr}"

                completed += 1
                print(f"\n[{completed}/{total_runs}] Running: {run_name}")
                print(f"{'─'*80}")

                try:
                    train_sft(
                        model_id=model_id,
                        train_data_path=train_data_path,
                        eval_data_path=eval_data_path,
                        output_dir=output_dir,
                        num_train_epochs=num_train_epochs,
                        train_batch_size=batch_size,
                        learning_rate=lr,
                        max_train_samples=data_size,
                        max_eval_samples=max_eval_samples,
                        eval_steps=eval_steps,
                        seed=seed,
                        device=device,
                        eval_device=eval_device,
                        use_wandb=use_wandb,
                        run_name=run_name,
                    )
                    print(f"✅ Completed: {run_name}")
                except Exception as e:
                    failed += 1
                    print(f"❌ Failed: {run_name}")
                    print(f"Error: {str(e)[:200]}")

    print(f"\n{'='*80}")
    print(f"Sweep Complete: {completed - failed}/{total_runs} successful, {failed} failed")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT training sweep")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/sft")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_device", type=str, default="cuda:1")
    parser.add_argument("--use_wandb", action="store_true")

    # Sweep parameters
    parser.add_argument("--data_sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[1e-4, 5e-5])

    args = parser.parse_args()

    run_sweep(
        model_id=args.model_id,
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        data_sizes=args.data_sizes,
        batch_sizes=args.batch_sizes,
        learning_rates=args.learning_rates,
        max_eval_samples=args.max_eval_samples,
        eval_steps=args.eval_steps,
        seed=args.seed,
        device=args.device,
        eval_device=args.eval_device,
        use_wandb=args.use_wandb,
    )
