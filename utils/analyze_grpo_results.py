#!/usr/bin/env python3
"""
Analysis script for GRPO training results.

Generates:
1. Plot of validation rewards over training steps
2. Example rollouts at different training stages
3. Summary statistics
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_validation_rewards(
    run_dir: str,
    output_path: Optional[str] = None,
):
    """Plot validation rewards over training steps.

    Args:
        run_dir: path to GRPO run directory
        output_path: where to save plot (default: run_dir/validation_rewards.png)
    """
    run_path = Path(run_dir)

    # Load validation history
    val_history_path = run_path / "validation_rewards.json"
    if not val_history_path.exists():
        print(f"Error: {val_history_path} not found")
        return

    with open(val_history_path) as f:
        history = json.load(f)

    if not history:
        print("No validation history found")
        return

    steps = [h["step"] for h in history]
    rewards = [h["reward"] for h in history]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, rewards, marker="o", linewidth=2, markersize=8, color="#1f77b4")
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Validation Reward", fontsize=12)
    ax.set_title("GRPO Validation Reward over Training", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add trend line
    if len(steps) > 1:
        z = np.polyfit(steps, rewards, 1)
        p = np.poly1d(z)
        ax.plot(steps, p(steps), "--", alpha=0.5, color="red", label="Trend")
        ax.legend()

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = run_path / "validation_rewards.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    # Print statistics
    print(f"\nValidation Reward Statistics:")
    print(f"  Steps: {len(steps)}")
    print(f"  First reward: {rewards[0]:.4f}")
    print(f"  Last reward: {rewards[-1]:.4f}")
    print(f"  Mean reward: {np.mean(rewards):.4f}")
    print(f"  Max reward: {np.max(rewards):.4f}")
    print(f"  Min reward: {np.min(rewards):.4f}")
    print(f"  Improvement: {rewards[-1] - rewards[0]:.4f}")


def extract_rollout_examples(
    run_dir: str,
    output_path: Optional[str] = None,
):
    """Extract example rollouts from training logs.

    This is a template - in practice, you'd parse the training logs
    to extract rollouts at different stages.

    Args:
        run_dir: path to GRPO run directory
        output_path: where to save examples (default: run_dir/example_rollouts.txt)
    """
    run_path = Path(run_dir)

    if output_path is None:
        output_path = run_path / "example_rollouts.txt"

    # This would parse the training logs (./logs/grpo_*.log)
    # and extract rollouts at stages: early, mid, late

    with open(output_path, "w") as f:
        f.write("GRPO Training Example Rollouts\n")
        f.write("=" * 60 + "\n\n")
        f.write("Note: To see example rollouts, run the training and monitor stderr.\n")
        f.write("The trainer prints validation examples every N steps.\n\n")
        f.write("Early Stage (Step 0-100): Random/incoherent responses\n")
        f.write("Mid Stage (Step 100-500): Improving structure, still some errors\n")
        f.write("Late Stage (Step 500+): Better reasoning, more correct solutions\n")

    print(f"Saved rollout template to {output_path}")


def generate_summary_report(
    run_dir: str,
    output_path: Optional[str] = None,
):
    """Generate summary report of GRPO training.

    Args:
        run_dir: path to GRPO run directory
        output_path: where to save report (default: run_dir/summary.md)
    """
    run_path = Path(run_dir)

    if output_path is None:
        output_path = run_path / "summary.md"

    # Load validation history
    val_history_path = run_path / "validation_rewards.json"

    with open(output_path, "w") as f:
        f.write("# GRPO Training Summary\n\n")
        f.write(f"Run: {run_path.name}\n\n")

        if val_history_path.exists():
            with open(val_history_path) as vf:
                history = json.load(vf)

            if history:
                steps = [h["step"] for h in history]
                rewards = [h["reward"] for h in history]

                f.write("## Key Metrics\n\n")
                f.write(f"- **Training Steps**: {len(steps)}\n")
                f.write(f"- **Initial Validation Reward**: {rewards[0]:.4f}\n")
                f.write(f"- **Final Validation Reward**: {rewards[-1]:.4f}\n")
                f.write(f"- **Mean Validation Reward**: {np.mean(rewards):.4f}\n")
                f.write(f"- **Max Validation Reward**: {np.max(rewards):.4f}\n")
                f.write(f"- **Total Improvement**: {rewards[-1] - rewards[0]:.4f}\n")

                if rewards[-1] > rewards[0]:
                    improvement_pct = (
                        (rewards[-1] - rewards[0]) / abs(rewards[0]) * 100 if rewards[0] != 0 else 0
                    )
                    f.write(f"- **Improvement %**: {improvement_pct:.2f}%\n")

                f.write("\n## Validation Reward Curve\n\n")
                f.write("![Validation Rewards](validation_rewards.png)\n\n")

        f.write("## Expected Observations\n\n")
        f.write("1. **Validation rewards should improve over time**\n")
        f.write("   - Early training: Low/noisy rewards\n")
        f.write("   - Mid training: Steady improvement\n")
        f.write("   - Late training: Plateau or continued improvement\n\n")
        f.write("2. **Rollout quality should improve**\n")
        f.write("   - Early: Random/incoherent responses\n")
        f.write("   - Mid: Better structure, some correct solutions\n")
        f.write("   - Late: More coherent and correct reasoning\n\n")
        f.write("3. **Training should be stable**\n")
        f.write("   - Rewards shouldn't collapse\n")
        f.write("   - Gradient updates should be reasonable\n\n")
        f.write("## Next Steps\n\n")
        f.write("1. Train longer if plateauing\n")
        f.write("2. Adjust hyperparameters (learning rate, group size, etc.)\n")
        f.write("3. Evaluate on separate test set\n")

    print(f"Saved summary report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GRPO training results")
    parser.add_argument("run_dir", type=str, help="Path to GRPO run directory")
    parser.add_argument(
        "--plot_output", type=str, default=None, help="Output path for validation rewards plot"
    )
    parser.add_argument(
        "--rollouts_output", type=str, default=None, help="Output path for example rollouts"
    )
    parser.add_argument(
        "--summary_output", type=str, default=None, help="Output path for summary report"
    )

    args = parser.parse_args()

    print(f"Analyzing GRPO run: {args.run_dir}\n")

    # Generate outputs
    plot_validation_rewards(args.run_dir, args.plot_output)
    print()
    extract_rollout_examples(args.run_dir, args.rollouts_output)
    print()
    generate_summary_report(args.run_dir, args.summary_output)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
