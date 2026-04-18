"""Load and display a sample Countdown dataset."""

import os
import sys

from student.grpo_trainer import (
    build_ground_truth,
    build_prompt,
    load_countdown_dataset,
    load_countdown_prompt,
)


def main():
    # Configuration
    data_path = "/Users/devansh/Desktop/NYU/Sem 2/Building LLM Reasoners/Assignment 3/data-distrib/countdown"
    prompt_file = "student/prompts/countdown.prompt"
    n_samples = 5

    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"⚠️  Data path not found: {data_path}")
        print("   Please provide a valid data_path argument")
        print(f"   Usage: python load_sample_data.py <data_path> [prompt_file] [n_samples]")
        sys.exit(1)

    try:
        # Load dataset
        print(f"Loading dev dataset from {data_path}...")
        examples = load_countdown_dataset(data_path, "dev")
        print(f"✓ Loaded {len(examples)} examples\n")

        # Load prompt template
        if os.path.exists(prompt_file):
            prompt_template = load_countdown_prompt(prompt_file)
            print(f"✓ Loaded prompt template from {prompt_file}\n")
        else:
            prompt_template = "Solve this countdown problem:\n"
            print(f"⚠️  Prompt file not found, using default template\n")

        # Display samples
        print(f"{'='*80}")
        print(f"SAMPLE DATASET ({min(n_samples, len(examples))} examples)")
        print(f"{'='*80}\n")

        for i, example in enumerate(examples[:n_samples]):
            print(f"Example {i+1}:")
            print(f"  Target: {example['target']}")
            print(f"  Numbers: {example['numbers']}")
            print(f"  Ground Truth: {build_ground_truth(example)}")
            print(f"\n  Generated Prompt:")
            prompt = build_prompt(example, prompt_template)
            print(f"  {prompt[:200]}..." if len(prompt) > 200 else f"  {prompt}")
            print(f"\n  {'-'*76}\n")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Allow command line overrides
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        prompt_file = sys.argv[2]
    if len(sys.argv) > 3:
        n_samples = int(sys.argv[3])

    main()
