#!/usr/bin/env python3
"""
Evaluation script for SFT models on test sets.
Computes accuracy on Prime Intellect and MATH test datasets.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def load_jsonl(file_path):
    """Load JSONL file and return list of examples."""
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def evaluate_with_vllm(model_path, test_data_path, batch_size=32, max_tokens=512):
    """
    Evaluate model on test set using vLLM for fast generation.
    """
    print(f"Loading model from {model_path}...")
    llm = LLM(model=model_path, tensor_parallel_size=1, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading test data from {test_data_path}...")
    test_examples = load_jsonl(test_data_path)

    # Prepare prompts
    prompts = [ex["prompt"] for ex in test_examples]
    gold_outputs = [ex["output"] for ex in test_examples]

    print(f"Total test examples: {len(prompts)}")

    # Generation parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=max_tokens,
        top_p=1.0,
    )

    # Generate responses in batches
    print("Generating predictions...")
    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch_prompts = prompts[i : i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for output in outputs:
            pred_text = output.outputs[0].text.strip()
            predictions.append(pred_text)

    # Compute accuracy
    print("\nComputing metrics...")
    exact_matches = 0
    partial_matches = 0

    for pred, gold in zip(predictions, gold_outputs):
        gold = gold.strip()
        # Exact match
        if pred.lower() == gold.lower():
            exact_matches += 1
        # Partial match: check if gold is substring of prediction
        elif gold.lower() in pred.lower():
            partial_matches += 1

    exact_accuracy = exact_matches / len(predictions)
    partial_accuracy = (exact_matches + partial_matches) / len(predictions)

    results = {
        "model_path": model_path,
        "test_data_path": test_data_path,
        "num_examples": len(predictions),
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "exact_accuracy": exact_accuracy,
        "partial_accuracy": partial_accuracy,
    }

    return results, predictions


def evaluate_with_transformers(model_path, test_data_path, batch_size=8, max_tokens=512):
    """
    Evaluate model on test set using transformers (slower but single GPU).
    """
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading test data from {test_data_path}...")
    test_examples = load_jsonl(test_data_path)

    # Prepare prompts
    prompts = [ex["prompt"] for ex in test_examples]
    gold_outputs = [ex["output"] for ex in test_examples]

    print(f"Total test examples: {len(prompts)}")

    # Generate responses in batches
    print("Generating predictions...")
    predictions = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(model.device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode predictions
            for output in outputs:
                pred_text = tokenizer.decode(output, skip_special_tokens=True)
                # Remove prompt from output
                pred_text = (
                    pred_text[len(batch_prompts[0]) :].strip()
                    if len(batch_prompts) > 0
                    else pred_text.strip()
                )
                predictions.append(pred_text)

    # Compute accuracy
    print("\nComputing metrics...")
    exact_matches = 0
    partial_matches = 0

    for pred, gold in zip(predictions, gold_outputs):
        gold = gold.strip()
        # Exact match
        if pred.lower() == gold.lower():
            exact_matches += 1
        # Partial match: check if gold is substring of prediction
        elif gold.lower() in pred.lower():
            partial_matches += 1

    exact_accuracy = exact_matches / len(predictions)
    partial_accuracy = (exact_matches + partial_matches) / len(predictions)

    results = {
        "model_path": model_path,
        "test_data_path": test_data_path,
        "num_examples": len(predictions),
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "exact_accuracy": exact_accuracy,
        "partial_accuracy": partial_accuracy,
    }

    return results, predictions


def print_results(results):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {results['model_path']}")
    print(f"Test Data: {results['test_data_path']}")
    print(f"Total Examples: {results['num_examples']}")
    print(f"Exact Matches: {results['exact_matches']}/{results['num_examples']}")
    print(f"Partial Matches: {results['partial_matches']}/{results['num_examples']}")
    print(f"Exact Accuracy: {results['exact_accuracy']:.4f} ({results['exact_accuracy']*100:.2f}%)")
    print(
        f"Partial Accuracy: {results['partial_accuracy']:.4f} ({results['partial_accuracy']*100:.2f}%)"
    )
    print("=" * 60 + "\n")


def save_results(results, output_path):
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SFT models on test sets")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument(
        "--test_data_path", type=str, required=True, help="Path to test data JSONL file"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument(
        "--use_vllm", action="store_true", default=True, help="Use vLLM for fast generation"
    )
    parser.add_argument(
        "--use_transformers", action="store_true", help="Use transformers instead of vLLM"
    )
    parser.add_argument("--output_file", type=str, default=None, help="Save results to JSON file")

    args = parser.parse_args()

    # Validate paths
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not Path(args.test_data_path).exists():
        raise FileNotFoundError(f"Test data path not found: {args.test_data_path}")

    # Evaluate
    try:
        if args.use_transformers:
            print("Using transformers for evaluation...")
            results, predictions = evaluate_with_transformers(
                args.model_path,
                args.test_data_path,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
            )
        else:
            print("Using vLLM for evaluation...")
            results, predictions = evaluate_with_vllm(
                args.model_path,
                args.test_data_path,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
            )

        print_results(results)

        if args.output_file:
            save_results(results, args.output_file)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
