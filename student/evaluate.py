"""Minimal evaluation script for MATH and Intellect test sets."""

import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from student.drgrpo_grader import question_only_reward_fn


# Detecting backend
def _has_cuda():
    return torch.cuda.is_available()


def _get_mps_or_cpu():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def load_model_vllm(model_id: str, gpu_memory_utilization: float = 0.85):
    import os

    # vLLM calls HF's validate_repo_id which rejects paths with multiple slashes.
    # A no-slash symlink in CWD passes validation and vLLM follows it to the real dir.
    if os.path.isdir(model_id):
        link = "_local_model"
        if os.path.islink(link):
            os.remove(link)
        os.symlink(os.path.abspath(model_id), link)
        model_id = link
    return LLM(
        model=model_id,
        tokenizer=model_id,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def generate_vllm(llm, prompts, max_tokens=2048, temperature=0.0):
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def load_model_transformers(model_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def generate_transformers(
    model, tokenizer, prompts, device, max_new_tokens=2048, batch_size=1, temperature=0.0
):
    do_sample = temperature > 0.0
    all_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        print(
            f"\n  [example {i+1}/{len(prompts)}] generating (max {max_new_tokens} tokens, temp={temperature})...",
            flush=True,
        )
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        t0 = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        prompt_len = inputs["input_ids"].shape[1]
        for out in outputs:
            new_tokens = out[prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(
                f"  → {len(new_tokens)} tokens in {elapsed:.1f}s ({len(new_tokens)/elapsed:.1f} tok/s)",
                flush=True,
            )
            all_responses.append(text)
    return all_responses


def evaluate(responses, prompts, ground_truths, dataset_name="dataset", max_log_examples=10):
    """Grade responses and print category breakdown + examples."""
    correct = 0
    categories = {
        "correct_format_answer": [],
        "correct_format": [],
        "neither_correct": [],
    }

    for i, (text, gt) in enumerate(
        tqdm(zip(responses, ground_truths), total=len(responses), desc="Grading")
    ):
        reward = question_only_reward_fn(text, gt)
        fmt = reward["format_reward"]
        ans = reward["answer_reward"]

        entry = {
            "index": i,
            "prompt": prompts[i],
            "model_output": text,
            "ground_truth": gt,
            "format_reward": fmt,
            "answer_reward": ans,
        }

        if fmt == 1 and ans == 1:
            categories["correct_format_answer"].append(entry)
        elif fmt == 1 and ans == 0:
            categories["correct_format"].append(entry)
        else:
            categories["neither_correct"].append(entry)

        correct += reward["reward"]

    accuracy = correct / len(responses)

    print(f"Correct answer and format: {len(categories['correct_format_answer'])}")
    print(f"Correct format but wrong answer: {len(categories['correct_format'])}")
    print(f"Neither format nor answer correct: {len(categories['neither_correct'])}")
    print(f"Total: {len(responses)}")

    log_path = Path(__file__).parent / f"eval_log_{dataset_name.replace(' ', '_')}.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "counts": {k: len(v) for k, v in categories.items()},
                "examples": {k: v[:20] for k, v in categories.items()},
            },
            f,
            indent=2,
        )
    print(f"\n[Full log saved at {log_path}]")
    return accuracy


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for Transformers backend (ignored for vLLM)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048, help="Max tokens to generate per example"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument("--max-log-examples", type=int, default=10)
    args = parser.parse_args()

    prompt_template = load_prompt("intellect")

    use_vllm = _has_cuda()
    if use_vllm:
        print("CUDA detected → using vLLM backend")
        llm = load_model_vllm(args.model, args.gpu_memory_utilization)
        model = tokenizer = device = None
    else:
        device = _get_mps_or_cpu()
        print(f"No CUDA detected → using Transformers backend on {device}")
        model, tokenizer = load_model_transformers(args.model, device)
        llm = None

    def run_inference(prompts):
        if use_vllm:
            return generate_vllm(
                llm, prompts, max_tokens=args.max_new_tokens, temperature=args.temperature
            )
        else:
            return generate_transformers(
                model,
                tokenizer,
                prompts,
                device,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                temperature=args.temperature,
            )

    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]

    print(f"Loaded {len(prompts)} examples")
    print(f"[Sample prompt tail] ...{prompts[0][-200:]}")
    responses = run_inference(prompts)
    acc = evaluate(
        responses, prompts, gts, dataset_name="MATH", max_log_examples=args.max_log_examples
    )
    print(f"\nMATH Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
