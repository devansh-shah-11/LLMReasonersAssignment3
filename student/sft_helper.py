from typing import Callable

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompts and outputs separately, concatenate, and build response_mask.

    Args:
        prompt_strs: list[str] of prompt strings.
        output_strs: list[str] of output/response strings.
        tokenizer: HuggingFace tokenizer.
    """

    # Tokenize prompts and outputs seperately
    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    batch_size = len(prompt_strs)
    # Concatenate prompt and output ids, and find max length for padding
    full_ids_list = []
    max_full_len = 0
    for prompt_ids, output_ids in zip(prompt_ids_list, output_ids_list):
        full_ids = prompt_ids + output_ids
        full_ids_list.append(full_ids)
        if len(full_ids) > max_full_len:
            max_full_len = len(full_ids)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Pad full_ids
    padded = torch.full((batch_size, max_full_len), pad_id, dtype=torch.long)
    response_mask_full = torch.zeros(batch_size, max_full_len, dtype=torch.long)

    # Create response_mask: 0 for prompt tokens, 1 for output tokens
    for i, (full_ids, prompt_ids, output_ids) in enumerate(
        zip(full_ids_list, prompt_ids_list, output_ids_list)
    ):
        seq_len = len(full_ids)
        padded[i, :seq_len] = torch.tensor(full_ids, dtype=torch.long)
        prompt_len = len(prompt_ids)
        response_mask_full[i, prompt_len:seq_len] = 1

    # Shift input_ids and labels for causal LM training
    input_ids = padded[:, :-1]
    labels = padded[:, 1:]
    response_mask = response_mask_full[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy of next-token predictions."""
    # Numerically stable: log_softmax via logsumexp
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.exp(log_probs)
    # H = -sum(p * log_p)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities from a causal LM."""
    logits = model(input_ids).logits  # (B, T, V)

    # log_probs for each position: log p(labels[t] | input_ids[:t])
    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)
    # Gather the log-prob of the actual label at each position
    log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

    result = {"log_probs": log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum masked tensor elements along a dimension and divide by normalize_constant"""
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One SFT microbatch forward+backward pass."""
    # Per-sequence masked sum (divided by normalize_constant), then mean over batch
    per_seq = masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant=normalize_constant,
        dim=1,
    )
    loss = -per_seq.mean()

    # Scale for gradient accumulation
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    return scaled_loss, {}


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute group-normalized rewards for GRPO.

    For each group of `group_size` rollouts per prompt, normalize rewards
    within the group. Optionally normalize by the group's standard deviation.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], reward function
        rollout_responses: list[str], all rollout responses (length = n_prompts * group_size)
        repeated_ground_truths: list[str], ground truths repeated group_size times
        group_size: int, number of rollouts per prompt
        advantage_eps: float, epsilon for numerical stability
        normalize_by_std: bool, whether to normalize by group std

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            - normalized_rewards: (batch_size,) tensor of normalized advantages
            - raw_rewards: (batch_size,) tensor of raw reward scores
            - metadata: dict with statistics
    """
    batch_size = len(rollout_responses)
    n_groups = batch_size // group_size

    # Compute raw rewards using the reward function
    raw_rewards_list = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        result = reward_fn(response, ground_truth)
        # Get 'reward' key from dict
        reward_val = result.get("reward", 0.0)
        raw_rewards_list.append(reward_val)

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    # Group-normalize: for each group, subtract mean and optionally divide by std
    normalized_rewards = torch.zeros_like(raw_rewards)

    for group_idx in range(n_groups):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size

        group_rewards = raw_rewards[start_idx:end_idx]
        group_mean = group_rewards.mean()

        if normalize_by_std:
            group_std = group_rewards.std(unbiased=True) + advantage_eps
            normalized = (group_rewards - group_mean) / group_std
        else:
            normalized = group_rewards - group_mean

        normalized_rewards[start_idx:end_idx] = normalized

    metadata = {
        "mean_raw_reward": raw_rewards.mean().item(),
        "std_raw_reward": raw_rewards.std().item(),
        "min_raw_reward": raw_rewards.min().item(),
        "max_raw_reward": raw_rewards.max().item(),
    }

    return normalized_rewards, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute naive policy gradient loss: -advantages * log_probs.

    Args:
        raw_rewards_or_advantages: (batch_size, 1) tensor
        policy_log_probs: (batch_size, sequence_length) tensor

    Returns:
        (batch_size, sequence_length) per-token loss
    """
    # Expand advantages to match log_probs shape
    # raw_rewards_or_advantages: (batch_size, 1)
    # policy_log_probs: (batch_size, sequence_length)
    # loss = -advantages * log_probs for each token
    loss = -raw_rewards_or_advantages * policy_log_probs
    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute GRPO-Clip loss with PPO-style clipping.

    Args:
        advantages: (batch_size, 1) tensor
        policy_log_probs: (batch_size, sequence_length) tensor
        old_log_probs: (batch_size, sequence_length) tensor
        cliprange: float, clipping range

    Returns:
        tuple[torch.Tensor, dict]:
            - loss: (batch_size, sequence_length) per-token loss
            - metadata: dict with clip statistics
    """
    # Compute probability ratio: r = exp(log_p_new - log_p_old)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Clip ratio to [1-cliprange, 1+cliprange]
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

    # Compute surrogate losses
    # surr1 = ratio * advantages
    # surr2 = clipped_ratio * advantages
    # loss = -min(surr1, surr2) for advantage maximization

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    loss = -torch.min(surr1, surr2)

    # Compute clipping fraction for logging
    is_clipped = (ratio < (1 - cliprange)) | (ratio > (1 + cliprange))
    clip_fraction = is_clipped.float().mean()

    metadata = {
        "clip_fraction": clip_fraction,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
    }

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Wrapper that delegates to appropriate PG loss function.

    Args:
        policy_log_probs: (batch_size, sequence_length)
        loss_type: one of "no_baseline", "reinforce_with_baseline", "grpo_clip"
        raw_rewards: (batch_size, 1)
        advantages: (batch_size, 1)
        old_log_probs: (batch_size, sequence_length)
        cliprange: float

    Returns:
        tuple[torch.Tensor, dict]: (per-token loss, metadata)
    """
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute masked mean of tensor along a dimension.

    Args:
        tensor: torch.Tensor
        mask: torch.Tensor (same shape as tensor, with 0s and 1s)
        dim: int | None, dimension to reduce. If None, computes global mean.

    Returns:
        torch.Tensor, the masked mean (0 where mask is all zeros)
    """
    masked = tensor * mask

    if dim is None:
        # Global mean
        masked_sum = masked.sum()
        mask_count = mask.sum().clamp(min=1)
        result = masked_sum / mask_count
    else:
        # Reduce along specified dimension
        masked_sum = masked.sum(dim=dim, keepdim=True)
        mask_count = mask.sum(dim=dim, keepdim=True).float()
        result = masked_sum / mask_count
        result = result.squeeze(dim)

    return result


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    use_length_normalize: bool = False,
    max_gen_len: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One GRPO microbatch forward+backward pass.

    Args:
        policy_log_probs: (batch_size, sequence_length)
        response_mask: (batch_size, sequence_length), mask for response tokens
        gradient_accumulation_steps: int
        loss_type: str, one of "no_baseline", "reinforce_with_baseline", "grpo_clip"
        raw_rewards: (batch_size, 1) if loss_type="no_baseline"
        advantages: (batch_size, 1) for other loss types
        old_log_probs: (batch_size, sequence_length) for grpo_clip
        cliprange: float for grpo_clip
        use_length_normalize: if True, divide per-sequence loss sum by max_gen_len
            (masked_normalize); otherwise use masked_mean per sequence (dim=1), then mean over batch.
        max_gen_len: the normalizer constant when use_length_normalize=True
            (typically sampling_max_tokens).

    Returns:
        tuple[torch.Tensor, dict]: (scaled_loss, metadata)
    """
    # Compute per-token loss based on loss_type
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    response_mask_float = response_mask.float()

    # Fix clip_fraction: the value from compute_grpo_clip_loss averages over all
    # positions including padding. Recompute here where we have response_mask.
    if "clip_fraction" in metadata and old_log_probs is not None:
        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        is_clipped = (ratio < (1 - cliprange)) | (ratio > (1 + cliprange))
        n_response_tokens = response_mask_float.sum().clamp(min=1)
        metadata["clip_fraction"] = (
            is_clipped.float() * response_mask_float
        ).sum() / n_response_tokens

    if use_length_normalize:
        # Sum per sequence, divide by max_gen_len, then mean over batch.
        # Equivalent to masked_normalize(loss, mask, normalize_constant=max_gen_len, dim=1).mean()
        assert (
            max_gen_len is not None and max_gen_len > 0
        ), "max_gen_len must be provided and > 0 when use_length_normalize=True"
        final_loss = masked_normalize(
            loss, response_mask_float, normalize_constant=max_gen_len, dim=1
        ).mean()
    else:
        # masked_mean per sequence (dim=1), then mean over batch
        final_loss = masked_mean(loss, response_mask_float, dim=1).mean()

    # Scale for gradient accumulation
    scaled_loss = final_loss / gradient_accumulation_steps
    scaled_loss.backward()

    return scaled_loss, metadata
