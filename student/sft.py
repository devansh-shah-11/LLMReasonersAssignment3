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
    """Compute per-token entropy of next-token predictions.

    Args:
        logits: (batch_size, sequence_length, vocab_size) unnormalized logits.

    Returns:
        (batch_size, sequence_length) entropy for each position.
    """
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
    """Get per-token conditional log-probabilities from a causal LM.

    Args:
        model: HuggingFace causal LM.
        input_ids: (batch_size, sequence_length) input token ids.
        labels: (batch_size, sequence_length) label token ids (shifted input_ids).
        return_token_entropy: if True, also return per-token entropy.

    Returns:
        dict with:
            "log_probs": (batch_size, sequence_length)
            "token_entropy": (batch_size, sequence_length) — only if return_token_entropy=True
    """
    logits = model(input_ids).logits  # (B, T, V)

    # log_probs for each position: log p(labels[t] | input_ids[:t])
    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)
    # Gather the log-prob of the actual label at each position
    log_probs = log_probs_all.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

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
    """Sum masked tensor elements along a dimension and divide by normalize_constant.

    Args:
        tensor: tensor to sum.
        mask: same shape as tensor; 1 for included positions, 0 otherwise.
        normalize_constant: divisor for normalization.
        dim: dimension to sum along; if None, sum over all dimensions.

    Returns:
        Normalized sum (masked elements contribute 0).
    """
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
    """One SFT microbatch forward+backward pass.

    Args:
        policy_log_probs: (batch_size, sequence_length) per-token log-probs.
        response_mask: (batch_size, sequence_length) 1 for response tokens.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        normalize_constant: divisor for the masked sum (default 1.0).

    Returns:
        (loss, metadata) where loss is the scalar microbatch loss (already
        divided by gradient_accumulation_steps and backpropagated).
    """
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