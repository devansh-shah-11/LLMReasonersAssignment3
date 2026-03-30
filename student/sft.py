import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompts and outputs separately, concatenate, and build response_mask."""
    
    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    batch_size = len(prompt_strs)
    # Full sequence: prompt + output tokens
    full_ids_list = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(prompt_ids_list, output_ids_list)
    ]

    max_full_len = max(len(ids) for ids in full_ids_list)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded = torch.full((batch_size, max_full_len), pad_id, dtype=torch.long)
    response_mask_full = torch.zeros(batch_size, max_full_len, dtype=torch.long)

    for i, (full_ids, prompt_ids, output_ids) in enumerate(
        zip(full_ids_list, prompt_ids_list, output_ids_list)
    ):
        seq_len = len(full_ids)
        padded[i, :seq_len] = torch.tensor(full_ids, dtype=torch.long)
        prompt_len = len(prompt_ids)
        response_mask_full[i, prompt_len:seq_len] = 1

    # input_ids: all tokens except the last  → (batch, max_full_len - 1)
    # labels:    all tokens except the first → (batch, max_full_len - 1)
    # response_mask: marks response tokens in labels position
    input_ids     = padded[:, :-1]
    labels        = padded[:, 1:]
    response_mask = response_mask_full[:, 1:]

    return {
        "input_ids":     input_ids,
        "labels":        labels,
        "response_mask": response_mask,
    }
