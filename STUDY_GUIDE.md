# Assignment 3 Study Guide: Reasoning RL

A function-by-function walkthrough for exam prep. Every section explains **what** the function does, **why** it exists, and **how** it works mechanically.

---

## Table of Contents

1. [Big Picture](#1-big-picture)
2. [Reward / Grading (`drgrpo_grader.py`)](#2-reward--grading)
3. [SFT Helpers (`sft_helper.py`)](#3-sft-helpers)
4. [GRPO Trainer (`grpo_trainer.py`)](#4-grpo-trainer)
5. [SFT Trainer (`sft_trainer.py`)](#5-sft-trainer)
6. [Key Concepts & Exam Questions](#6-key-concepts--exam-questions)

---

## 1. Big Picture

```
Base Model (Qwen 2.5 Math 1.5B)
        │
        ▼
 [Part 3] Zero-shot evaluation (no training)
        │
        ▼
 [Part 4] SFT: supervised fine-tuning on chain-of-thought traces
        │
        ▼
 [Part 7] GRPO: RL fine-tuning with verified rewards on Countdown
```

**Three training paradigms in this assignment:**

| Method | Data needed | Training signal | Model used |
|--------|-------------|-----------------|------------|
| Zero-shot | None | None | Qwen2.5-Math-1.5B-Base |
| SFT | (question, CoT reasoning + answer) pairs | Cross-entropy on response tokens | Qwen2.5-Math-1.5B-Base |
| GRPO | (question, correct answer only) | Binary reward (correct/incorrect) | Qwen2.5-Math-1.5B-Instruct |

---

## 2. Reward / Grading

**File:** `student/drgrpo_grader.py`

This file is entirely about turning model text outputs into reward signals (0 or 1). It's the "judge" that tells the RL algorithm whether the model got the answer right.

---

### `question_only_reward_fn(response, ground_truth)`

**What it does:** The main reward function for MATH evaluation. Given a model's text response and the correct answer, returns a dict `{format_reward, answer_reward, reward}`.

**How it works:**
1. Tries to extract `\boxed{...}` from the response using `extract_answer()`
2. If no boxed answer found → `format_reward=0, answer_reward=0`
3. If boxed answer found → `format_reward=1`, then checks correctness
4. If correct → `answer_reward=1, reward=1`, else `answer_reward=0, reward=0`

**Why format_reward=1 even if wrong?** It only checks that `\boxed{}` exists for format. The model formatted correctly but got the math wrong.

**Key insight for exam:** Format reward and answer reward are separate. A model can get format reward=1 but answer reward=0 (found a boxed answer, but it's wrong). Format reward=0 means the model didn't even produce a `\boxed{}`.

---

### `r1_zero_reward_fn(response, ground_truth)`

**What it does:** A stricter reward function that requires `</think> <answer>` format (for R1-style thinking models).

**Key difference from `question_only_reward_fn`:** Requires the response to contain `</think> <answer>.....</answer>` structure. Used for R1-style models, not for the MATH baseline.

---

### `grade(model_answer, gt_answer, fast=True)`

**What it does:** The actual math comparison logic. Tries multiple methods to determine if two math expressions are equal.

**How it works (in order):**
1. `grade_answer_mathd()` — string normalization (strip units, normalize fracs, etc.)
2. `grade_answer_sympy()` — symbolic math via SymPy (handles 1/2 == 0.5, etc.)
3. If `fast=False`: also tries `is_latex_equal()` which uses `math_verify` library

**Why multiple methods?** Math has many equivalent representations: `1/2 == 0.5 == \frac{1}{2} == 50\%`. A single string match would miss most correct answers.

---

### `extract_answer(passage)` / `last_boxed_only_string()` / `remove_boxed()`

**What they do together:** Extract the content inside `\boxed{...}` from a model's response.

**How it works:**
- `last_boxed_only_string()`: scans backwards for `\boxed`, then counts matching braces to find where it ends
- `remove_boxed()`: strips the `\boxed{` prefix and `}` suffix
- `extract_answer()`: calls both

**Why "last" boxed?** Models sometimes show intermediate steps with boxed values. The **last** `\boxed{}` is the final answer.

---

### `countdown_reward_fn(response, ground_truth)` *(in grpo_trainer.py)*

**What it does:** Reward function for the Countdown task (not MATH). Returns 1.0 only if:
1. Response contains `<answer>...</answer>` tags
2. The arithmetic expression inside evaluates to the target number
3. The exact set of allowed numbers is used (each exactly once)

**How it works:**
1. Parse `ground_truth` JSON to get `target` and `allowed` numbers
2. Extract text between `<answer>` tags
3. Try evaluating each line/expression
4. Check if result equals target AND numbers used match exactly

---

## 3. SFT Helpers

**File:** `student/sft_helper.py`

These are the building blocks used by both SFT and GRPO training.

---

### `tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)`

**What it does:** Tokenizes a batch of (prompt, response) pairs and produces the tensors needed for training.

**Returns:**
- `input_ids`: shape `(batch_size, max_len - 1)` — the input to the model (all tokens except last)
- `labels`: shape `(batch_size, max_len - 1)` — the target tokens (all tokens except first, i.e. shifted by 1)
- `response_mask`: shape `(batch_size, max_len - 1)` — 1 for response tokens, 0 for prompt/padding

**How it works step by step:**
```
prompt_tokens = [5, 10, 20, 30]        # tokenize prompt
output_tokens = [100, 200, 300]        # tokenize response
full_ids      = [5, 10, 20, 30, 100, 200, 300]  # concatenate

input_ids = full_ids[:-1] = [5, 10, 20, 30, 100, 200]   # drop last
labels    = full_ids[1:]  = [10, 20, 30, 100, 200, 300]  # drop first (shifted)

# response_mask marks ONLY the response tokens in the labels tensor
# prompt positions (0,1,2) = 0, response positions (3,4,5) = 1
response_mask = [0, 0, 0, 1, 1, 1]
```

**Why shift by 1?** Causal language models predict `labels[t]` given `input_ids[:t]`. The "shift" aligns each input token with its next-token target.

**Why the response_mask?** During SFT, we only want to compute loss on the assistant's response tokens, not on the prompt tokens that were given as input.

**Padding:** All sequences in the batch are padded to the same length. Pad positions have `response_mask=0` so they don't contribute to the loss.

---

### `compute_entropy(logits)`

**What it does:** Computes the entropy of the next-token probability distribution at each position.

**Formula:** `H = -sum(p * log(p))` over the vocabulary dimension

**How it works:**
```python
log_probs = logits - logsumexp(logits)   # numerically stable log-softmax
probs = exp(log_probs)
entropy = -(probs * log_probs).sum(dim=-1)  # sum over vocab, keep (B, T)
```

**Why numerically stable?** Naive `log(softmax(logits))` can overflow/underflow for large logits. Using `logits - logsumexp(logits)` is equivalent but numerically safe.

**Why track entropy during RL?** Low entropy = overconfident model (collapsing to always producing the same tokens). High entropy = model is uncertain. Entropy collapse is a sign of mode collapse, a failure mode in RL training.

---

### `get_response_log_probs(model, input_ids, labels, return_token_entropy=False)`

**What it does:** Given a model and a tokenized batch, returns the log-probability of each token given its prefix.

**Returns:**
- `log_probs`: shape `(batch_size, seq_len)` — `log p(labels[t] | input_ids[:t])` at every position
- `token_entropy` (optional): shape `(batch_size, seq_len)` — entropy at each position

**How it works:**
```python
logits = model(input_ids).logits         # (B, T, V) — forward pass
log_probs_all = F.log_softmax(logits, -1) # (B, T, V) — normalize over vocab
# .gather picks the log-prob of the actual next token
log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
```

**Key insight:** This is called in "training mode" to get gradients (for the backward pass) and in "inference mode" (`torch.no_grad()`) to get old log-probs for GRPO-Clip.

---

### `masked_normalize(tensor, mask, normalize_constant, dim)`

**What it does:** Sums only the masked (non-zero) elements, then divides by `normalize_constant`.

**Formula:** `result[i] = sum(tensor[i] * mask[i]) / normalize_constant`

**Use case in SFT:** Computing the cross-entropy loss. Only response tokens (mask=1) contribute to the sum, then divide by a constant (usually 1.0 or sequence length).

**Use case in GRPO (length normalization):** Instead of averaging per sequence, sum the per-token losses and divide by `max_gen_len`. This way, shorter responses aren't penalized/rewarded more per-token.

---

### `sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant)`

**What it does:** Computes the SFT loss, runs backward pass, and returns the loss value for logging.

**How it works:**
```python
# For each sequence in the batch: sum log_probs of response tokens
per_seq = masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=1)
# Negative log-likelihood (we want to maximize log prob, so minimize negative)
loss = -per_seq.mean()
# Scale down for gradient accumulation (gradients accumulate over multiple calls)
scaled_loss = loss / gradient_accumulation_steps
scaled_loss.backward()  # accumulate gradients
```

**Why divide by `gradient_accumulation_steps`?** If you take k micro-steps before calling `optimizer.step()`, each step adds to the gradient buffer. Dividing by k ensures the final accumulated gradient equals the gradient you'd get from a single large batch.

**What is gradient accumulation?** Instead of waiting for a big batch that doesn't fit in GPU memory, you run many small batches and add up ("accumulate") the gradients before doing a single optimizer update. Effective batch size = micro_batch_size × gradient_accumulation_steps.

---

### `compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, normalize_by_std)`

**What it does:** Computes GRPO advantages. For each group of `group_size` responses to the same prompt, normalizes the rewards within the group.

**Two variants:**
- `normalize_by_std=True` (standard GRPO): `A = (r - mean) / (std + eps)`
- `normalize_by_std=False` (Dr. GRPO): `A = r - mean`

**How it works:**
```python
# Example: group_size=4, rewards=[1, 0, 1, 0]
# mean = 0.5, std = 0.5
# normalized (std=True):  [1, -1, 1, -1]
# normalized (std=False): [0.5, -0.5, 0.5, -0.5]
```

**Why normalize within groups?** Without a baseline, all positive-reward trajectories get reinforced equally regardless of how hard the question was. Group normalization creates relative advantages: responses better than the group average get positive advantage, worse than average get negative advantage.

**Why `normalize_by_std=False` might be better?** If ALL responses in a group are wrong (reward=0 for all), `std=0` so we'd divide by `eps`, amplifying noise. Dr. GRPO avoids this: `mean=0, A=0 for all`, meaning we skip learning from those examples.

---

### `compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs)`

**What it does:** Implements the REINFORCE policy gradient loss at the token level.

**Formula:** `loss[i,t] = -A[i] * log_prob[i,t]`

**How it works:**
```python
loss = -raw_rewards_or_advantages * policy_log_probs
# raw_rewards_or_advantages: (batch_size, 1) — same advantage for all tokens in a response
# policy_log_probs: (batch_size, seq_len)
# Broadcasting: advantage is broadcast over seq_len dimension
```

**Intuition:** If `A[i] > 0` (response was better than average), increasing `log_prob[i,t]` decreases the loss — so gradient descent will make those tokens more likely. If `A[i] < 0`, gradient descent will make those tokens less likely.

---

### `compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)`

**What it does:** Implements the PPO-style clipped loss used in GRPO.

**Formula:**
```
ratio = exp(log_prob_new - log_prob_old)   # = p_new / p_old
surr1 = ratio * A
surr2 = clip(ratio, 1-ε, 1+ε) * A
loss  = -min(surr1, surr2)
```

**Why clipping?** When you take multiple gradient steps on the same rollout batch (off-policy), the policy can drift far from the old policy. The clip prevents any single token from being updated too aggressively.

**Intuition by case:**
- `A > 0, ratio > 1+ε`: clip kicks in, no gradient (policy already increased this token enough)
- `A > 0, ratio < 1+ε`: normal gradient, make token more likely
- `A < 0, ratio < 1-ε`: clip kicks in, no gradient (policy already decreased this token enough)
- `A < 0, ratio > 1-ε`: normal gradient, make token less likely

**Why `exp(log_p_new - log_p_old)` instead of `p_new / p_old`?** Numerically stable. Log probabilities are stored as negative numbers; exponentiating their difference is safe.

**Metadata returned:** `clip_fraction` — what fraction of tokens had their ratio clipped. High clip fraction = policy is moving too fast.

---

### `compute_policy_gradient_loss(policy_log_probs, loss_type, ...)`

**What it does:** A dispatcher/wrapper that routes to the right loss function based on `loss_type`.

| `loss_type` | Uses | Advantage |
|-------------|------|-----------|
| `no_baseline` | `compute_naive_policy_gradient_loss` | Raw reward R(q,o) |
| `reinforce_with_baseline` | `compute_naive_policy_gradient_loss` | Group-normalized advantage |
| `grpo_clip` | `compute_grpo_clip_loss` | Group-normalized advantage + clipping |

---

### `masked_mean(tensor, mask, dim)`

**What it does:** Computes the mean over only the masked (non-zero) positions.

**Formula:** `result = sum(tensor * mask) / sum(mask)`

**Difference from `masked_normalize`:**
- `masked_mean`: divides by the **count of masked tokens** (variable per sequence)
- `masked_normalize`: divides by a **fixed constant** (e.g., `max_gen_len`)

**Impact on gradient:**
- `masked_mean` per sequence: shorter sequences get larger per-token gradients (each token's loss is divided by fewer tokens)
- `masked_normalize` with `max_gen_len`: all sequences get equal per-token scale regardless of length

---

### `grpo_microbatch_train_step(...)`

**What it does:** The GRPO equivalent of `sft_microbatch_train_step`. Computes the policy gradient loss, handles length normalization choice, runs backward, and returns the scaled loss.

**How it works:**
```python
# 1. Compute per-token loss (shape: batch_size × seq_len)
loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, ...)

# 2. Aggregate across sequence dimension
if use_length_normalize:
    # Sum per sequence, divide by max_gen_len (constant normalizer)
    final_loss = masked_normalize(loss, mask, normalize_constant=max_gen_len, dim=1).mean()
else:
    # Average per sequence (variable normalizer = number of response tokens)
    final_loss = masked_mean(loss, mask, dim=1).mean()

# 3. Scale for gradient accumulation and backprop
scaled_loss = final_loss / gradient_accumulation_steps
scaled_loss.backward()
```

---

## 4. GRPO Trainer

**File:** `student/grpo_trainer.py`

The full training loop. Implements Algorithm 2 from the assignment.

---

### `grpo_train(...)` — The Main Loop

**Six phases per GRPO step:**

#### Phase 1: Rollout
```python
load_policy_into_vllm_instance(policy, llm)  # sync weights to vLLM
rollout_responses = generate_rollouts(llm, prompts, group_size, temperature, ...)
```
- The policy model's current weights are copied into the vLLM inference engine
- For each of the `n_prompts_per_rollout` prompts, generate `group_size` responses (total = `rollout_batch_size`)
- Uses **temperature > 0** for diversity (needed for GRPO — if all responses are identical, advantage=0 everywhere)

#### Phase 2: Advantages
```python
advantages, raw_rewards, meta = compute_group_normalized_rewards(
    countdown_reward_fn, rollout_responses, repeated_gts, group_size, ...)
```
- Score each response with the reward function
- Compute group-normalized advantages

#### Phase 3: Tokenize
```python
tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
```
- Prepare `input_ids`, `labels`, `response_mask` for all rollout responses

#### Phase 4: Old log-probs (for GRPO-Clip / off-policy)
```python
with torch.inference_mode():
    old_lp = get_response_log_probs(policy, all_input_ids, all_labels)
old_log_probs_all = old_lp["log_probs"].detach()
```
- Compute log-probs of the rollout responses **before** any gradient steps
- These are the "reference" log-probs `π_θ_old`
- `.detach()` — not needed for gradients, just for computing the ratio in GRPO-Clip

#### Phase 5: Inner gradient steps
```python
for epoch in range(epochs_per_rollout_batch):
    for microbatch in microbatches:
        policy_lp = get_response_log_probs(policy, ...)  # with grad
        grpo_microbatch_train_step(policy_lp, ...)       # backward
    clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
```
- Run gradient steps on the collected rollouts
- `epochs_per_rollout_batch=1` = on-policy; `>1` = off-policy (requires `grpo_clip`)
- Gradient clipping at 1.0 prevents catastrophic updates

#### Phase 6: Evaluation
```python
if grpo_step % eval_every == 0:
    eval_metrics = evaluate(llm, val_examples, ...)
```
- Greedy evaluation (`temperature=0`) on validation set every `eval_every` steps

---

### `generate_rollouts(llm, prompts, group_size, temperature, max_tokens, min_tokens)`

**What it does:** Uses vLLM to generate `group_size` completions per prompt.

**Key detail:** Stops generation at `</answer>` tag (for Countdown), then appends it back:
```python
return [out.outputs[0].text + "</answer>" for out in outputs]
```
This ensures the reward function can parse the complete format.

---

### `load_policy_into_vllm_instance(policy, llm)`

**What it does:** Copies the PyTorch policy model weights into the vLLM engine in-place.

**Why needed?** The policy model (on `cuda:0`) and the vLLM engine (on `cuda:1`) are separate. After each gradient update, you need to sync weights before the next rollout phase so vLLM generates with the latest policy.

**How it works:** Directly loads `policy.state_dict()` into the vLLM model runner's weight buffers. No disk I/O, no extra GPU memory.

---

### `countdown_reward_fn(response, ground_truth)`

**What it does:** Binary reward for Countdown task.

**Returns 1.0 only if:**
1. `<answer>...</answer>` tags present
2. Some expression in the answer evaluates to the target number
3. The numbers used match exactly the allowed numbers (each used once)

**Tricky part:** The model might write steps like:
```
Step 1: 96 + 97 = 193
Step 2: 193 - 68 = 125
```
The code extracts the LHS of each `=` and tries to evaluate each expression.

---

## 5. SFT Trainer

**File:** `student/sft_trainer.py`

Standard supervised fine-tuning loop. Less complex than GRPO but uses the same helper functions.

---

### `train(args)` — The SFT Loop

**Key steps:**
1. Load JSONL data (question + chain-of-thought reasoning trace + answer)
2. Create `MathSFTDataset` → `DataLoader`
3. For each microbatch:
   - `get_response_log_probs(model, input_ids, labels)` — forward pass
   - `sft_microbatch_train_step(log_probs, response_mask, grad_accum)` — backward pass
   - Every `grad_accum` steps: `optimizer.step()`
4. Evaluate with vLLM periodically

---

### `MathSFTDataset`

**What it does:** PyTorch Dataset wrapping the Prime Intellect JSONL data.

Each record has `messages` in the format:
```json
{"messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "problem text"},
    {"role": "assistant", "content": "reasoning + \\boxed{answer}"}
]}
```

`build_prompt_and_output()` extracts the prompt and target output from this structure.

---

### `vllm_accuracy(llm, records, max_new_tokens)`

**What it does:** Evaluates model accuracy by generating responses and checking against ground truth.

**Uses `extract_answer()`:** Tries `\boxed{...}` extraction first, falls back to the last number in the output if no boxed answer found.

---

### `sync_weights_to_vllm(llm, policy_model)`

Same as `load_policy_into_vllm_instance` in grpo_trainer — copies updated policy weights into vLLM for evaluation.

---

## 6. Key Concepts & Exam Questions

### On the REINFORCE / Policy Gradient

**Q: Why do we multiply log-probs by the advantage instead of the raw reward?**

A: Using raw rewards (no baseline) has extremely high variance — all positive-reward trajectories get reinforced equally, regardless of how relatively good they were. The advantage `A = r - baseline` tells us how much *better* than expected this response was. GRPO uses group mean as the baseline.

**Q: Why is pg_loss not a meaningful evaluation metric?**

A: The policy gradient "loss" is just a scalar whose `.backward()` produces the approximate policy gradient. Its value doesn't measure anything interpretable — a higher pg_loss doesn't mean worse performance. Always report **reward** as the evaluation metric.

---

### On Gradient Accumulation

**Q: Why divide the loss by `gradient_accumulation_steps` before `.backward()`?**

A: PyTorch accumulates (adds) gradients across `.backward()` calls. If you call backward k times without zeroing gradients, you get k times the gradient you wanted. Dividing by k ensures the accumulated gradient equals what you'd get from one pass on the whole large batch.

---

### On `masked_mean` vs `masked_normalize`

**Q: What is the difference, and when does it matter?**

A: `masked_mean` divides by the **number of response tokens** per sequence. `masked_normalize` divides by a **fixed constant** (e.g., `max_gen_len=1024`).

With `masked_mean`: a short response (50 tokens) has each token's gradient scaled by `1/50`. A long response (500 tokens) scales by `1/500`. **Short responses get much larger gradient signals per token.**

With `masked_normalize(max_gen_len)`: both get scaled by `1/1024`. All token-level gradients are on the same scale regardless of response length.

**Practical implication:** `masked_mean` can destabilize training if response lengths vary a lot (large gradient norm spikes from very short responses). `masked_normalize` is more stable.

---

### On GRPO Clipping

**Q: Why does GRPO-Clip only make sense in the off-policy setting?**

A: The clip mechanism limits how much the new policy `π_θ` can deviate from the old policy `π_θ_old`. In the on-policy setting (`epochs_per_rollout_batch=1`), you take one gradient step and immediately collect new rollouts — `π_θ ≈ π_θ_old` always, so clipping never activates and has no effect. Clipping is only meaningful when you take multiple gradient steps per rollout batch (off-policy).

---

### On Std Normalization (`normalize_by_std`)

**Q: When does removing std normalization (Dr. GRPO) help?**

A: When a question is too easy (all responses correct, reward=1) or too hard (all responses wrong, reward=0), the group std≈0. Standard GRPO divides by `std + eps ≈ eps`, amplifying the advantages to be very large and introducing noisy gradients. Dr. GRPO: all advantages = 0 for that group → skip learning from uninformative batches.

---

### On the Two-GPU Setup

**Q: Why are the policy model and vLLM on separate GPUs?**

A: vLLM uses `gpu_memory_utilization=0.8` to pre-allocate KV cache for fast inference. The policy model needs its own memory for activations and gradients during training. Putting both on the same GPU would cause OOM errors.

**Q: What's the purpose of `load_policy_into_vllm_instance`?**

A: After each gradient update, the policy weights change. Before the next rollout phase, we copy the updated weights into the vLLM engine so it generates with the current policy, not the old one.

---

### On Tokenization Details

**Q: Why do `input_ids` and `labels` have length `max_len - 1`?**

A: In causal LM training, at position `t`, the model receives `input_ids[t]` and predicts `labels[t] = input_ids[t+1]`. To align these, we drop the last token from inputs and the first from labels. This "shift by 1" encodes the autoregressive prediction structure.

**Q: Why is `response_mask` applied to the `labels` (shifted) tensor, not `input_ids`?**

A: The loss is computed on `labels` (what the model predicted). We only want loss on response tokens in `labels`. Since labels = input_ids shifted left by 1, the response mask on labels correctly covers positions where the model was predicting response tokens.

---

### On the Countdown Reward Function

**Q: Why does `generate_rollouts` append `</answer>` to every response?**

A: vLLM stops generation when it sees `</answer>` (it's in the stop list). The generated text contains everything *before* that stop token but not the stop token itself. We append it back so `countdown_reward_fn` can find the closing tag when parsing.

---

### On SFT vs GRPO

**Q: Why use GRPO instead of just SFT?**

A: SFT requires high-quality annotated data (chain-of-thought traces). GRPO only needs the correct answer — the model generates its own reasoning and gets rewarded if the final answer is correct. GRPO can also discover reasoning strategies *better* than the SFT data, since it explores freely.

**Q: Why does GRPO use temperature > 0 during rollouts?**

A: With temperature=0 (greedy), all `group_size` responses to the same prompt would be identical. Then `reward_i = same for all i`, group mean = reward, advantage = 0 for all → no learning signal. Temperature introduces diversity so some responses are correct and some are wrong, creating meaningful advantages.
