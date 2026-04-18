GRPO Algorithm
GRPO (Group Relative Policy Optimization) is a variant of REINFORCE designed for LLMs. The key insight: instead of a learned value baseline (which is expensive), it uses the group's own rewards as the baseline. Here's the full loop in this code:

The 6 Phases per GRPO Step
Phase 1: Rollout (grpo_trainer.py:392-405)

Sync current policy weights into vLLM (fast inference engine)
Sample n_prompts_per_rollout problems from the dataset
For each problem, generate group_size independent responses (all at the same temperature)
Result: rollout_batch_size total responses (n_prompts * group_size)

Phase 2: Compute Advantages (grpo_trainer.py:411-418)

Score every response with the reward function (binary 0/1 for Countdown)
Within each group of group_size responses to the same prompt, compute:

advantage_i = (reward_i - mean_group_reward) / (std_group_reward + eps)
This is the "group relative" part — responses better than the group average get positive advantage, worse get negative
Phase 3: Tokenize (grpo_trainer.py:421-431)

Tokenize all prompt+response pairs
Create a response_mask that is 1 only on response tokens (not the prompt)
Phase 4: Old Log-Probs (grpo_trainer.py:435-445)

For grpo_clip loss: compute log π_old(a|s) for all tokens
Needed to compute the importance ratio r = π_θ / π_old for clipping (like PPO)
Skipped for REINFORCE variants
Phase 5: Gradient Steps (grpo_trainer.py:450-499)

Loop over epochs_per_rollout_batch epochs on the same rollout data
Within each epoch, split the batch into microbatches and call .backward() on each
After all microbatches in an epoch: clip gradients and optimizer.step()
Phase 6: Eval (every eval_every steps)

Greedy decode on validation set and report reward
Loss Functions
loss_type	Formula	When
no_baseline	-log π(a|s) * reward	Vanilla REINFORCE
reinforce_with_baseline	-log π(a|s) * advantage	GRPO (no clipping)
grpo_clip	-min(r*A, clip(r,1±ε)*A)	PPO-style GRPO
The 4 Confusing Hyperparameters
Let's use the default values: rollout_batch_size=16, group_size=8, epochs_per_rollout_batch=1, gradient_accumulation_steps=16.

group_size (default: 8)
What it is: How many responses to generate per prompt before computing advantages.

With group_size=8 and a problem "make 24 from [3, 8, 3, 8]", you generate 8 independent responses. Then you normalize rewards within those 8. If 2/8 got reward 1.0, those two get positive advantage; the other 6 get negative.

Larger = more stable advantage estimate, but more generation cost
If all responses in a group get the same reward (all right or all wrong), advantage = 0 for everyone → no learning signal for that prompt
rollout_batch_size (default: 16)
What it is: Total number of (prompt, response) pairs generated per GRPO step.


n_prompts_per_rollout = rollout_batch_size // group_size  # 16 // 8 = 2 prompts
So with defaults: 2 distinct problems × 8 responses each = 16 total rollout samples.

This is the "experience buffer" for one step. Larger = more diverse problems sampled, more stable gradient — but more GPU memory and generation time.

gradient_accumulation_steps (default: 16)
What it is: How many .backward() calls before one optimizer.step().

This simulates a larger logical batch without fitting it all in GPU memory at once.


micro_bs = train_batch_size // gradient_accumulation_steps
# = (16 * 1) // 16 = 1
n_micro = rollout_batch_size // micro_bs
# = 16 // 1 = 16
With defaults: each microbatch is 1 sample, and you do 16 .backward() passes before stepping. Each pass contributes loss / gradient_accumulation_steps to the gradient (scaled so the sum = the true full-batch gradient).

Why 16 accum steps with only 16 samples? Because you might have only 16 rollout samples but want the gradient to represent the whole batch properly scaled, rather than taking a noisy step after each single sample.

epochs_per_rollout_batch (default: 1)
What it is: How many times to reuse the same rollout data for gradient steps before generating new rollouts.


for grpo_step in range(n_grpo_steps):         # one rollout generation
    generate 16 responses                      # expensive (vLLM)
    for epoch in range(epochs_per_rollout_batch):  # reuse same data N times
        shuffle and do gradient steps
epochs=1: generate → train once → discard. Safest but expensive (generation dominates)
epochs>1: generate → train multiple times on same data. More compute-efficient, but the policy drifts away from the rollout policy, making old advantages stale. This is why epochs > 1 requires grpo_clip — the PPO clip prevents the policy from moving too far from where the responses were generated.
How They All Fit Together (concrete example)
With defaults rollout_batch_size=16, group_size=8, epochs=1, grad_accum=16:


GRPO Step:
  1. Pick 2 prompts
  2. Generate 8 responses each → 16 total
  3. Score & normalize: 16 advantages
  4. Tokenize: 16 (input_ids, labels, mask) tensors
  5. Train:
     epoch 1 of 1:
       shuffle 16 indices
       microbatch 0: indices [0]    → forward → backward (scaled 1/16)
       microbatch 1: indices [1]    → forward → backward (scaled 1/16)
       ...
       microbatch 15: indices [15]  → forward → backward (scaled 1/16)
       optimizer.step()  ← gradient = sum of all 16 scaled grads
  6. scheduler.step()
The key constraint in the code (grpo_trainer.py:315-316):


train_batch_size = rollout_batch_size * epochs_per_rollout_batch
assert train_batch_size % gradient_accumulation_steps == 0
This ensures microbatches divide evenly across all data you'll process.
