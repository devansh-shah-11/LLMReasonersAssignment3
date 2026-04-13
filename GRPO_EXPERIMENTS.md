# GRPO Experiments

Best LR from sweep: **5e-5** (fixed for all experiments below).

All scripts are submitted via `sbatch`. Each runs two sequential training runs inside a single job (2× A100 GPUs per run: cuda:0 = policy, cuda:1 = vLLM).

---

## Default behavior (what the LR sweep used)

| Setting | Default |
|---|---|
| Loss type | `reinforce_with_baseline` |
| Length normalization | `masked_mean` — global mean over all response tokens across the batch |
| Std normalization | `--use_std_normalization` (divide by group std) |

The assignment's suggested hyperparameters (targeting ~37% val accuracy) assume these defaults. The LR sweep and baselines experiment both run with these defaults.

---

## Experiment 1 — Effect of Baselines (`grpo_baselines_experiment.sh`)

**Varies:** `--loss_type`

| Run | Loss type | What the advantage is |
|---|---|---|
| `baseline_reinforce_with_baseline` | `reinforce_with_baseline` | Group-normalized reward: `(r - mean) / (std + eps)` |
| `baseline_no_baseline` | `no_baseline` | Raw reward `r` with no centering |

**Expected behavior:**
- `reinforce_with_baseline` should converge faster and more stably. Subtracting the group mean acts as a variance-reduction baseline — responses that are better than the group average get pushed up, worse ones get pushed down, regardless of the absolute reward level.
- `no_baseline` will have high gradient variance because the absolute reward scale dominates. If most rollouts score 0 (early training), almost no learning signal gets through. Expect slower convergence and noisier reward curves.
- Watch `train/grad_norm`: it should be noticeably higher for `no_baseline`.

**After this experiment:** set `BEST_LOSS_TYPE` in the subsequent scripts.

---

## Experiment 2 — Effect of Length Normalization (`grpo_length_norm_experiment.sh`)

**Varies:** `--use_length_normalize` flag

| Run | Method | Formula |
|---|---|---|
| `length_norm_masked_mean` | `masked_mean` (default) | `sum(masked losses) / total_masked_tokens` globally across batch |
| `length_norm_masked_normalize` | `masked_normalize` | `mean over batch of [sum(seq_losses) / max_gen_len]` |

**Key difference:** With `masked_mean`, a long response contributes more gradient than a short one (proportional to its token count). With `masked_normalize`, every sequence contributes equally regardless of length — short responses get upweighted per-token, long ones get downweighted.

**Expected behavior:**
- `masked_normalize` should produce more stable gradient norms (`train/grad_norm`) since sequence-length variance is removed from the gradient scale.
- `masked_mean` may bias the policy toward longer responses if longer responses happen to be rewarded — the model gets more gradient signal per long correct answer.
- In practice, `masked_normalize` (DAPO / Dr. GRPO style) often leads to better or equal performance with lower gradient norm spikes.
- Watch: `train/grad_norm` and `eval/answer_reward`.

**After this experiment:** set `BEST_LENGTH_NORM_FLAG` in the std_norm script (either `""` or `"--use_length_normalize"`).

---

## Experiment 3 — Effect of Std Normalization (`grpo_std_norm_experiment.sh`)

**Varies:** `--use_std_normalization` vs `--no_std_normalization`

| Run | Method | Formula |
|---|---|---|
| `std_norm_true` | DeepSeekMath / R1 style (default) | `A = (r - mean) / (std + eps)` |
| `std_norm_false` | Dr. GRPO style (Liu et al. 2025) | `A = r - mean` |

**Key difference:** Dividing by group std re-scales advantages so that groups with low reward variance (all correct, or all wrong) get amplified. Dr. GRPO argues this is undesirable — easy groups (all reward=1) or hard groups (all reward=0) should contribute little signal, not get up-weighted.

**Expected behavior:**
- `std_norm_false` should be more stable on easy/hard batches. When the model is very good or very bad at a prompt, the std≈0 case in `std_norm_true` can amplify near-zero differences into large advantages.
- `std_norm_true` may converge faster early in training when reward variance is meaningful, but can destabilize later.
- Watch: `train/grad_norm` spikes and `eval/answer_reward` trajectory.

---

## Running order

```
sbatch student/grpo_baselines_experiment.sh
# → check results, update BEST_LOSS_TYPE in next two scripts

sbatch student/grpo_length_norm_experiment.sh
# → check results, update BEST_LENGTH_NORM_FLAG in std_norm script

sbatch student/grpo_std_norm_experiment.sh
```
