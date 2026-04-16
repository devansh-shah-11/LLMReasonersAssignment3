#!/bin/bash
#SBATCH --job-name=q1-dns5508-stdnorm
#SBATCH --output=./grpo_logs_dns5508_q1/%j_%x_%a.out
#SBATCH --error=./grpo_logs_dns5508_q1/%j_%x_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=at6646@nyu.edu
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:30:00
#SBATCH --requeue
#SBATCH --array=0-1%2

# ============================================================
# Experiment: Effect of Group Standard Deviation Normalization
# Compares:
#   (0) --use_std_normalization  (default, DeepSeekMath/R1 style)
#         A^i = (r^i - mean) / (std + eps)
#         Pro: scale-invariant advantages
#         Con: up-weights easy/hard groups where all rewards are 0 or 1
#   (1) --no_std_normalization   (Dr. GRPO style, Liu et al. 2025)
#         A^i = r^i - mean
#         Pro: no bias toward low-variance groups
#         Con: advantages are not scale-normalized, may need lower LR
#
# Best LR from lr sweep: 2e-5
# Best loss_type from baselines: reinforce_with_baseline
# Best length_norm: update BEST_LENGTH_NORM_FLAG after length_norm experiment
# ============================================================

STD_NORM_FLAGS=(""  "--no_std_normalization")
STD_NORM_NAMES=(std_norm_true  std_norm_false)

STD_NORM_FLAG=${STD_NORM_FLAGS[$SLURM_ARRAY_TASK_ID]}
STD_NORM_NAME=${STD_NORM_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Configuration
BEST_LR=3e-5
BEST_LOSS_TYPE=reinforce_with_baseline
BEST_LENGTH_NORM_FLAG=""                 # set to "--use_length_normalize" if that won
DATASET_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown"
OUTPUT_DIR="../grpo_q1_dn5508/model_grpo/std_norm"
PROMPT_FILE="/gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/student/prompts/countdown.prompt"

mkdir -p ./grpo_logs_dns5508_q1
mkdir -p $OUTPUT_DIR

# Load environment variables
if [ -f /gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/.env ]; then
  export $(cat /gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/.env | grep WANDB_API_KEY | xargs)
fi

echo "=============================="
echo "GRPO Std Norm Experiment"
echo "=============================="
echo "Std norm: $STD_NORM_NAME"
echo "Learning rate: $BEST_LR"
echo "Loss type: $BEST_LOSS_TYPE"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo "=============================="

wandb login --relogin $WANDB_API_KEY

uv run python /gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/student/grpo_trainer.py \
  --data_path "$DATASET_DIR" \
  --prompt_file "$PROMPT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --policy_device cuda:0 \
  --vllm_device cuda:1 \
  --learning_rate $BEST_LR \
  --loss_type $BEST_LOSS_TYPE \
  $BEST_LENGTH_NORM_FLAG \
  $STD_NORM_FLAG \
  --rollout_batch_size 16 \
  --group_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs_per_rollout_batch 1 \
  --gpu_memory_utilization 0.45 \
  --wandb_run_name "${STD_NORM_NAME}"

echo "Done: $(date +%Y-%m-%d_%H:%M:%S)"
