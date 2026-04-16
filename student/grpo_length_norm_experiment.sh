#!/bin/bash
#SBATCH --job-name=q1-dns5508-lenorm
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
# Experiment: Effect of Length Normalization
# Compares:
#   (0) no --use_length_normalize  -- default masked_mean
#   (1) --use_length_normalize     -- masked_normalize (per-sequence weighting)
#
# Best LR from lr sweep: 2e-5
# Best loss_type from baselines: reinforce_with_baseline
# ============================================================

USE_LEN_NORM_FLAGS=(""  "--use_length_normalize")
LEN_NORM_NAMES=(masked_mean  masked_normalize)

USE_LEN_NORM=${USE_LEN_NORM_FLAGS[$SLURM_ARRAY_TASK_ID]}
LEN_NORM_NAME=${LEN_NORM_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Configuration
BEST_LR=2e-5
BEST_LOSS_TYPE=reinforce_with_baseline
DATASET_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown"
OUTPUT_DIR="../grpo_q1_dn5508/model_grpo/length_norm"
PROMPT_FILE="/gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/student/prompts/countdown.prompt"

mkdir -p ./grpo_logs_dns5508_q1
mkdir -p $OUTPUT_DIR

# Load environment variables
if [ -f /gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/.env ]; then
  export $(cat /gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/.env | grep WANDB_API_KEY | xargs)
fi

echo "=============================="
echo "GRPO Length Norm Experiment"
echo "=============================="
echo "Length norm: $LEN_NORM_NAME"
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
  $USE_LEN_NORM \
  --rollout_batch_size 16 \
  --group_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs_per_rollout_batch 1 \
  --gpu_memory_utilization 0.45 \
  --wandb_run_name "length_norm_${LEN_NORM_NAME}"

echo "Done: $(date +%Y-%m-%d_%H:%M:%S)"
