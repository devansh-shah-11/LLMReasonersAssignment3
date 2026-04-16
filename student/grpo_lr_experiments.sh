#!/bin/bash
#SBATCH --job-name=q1-dns5508-lrsweep
#SBATCH --output=./grpo_logs_dns5508_q1/%j_%x_%a.out
#SBATCH --error=./grpo_logs_dns5508_q1/%j_%x_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=at6646@nyu.edu
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --array=0-3%2

# --- Learning rate sweep ---
LR_CONFIGS=(1e-5 2e-5 5e-4 1e-4)
LEARNING_RATE=${LR_CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Configuration
DATASET_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown"
OUTPUT_DIR="../grpo_q1_dn5508/model_grpo/lr_sweep"
PROMPT_FILE="/gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/student/prompts/countdown.prompt"

mkdir -p ./grpo_logs_dns5508_q1
mkdir -p $OUTPUT_DIR

# Load environment variables
if [ -f /gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/.env ]; then
  export $(cat /gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/.env | grep WANDB_API_KEY | xargs)
fi

echo "=============================="
echo "GRPO LR Sweep"
echo "=============================="
echo "Learning rate: $LEARNING_RATE"
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
  --learning_rate $LEARNING_RATE \
  --rollout_batch_size 16 \
  --group_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs_per_rollout_batch 1 \
  --gpu_memory_utilization 0.45 \
  --wandb_run_name "lr_sweep_${LEARNING_RATE}"

echo "Done: $(date +%Y-%m-%d_%H:%M:%S)"
