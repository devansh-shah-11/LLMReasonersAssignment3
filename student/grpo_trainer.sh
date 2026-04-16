#!/bin/bash
#SBATCH --job-name=q1-dns5508
#SBATCH --output=./grpo_logs_dns5508_q1/%j_%x.out
#SBATCH --error=./grpo_logs_dns5508_q1/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=at6646@nyu.edu
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --requeue
# Configuration
DATASET_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown"
OUTPUT_DIR="../grpo_q1_dn5508/model_grpo"
PROMPT_FILE="/gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/student/prompts/countdown.prompt"

RUN_NAME="grpo_$(date +%Y%m%d_%H%M%S)"

# Create logs directory
mkdir -p ./logs
mkdir -p $OUTPUT_DIR

# Load environment variables
if [ -f /gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/.env ]; then
  export $(cat /gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/.env | grep WANDB_API_KEY | xargs)
fi

echo "=============================="
echo "GRPO Training"
echo "=============================="
echo "Run: $RUN_NAME"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo "=============================="

# Run GRPO training (using default hyperparameters)


wandb login --relogin $WANDB_API_KEY

uv run python /gpfs/scratch/an4462/at6646/dns5508/LLMReasonersAssignment3/student/grpo_trainer.py \
  --data_path "$DATASET_DIR" \
  --prompt_file "$PROMPT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --policy_device cuda:0 \
  --vllm_device cuda:1 \
  --rollout_batch_size 16 \
  --group_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs_per_rollout_batch 1 \
  --gpu_memory_utilization 0.85 \
  --wandb_run_name "$RUN_NAME"
