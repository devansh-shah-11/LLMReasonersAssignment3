#!/bin/bash
#SBATCH --job-name=grpo_lr_sweep
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=./logs/%j_%x_%a.out
#SBATCH --error=./logs/%j_%x_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --mail-type=all
#SBATCH --array=0-3%2

# --- Learning rate sweep ---
LR_CONFIGS=(1e-5 5e-5 1e-4 5e-4)
LEARNING_RATE=${LR_CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Configuration
DATASET_DIR="/scratch/dns5508/dataset/countdown"
OUTPUT_DIR="/scratch/dns5508/model_grpo/lr_sweep"
PROMPT_FILE="student/prompts/countdown.prompt"

mkdir -p ./logs
mkdir -p $OUTPUT_DIR

echo "=============================="
echo "GRPO LR Sweep"
echo "=============================="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Learning rate: $LEARNING_RATE"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo "=============================="

singularity exec --bind /scratch --nv \
--overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
/scratch/dns5508/ubuntu-20.04.3.sif \
/bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/home/dns5508/.local/bin:\$PATH
conda activate llmr
cd /scratch/dns5508/LLMReasonersAssignment3

python3 student/grpo_trainer.py \
  --data_path $DATASET_DIR \
  --prompt_file $PROMPT_FILE \
  --output_dir $OUTPUT_DIR \
  --policy_device cuda:0 \
  --vllm_device cuda:1 \
  --learning_rate $LEARNING_RATE \
  --rollout_batch_size 16 \
  --group_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs_per_rollout_batch 1 \
  --gpu_memory_utilization 0.45 \
  --wandb_run_name lr_${LEARNING_RATE}
"
