#!/bin/bash
#SBATCH --job-name=grpo_trainer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --gres=gpu:a100:2
#SBATCH --time=4:00:00
#SBATCH --partition=c12m85-a100-2
#SBATCH --output=./logs/grpo_%j.log

# Script for running GRPO training on remote server via SLURM

# Load modules
module load singularity

# SLURM parameters (can be overridden by command line arguments)
RUN_NAME=${1:-grpo_$(date +%Y%m%d_%H%M%S)}
ROLLOUT_BATCH_SIZE=${2:-8}
GROUP_SIZE=${3:-2}
BATCH_SIZE=${4:-4}
LEARNING_RATE=${5:-1e-5}
NUM_TRAIN_STEPS=${6:-1000}

# Dataset path
DATASET_PATH="/scratch/dns5508/dataset/countdown_train.jsonl"

# Output directory
OUTPUT_DIR="/scratch/dns5508/model"

# Create logs directory
mkdir -p ./logs

# Set WANDB API key (set this before running)
# export WANDB_API_KEY="your_api_key_here"

# Run training in singularity container
singularity exec \
  --nv \
  --overlay /scratch/dns5508/overlay.ext3 \
  /scratch/dns5508/container.sif \
  bash -c "
    source activate llmr && \
    cd /Users/devansh/Desktop/NYU/Sem\ 2/Building\ LLM\ Reasoners/Assignment\ 3/nyu-llm-reasoners-a3 && \
    python student/grpo_trainer.py \
      --model_name Qwen/Qwen2.5-Math-1.5B \
      --dataset_path ${DATASET_PATH} \
      --output_dir ${OUTPUT_DIR} \
      --run_name ${RUN_NAME} \
      --rollout_batch_size ${ROLLOUT_BATCH_SIZE} \
      --group_size ${GROUP_SIZE} \
      --batch_size ${BATCH_SIZE} \
      --learning_rate ${LEARNING_RATE} \
      --num_train_steps ${NUM_TRAIN_STEPS} \
      --validation_interval 50 \
      --gradient_accumulation_steps 1 \
      --cliprange 0.5 \
      --advantage_eps 1e-4 \
      --normalize_by_std
  "

echo "GRPO training job completed!"
