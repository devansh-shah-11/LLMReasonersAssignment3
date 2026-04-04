#!/bin/bash
#SBATCH --job-name=sft_trainer_sweep
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2 
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue

# Configuration
DATASET_DIR="/scratch/dns5508/dataset/intellect_math"
MODEL_DIR="/scratch/dns5508/model"
NUM_EPOCHS=3

# Hyperparameter sweep
DATA_SIZES=(128 256 512 1024 "")
BATCH_SIZES=(2 4)
LEARNING_RATES=(1e-4 5e-5)

mkdir -p ./logs

# Run Python sweep (model loaded once, only data/config changes)
python3 -m student.sft_sweep \
  --train_data_path ${DATASET_DIR}/train/data.json \
  --eval_data_path ${DATASET_DIR}/test/data.json \
  --output_dir ${MODEL_DIR} \
  --num_epochs ${NUM_EPOCHS} \
  --data_sizes ${DATA_SIZES[@]} \
  --batch_sizes ${BATCH_SIZES[@]} \
  --learning_rates ${LEARNING_RATES[@]} \
  --device cuda:0 \
  --eval_device cuda:1 \
  --use_wandb

echo "=============================="
echo "SWEEP COMPLETE"
echo "=============================="
