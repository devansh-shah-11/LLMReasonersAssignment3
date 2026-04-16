#!/bin/bash
#SBATCH --job-name=sft_sweep
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue
#SBATCH --array=0-3%2

# Configuration
DATASET_DIR="/scratch/dns5508/dataset/intellect_math"
MODEL_DIR="/scratch/dns5508/model_V2"
DATA_SIZES=(128)
LEARNING_RATES=(1e-4 5e-5)
BATCH_SIZES=(1 2)
NUM_EPOCHS=5
MIN_EVAL_STEPS=8

# Map array task ID to hyperparameter combination
TASK_ID=$SLURM_ARRAY_TASK_ID
NUM_LRS=${#LEARNING_RATES[@]}
NUM_BS=${#BATCH_SIZES[@]}

LR_IDX=$((TASK_ID / NUM_BS))
BS_IDX=$((TASK_ID % NUM_BS))

SIZE=${DATA_SIZES[0]}
LR=${LEARNING_RATES[$LR_IDX]}
BS=${BATCH_SIZES[$BS_IDX]}

SIZE_TAG=${SIZE:-full}
RUN_NAME="sft_${SIZE_TAG}_bs${BS}_lr${LR}"

echo "Task $TASK_ID: Running $RUN_NAME"

source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate llmr
cd /scratch/dns5508/LLMReasonersAssignment3

python3 -m student.sft_trainer \
  --train_data_path ${DATASET_DIR}/train/data.json \
  --eval_data_path ${DATASET_DIR}/test/data.json \
  --output_dir ${MODEL_DIR} \
  --run_name ${RUN_NAME} \
  --num_epochs ${NUM_EPOCHS} \
  --train_batch_size ${BS} \
  --learning_rate ${LR} \
  --min_eval_steps ${MIN_EVAL_STEPS} \
  $([ -n "$SIZE" ] && echo "--max_train_samples ${SIZE}") \
  --device cuda:0 \
  --eval_device cuda:1 \
  --use_wandb