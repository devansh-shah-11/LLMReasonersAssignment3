#!/bin/bash
#SBATCH --job-name=sft_trainer
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
RUN_NAME="${1:-sft_run_$(date +%Y%m%d_%H%M%S)}"
MAX_SAMPLES="${2:-1024}"
BATCH_SIZE="${3:-32}"
LEARNING_RATE="${4:-5e-5}"
NUM_EPOCHS="${5:-3}"

singularity exec --bind /scratch --nv \
--overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
/scratch/dns5508/ubuntu-20.04.3.sif \
/bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/home/dns5508/.local/bin:\$PATH
conda activate llmr
cd /scratch/dns5508/LLMReasonersAssignment3

python3 -m student.sft_trainer \
  --train_data_path ${DATASET_DIR}/train \
  --eval_data_path ${DATASET_DIR}/dev \
  --output_dir ${MODEL_DIR} \
  --run_name ${RUN_NAME} \
  --num_epochs ${NUM_EPOCHS} \
  --train_batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --max_train_samples ${MAX_SAMPLES} \
  --eval_steps 100 \
  --device cuda:0 \
  --eval_device cuda:1 \
  --use_wandb
"

echo "Training complete! Models saved to: ${MODEL_DIR}/${RUN_NAME}/"
