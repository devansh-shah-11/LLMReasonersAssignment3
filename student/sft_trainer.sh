#!/bin/bash
#SBATCH --job-name=sft_trainer_sweep
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue

# Configuration
DATASET_DIR="/scratch/dns5508/dataset/intellect_math"
MODEL_DIR="/scratch/dns5508/model_V2"

# Dataset sizes ("" = full dataset)
DATA_SIZES=(256)
# Hyperparareters
LEARNING_RATES=(1e-5 5e-5)
BATCH_SIZES=(1 2)

NUM_EPOCHS=4
# MIN_EVAL_STEPS per batch size: bs=1 -> 8, bs=2 -> 4 (equal evaluations per epoch)
declare -A MIN_EVAL_STEPS_MAP
MIN_EVAL_STEPS_MAP[1]=16
MIN_EVAL_STEPS_MAP[2]=8

# ========================
# RUN FUNCTION
# ========================
run_training () {
    local RUN_NAME=$1
    local MAX_SAMPLES=$2
    local BATCH_SIZE=$3
    local LEARNING_RATE=$4
    local MIN_EVAL_STEPS=${MIN_EVAL_STEPS_MAP[$BATCH_SIZE]}

    echo "=============================="
    echo "Starting: $RUN_NAME"
    echo "Samples: ${MAX_SAMPLES:-FULL}"
    echo "Batch: $BATCH_SIZE | LR: $LEARNING_RATE | Min eval steps: $MIN_EVAL_STEPS"
    echo "=============================="

    OUTPUT=$(singularity exec --bind /scratch --nv \
    --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
    /scratch/dns5508/ubuntu-20.04.3.sif \
    /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr
    cd /scratch/dns5508/LLMReasonersAssignment3

    python3 -m student.sft_trainer \
      --train_data_path ${DATASET_DIR}/train/data.json \
      --eval_data_path ${DATASET_DIR}/test/data.json \
      --output_dir ${MODEL_DIR} \
      --run_name ${RUN_NAME} \
      --num_epochs ${NUM_EPOCHS} \
      --train_batch_size ${BATCH_SIZE} \
      --learning_rate ${LEARNING_RATE} \
      --min_eval_steps ${MIN_EVAL_STEPS} \
      $( [ -n "$MAX_SAMPLES" ] && echo "--max_train_samples $MAX_SAMPLES" ) \
      --device cuda:0 \
      --eval_device cuda:1 \
      --use_wandb
    " 2>&1)

    STATUS=$?

    echo "$OUTPUT"

    if [ $STATUS -eq 0 ]; then
        echo "✅ SUCCESS: $RUN_NAME"
    else
        echo "❌ FAILED: $RUN_NAME"
    fi

    echo ""
}

# Sweep Configuration

for SIZE in "${DATA_SIZES[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        for BS in "${BATCH_SIZES[@]}"; do

            SIZE_TAG=${SIZE:-full}

            RUN_NAME="sft_${SIZE_TAG}_bs${BS}_lr${LR}"

            # Run each experiment independently
            run_training "$RUN_NAME" "$SIZE" "$BS" "$LR"

        done
    done
done

echo "=============================="
echo "SWEEP COMPLETE"
echo "=============================="
