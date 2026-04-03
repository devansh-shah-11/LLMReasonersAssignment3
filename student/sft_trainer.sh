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

# Dataset sizes ("" = full dataset)
DATA_SIZES=(128 256 512 1024 "")

# Hyperparameters
LEARNING_RATES=(1e-4 5e-5)
BATCH_SIZES=(16 32)

NUM_EPOCHS=3

# STATE TRACKING
STATE_DIR="/scratch/dns5508/sft_sweep_state"
mkdir -p $STATE_DIR

COMPLETED_FILE="$STATE_DIR/completed_runs.txt"
RESULTS_FILE="$STATE_DIR/results.csv"

touch $COMPLETED_FILE

# Initialize results file if not exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "run_name,size,batch_size,lr,status" > $RESULTS_FILE
fi

# ========================
# RUN FUNCTION
# ========================
run_training () {
    local RUN_NAME=$1
    local MAX_SAMPLES=$2
    local BATCH_SIZE=$3
    local LEARNING_RATE=$4

    # Skip if already completed
    if grep -Fxq "$RUN_NAME" $COMPLETED_FILE; then
        echo "⏭ Skipping $RUN_NAME (already completed)"
        return
    fi

    echo "=============================="
    echo "Starting: $RUN_NAME"
    echo "Samples: ${MAX_SAMPLES:-FULL}"
    echo "Batch: $BATCH_SIZE | LR: $LEARNING_RATE"
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
      $( [ -n "$MAX_SAMPLES" ] && echo "--max_train_samples $MAX_SAMPLES" ) \
      --eval_steps 100 \
      --device cuda:0 \
      --eval_device cuda:1 \
      --use_wandb
    " 2>&1)

    STATUS=$?

    echo "$OUTPUT"

    if [ $STATUS -eq 0 ]; then
        echo "✅ SUCCESS: $RUN_NAME"

        # Atomic append
        echo "$RUN_NAME" >> "${COMPLETED_FILE}.tmp"
        mv "${COMPLETED_FILE}.tmp" "$COMPLETED_FILE"

        echo "${RUN_NAME},${MAX_SAMPLES:-full},${BATCH_SIZE},${LEARNING_RATE},success" >> $RESULTS_FILE
    else
        echo "❌ FAILED: $RUN_NAME"
        echo "${RUN_NAME},${MAX_SAMPLES:-full},${BATCH_SIZE},${LEARNING_RATE},failed" >> $RESULTS_FILE
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