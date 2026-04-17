#!/bin/bash
#SBATCH --job-name=sft_trainer_sweep
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue

DATASET_DIR="/scratch/dns5508/dataset/intellect_math"
MODEL_DIR="/scratch/dns5508/model_V2"

VAL_DATA="${DATASET_DIR}/dev/data.json"
TEST_DATA="${DATASET_DIR}/test/data.json"

GRAD_ACCUM=8

DATA_SIZES=(128 256 512 1024 "")
LEARNING_RATES=(1e-5 3e-5)
BATCH_SIZES=(2)

declare -A STEPS_MAP
STEPS_MAP[128]=64
STEPS_MAP[256]=128
STEPS_MAP[512]=128
STEPS_MAP[1024]=256
STEPS_MAP[full]=1250

# Eval every 8 steps → ~8 eval points per curve
declare -A EVAL_STEPS_MAP
EVAL_STEPS_MAP[128]=8
EVAL_STEPS_MAP[256]=16
EVAL_STEPS_MAP[512]=16
EVAL_STEPS_MAP[1024]=32
EVAL_STEPS_MAP[full]=156

SIZES_LIST=()
LRS_LIST=()
BSS_LIST=()
for SIZE in "${DATA_SIZES[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        for BS in "${BATCH_SIZES[@]}"; do
            SIZES_LIST+=("$SIZE")
            LRS_LIST+=("$LR")
            BSS_LIST+=("$BS")
        done
    done
done

mkdir -p ./logs

echo "############### Sweep start: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Total runs: ${#SIZES_LIST[@]}"

for i in "${!SIZES_LIST[@]}"; do
    SIZE=${SIZES_LIST[$i]}
    LR=${LRS_LIST[$i]}
    BS=${BSS_LIST[$i]}

    SIZE_TAG=${SIZE:-full}
    NUM_STEPS=${STEPS_MAP[$SIZE_TAG]}
    EVAL_STEPS=${EVAL_STEPS_MAP[$SIZE_TAG]}
    RUN_NAME="sft_${SIZE_TAG}_bs${BS}_lr${LR}"

    echo "=============================="
    echo "Run $((i+1))/${#SIZES_LIST[@]}: $RUN_NAME"
    echo "Samples: ${SIZE:-FULL} | Steps: $NUM_STEPS | Eval every: $EVAL_STEPS"
    echo "Batch: $BS (grad_accum=$GRAD_ACCUM) | LR: $LR"
    echo "=============================="

    singularity exec --bind /scratch --nv \
      --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
      /scratch/dns5508/ubuntu-20.04.3.sif \
      /bin/bash -c "
      source /ext3/miniconda3/etc/profile.d/conda.sh
      export PATH=/home/dns5508/.local/bin:\$PATH
      conda activate llmr
      cd /scratch/dns5508/LLMReasonersAssignment3

      python3 -m student.sft_trainer \
        --train_data_path ${DATASET_DIR}/train/data.json \
        --eval_data_path  ${VAL_DATA} \
        --test_data_path  ${TEST_DATA} \
        --output_dir      ${MODEL_DIR} \
        --run_name        ${RUN_NAME} \
        --num_train_steps ${NUM_STEPS} \
        --train_batch_size ${BS} \
        --grad_accum_steps ${GRAD_ACCUM} \
        --learning_rate   ${LR} \
        --eval_steps      ${EVAL_STEPS} \
        $( [ -n "$SIZE" ] && echo "--max_train_samples $SIZE" ) \
        --device      cuda:0 \
        --eval_device cuda:1 \
        --gpu_memory_utilization 0.85 \
        --use_wandb
      "

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "SUCCESS: $RUN_NAME"
    else
        echo "FAILED:  $RUN_NAME (exit $STATUS)"
    fi
done

echo "=============================="
echo "SWEEP COMPLETE: $(date +%Y-%m-%d_%H:%M:%S)"
echo "=============================="