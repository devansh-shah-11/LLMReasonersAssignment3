#!/bin/bash
#SBATCH --job-name=evaluate_sft_sweep
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue

MODELS=(
    sft_full_bs2_lr3e-5
    sft_full_bs2_lr1e-5
    sft_1024_bs2_lr3e-5
    sft_1024_bs2_lr1e-5
    sft_512_bs2_lr3e-5
    sft_512_bs2_lr1e-5
    sft_256_bs2_lr3e-5
    sft_256_bs2_lr1e-5
    sft_128_bs2_lr3e-5
    sft_128_bs2_lr1e-5
)

MODEL_BASE=/scratch/dns5508/model_V2
RESULTS_FILE=student/sft_sweep_math_results.txt

mkdir -p ./logs

singularity exec --bind /scratch --nv \
--overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
/scratch/dns5508/ubuntu-20.04.3.sif \
/bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/home/dns5508/.local/bin:\$PATH
conda activate llmr
cd /scratch/dns5508/LLMReasonersAssignment3

MODELS=(
    sft_full_bs2_lr3e-5
    sft_full_bs2_lr1e-5
    sft_1024_bs2_lr3e-5
    sft_1024_bs2_lr1e-5
    sft_512_bs2_lr3e-5
    sft_512_bs2_lr1e-5
    sft_256_bs2_lr3e-5
    sft_256_bs2_lr1e-5
    sft_128_bs2_lr3e-5
    sft_128_bs2_lr1e-5
)

MODEL_BASE=/scratch/dns5508/model_V2
RESULTS_FILE=student/sft_sweep_math_results.txt

echo 'SFT Sweep — MATH Evaluation' > \"\$RESULTS_FILE\"
echo '==============================' >> \"\$RESULTS_FILE\"
echo \"Date: \$(date)\" >> \"\$RESULTS_FILE\"

for MODEL in \"\${MODELS[@]}\"; do
    MODEL_PATH=\"\${MODEL_BASE}/\${MODEL}\"

    echo '' | tee -a \"\$RESULTS_FILE\"
    echo '========================================' | tee -a \"\$RESULTS_FILE\"
    echo \"Model: \$MODEL\" | tee -a \"\$RESULTS_FILE\"

    if [ ! -d \"\$MODEL_PATH\" ]; then
        echo '[SKIP] directory not found' | tee -a \"\$RESULTS_FILE\"
        continue
    fi

    OUTPUT=\$(python3 student/evaluate.py \
        --model \"\$MODEL_PATH\" \
        --max-examples 500 \
        --max-log-examples 5 \
        2>&1)

    echo \"\$OUTPUT\"

    MATH_ACC=\$(echo \"\$OUTPUT\" | grep 'MATH Accuracy:' | awk '{print \$NF}')
    echo \"MATH Accuracy: \$MATH_ACC\" | tee -a \"\$RESULTS_FILE\"
done

echo '' | tee -a \"\$RESULTS_FILE\"
echo 'Sweep complete.' | tee -a \"\$RESULTS_FILE\"
"
