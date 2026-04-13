#!/bin/bash
#SBATCH --job-name=grpo_std_norm
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --mail-type=all

# ============================================================
# Experiment: Effect of Group Standard Deviation Normalization
# Compares:
#   (1) --use_std_normalization  (default, DeepSeekMath/R1 style)
#         A^i = (r^i - mean) / (std + eps)
#         Pro: scale-invariant advantages
#         Con: up-weights easy/hard groups where all rewards are 0 or 1
#   (2) --no_std_normalization   (Dr. GRPO style, Liu et al. 2025)
#         A^i = r^i - mean
#         Pro: no bias toward low-variance groups
#         Con: advantages are not scale-normalized, may need lower LR
#
# Best LR: 5e-5. Best loss_type and best length_norm from prior experiments.
# Update BEST_LOSS_TYPE and BEST_LENGTH_NORM_FLAG below if needed.
# ============================================================

BEST_LR=5e-5
BEST_LOSS_TYPE=reinforce_with_baseline   # update from baselines experiment
BEST_LENGTH_NORM_FLAG=""                 # set to "--use_length_normalize" if that won

DATASET_DIR="/scratch/dns5508/dataset/countdown"
PROMPT_FILE="student/prompts/countdown.prompt"
BASE_OUTPUT_DIR="/scratch/dns5508/model_grpo"

mkdir -p ./logs

singularity exec --bind /scratch --nv \
--overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
/scratch/dns5508/ubuntu-20.04.3.sif \
/bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/home/dns5508/.local/bin:\$PATH
conda activate llmr
cd /scratch/dns5508/LLMReasonersAssignment3

echo '=============================='
echo 'Run 1/2: with std normalization (default)'
echo '=============================='
python3 student/grpo_trainer.py \
  --data_path $DATASET_DIR \
  --prompt_file $PROMPT_FILE \
  --learning_rate $BEST_LR \
  --loss_type $BEST_LOSS_TYPE \
  $BEST_LENGTH_NORM_FLAG \
  --use_std_normalization \
  --output_dir ${BASE_OUTPUT_DIR}/std_norm_true \
  --wandb_run_name std_norm_true

echo '=============================='
echo 'Run 2/2: without std normalization (Dr. GRPO)'
echo '=============================='
python3 student/grpo_trainer.py \
  --data_path $DATASET_DIR \
  --prompt_file $PROMPT_FILE \
  --learning_rate $BEST_LR \
  --loss_type $BEST_LOSS_TYPE \
  $BEST_LENGTH_NORM_FLAG \
  --no_std_normalization \
  --output_dir ${BASE_OUTPUT_DIR}/std_norm_false \
  --wandb_run_name std_norm_false
"
