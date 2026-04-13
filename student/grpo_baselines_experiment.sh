#!/bin/bash
#SBATCH --job-name=grpo_baselines
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
# Experiment: Effect of Baselines
# Compares:
#   (1) reinforce_with_baseline  -- group-normalized advantage
#   (2) no_baseline              -- raw reward, no centering
#
# Best LR from sweep: 5e-5 (fixed for all subsequent experiments).
# All other hyperparams use trainer defaults.
# ============================================================

BEST_LR=5e-5

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
echo 'Run 1/2: reinforce_with_baseline'
echo '=============================='
python3 student/grpo_trainer.py \
  --data_path $DATASET_DIR \
  --prompt_file $PROMPT_FILE \
  --learning_rate $BEST_LR \
  --loss_type reinforce_with_baseline \
  --output_dir ${BASE_OUTPUT_DIR}/baselines_reinforce \
  --wandb_run_name baseline_reinforce_with_baseline

echo '=============================='
echo 'Run 2/2: no_baseline'
echo '=============================='
python3 student/grpo_trainer.py \
  --data_path $DATASET_DIR \
  --prompt_file $PROMPT_FILE \
  --learning_rate $BEST_LR \
  --loss_type no_baseline \
  --output_dir ${BASE_OUTPUT_DIR}/baselines_no_baseline \
  --wandb_run_name baseline_no_baseline
"
