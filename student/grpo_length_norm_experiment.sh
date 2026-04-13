#!/bin/bash
#SBATCH --job-name=grpo_length_norm
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
# Experiment: Effect of Length Normalization
# Compares:
#   (1) masked_mean   -- default; global mean over all response tokens
#                        (equal gradient weight per token regardless of seq length)
#   (2) masked_normalize (--use_length_normalize) -- sum per sequence / max_gen_len
#                        (equal gradient weight per sequence regardless of seq length)
#
# Best LR: 5e-5. Best loss_type from baselines experiment assumed to be
# reinforce_with_baseline (update if different). All other params use defaults.
# ============================================================

BEST_LR=5e-5
BEST_LOSS_TYPE=reinforce_with_baseline   # update if baselines experiment says otherwise

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

echo 'Warming up CUDA kernels...'
python3 student/grpo_trainer.py \
  --data_path $DATASET_DIR \
  --prompt_file $PROMPT_FILE \
  --learning_rate $BEST_LR \
  --loss_type reinforce_with_baseline \
  --n_grpo_steps 20 \
  --output_dir ${BASE_OUTPUT_DIR}/warmup_discard \
  --wandb_project grpo-warmup-discard

echo '=============================='
echo 'Run 1/2: masked_mean (default)'
echo '=============================='
python3 student/grpo_trainer.py \
  --data_path $DATASET_DIR \
  --prompt_file $PROMPT_FILE \
  --learning_rate $BEST_LR \
  --loss_type $BEST_LOSS_TYPE \
  --output_dir ${BASE_OUTPUT_DIR}/length_norm_masked_mean \
  --wandb_run_name length_norm_masked_mean

echo '=============================='
echo 'Run 2/2: masked_normalize (--use_length_normalize)'
echo '=============================='
python3 student/grpo_trainer.py \
  --data_path $DATASET_DIR \
  --prompt_file $PROMPT_FILE \
  --learning_rate $BEST_LR \
  --loss_type $BEST_LOSS_TYPE \
  --use_length_normalize \
  --output_dir ${BASE_OUTPUT_DIR}/length_norm_masked_normalize \
  --wandb_run_name length_norm_masked_normalize
"
