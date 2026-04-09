#!/bin/bash
#SBATCH --job-name=grpo_trainer
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2 
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue

# Configuration
DATASET_DIR="/scratch/dns5508/dataset/countdown"
OUTPUT_DIR="/scratch/dns5508/model_grpo"
PROMPT_FILE="student/prompts/countdown.prompt"

RUN_NAME="grpo_$(date +%Y%m%d_%H%M%S)"

# Create logs directory
mkdir -p ./logs
mkdir -p $OUTPUT_DIR

echo "=============================="
echo "GRPO Training"
echo "=============================="
echo "Run: $RUN_NAME"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo "=============================="

# Run GRPO training (using default hyperparameters)
singularity exec --bind /scratch --nv \
--overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
/scratch/dns5508/ubuntu-20.04.3.sif \
/bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/home/dns5508/.local/bin:\$PATH
conda activate llmr
cd /scratch/dns5508/LLMReasonersAssignment3
python3 student/grpo_trainer.py \
  --data_path "$DATASET_DIR" \
  --prompt_file "$PROMPT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --policy_device cuda:0 \
  --vllm_device cuda:1
"