#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --job-name=vgdl_bait
#SBATCH --output=logs/vgdl_bait_%j.out
#SBATCH --error=logs/vgdl_bait_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"



module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate ctm-vgdl-py38
export PATH="$HOME/.local/bin:$PATH"
# export PYTHONPATH="/gpfs3/well/costa/users/zqa082/brain-wide_strategies/RC_RL:$PYTHONPATH"


module load Python/3.11.3-GCCcore-12.3.0
source "/gpfs3/well/costa/users/zqa082/brain-wide_strategies/continuous-thought-machines/.venv/bin/activate"

export PYTHONPATH="/gpfs3/well/costa/users/zqa082/brain-wide_strategies/RC_RL:$PYTHONPATH"

set -euo pipefail
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "V100"; then
    echo "Error: This job requires a V100 GPU. Allocated GPU(s):"
    nvidia-smi
    exit 1
  fi
fi

echo "Set up complete"
# --vgdl_curriculum 0 1 2 3 \
python -m tasks.rl.train \
  --env_id VGDL \
  --vgdl_game vgfmri4_bait \
  --vgdl_games_root /well/costa/users/zqa082/brain-wide_strategies/RC_RL/all_games \
  --num_envs 4 \
  --max_environment_steps 500 \
  --neuron_select_type first-last \
  --log_dir logs/rl/vgdl_bait \
  --run_name vgdl_bait \
  --use_wandb \
  --wandb_project continuous-thought-machines \
  "$@"

echo "Done"
