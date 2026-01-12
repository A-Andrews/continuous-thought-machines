#!/bin/bash
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --job-name=vgdl_function_test
#SBATCH --output=logs/vgdl_bait_%j.out
#SBATCH --error=logs/vgdl_bait_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

set -euo pipefail

module load Python/3.11.3-GCCcore-12.3.0
source "/gpfs3/well/costa/users/zqa082/brain-wide_strategies/continuous-thought-machines/.venv/bin/activate"

export PYTHONPATH="/gpfs3/well/costa/users/zqa082/brain-wide_strategies/RC_RL:$PYTHONPATH"
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

python -m tasks.rl.train \
  --env_id VGDL \
  --vgdl_game vgfmri4_bait \
  --vgdl_games_root /well/costa/users/zqa082/brain-wide_strategies/RC_RL/all_games \
  --num_envs 4 \
  --max_environment_steps 500 \
  --neuron_select_type first-last \
  --log_dir logs/rl/vgdl_bait \
  --run_name vgdl_bait

echo "Done"
