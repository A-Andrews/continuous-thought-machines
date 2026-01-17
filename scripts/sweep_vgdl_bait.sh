#!/bin/bash
set -euo pipefail

SWEEP_TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_GROUP="vgdl_bait_sweep_${SWEEP_TS}"

# Each entry: name|args
configs=(
  "cfg_a|--num_envs 4 --num_steps 256 --update_epochs 4 --num_minibatches 4 --ent_coef 0.01 --vf_coef 0.5 --clip_coef 0.2 --clip_vloss --lr 3e-4 --total_timesteps 2000000"
  "cfg_b|--num_envs 8 --num_steps 128 --update_epochs 8 --num_minibatches 8 --ent_coef 0.005 --vf_coef 0.5 --clip_coef 0.2 --clip_vloss --lr 2e-4 --total_timesteps 3000000"
  "cfg_c|--num_envs 8 --num_steps 256 --update_epochs 4 --num_minibatches 4 --ent_coef 0.02 --vf_coef 0.5 --clip_coef 0.1 --clip_vloss --lr 3e-4 --total_timesteps 3000000"
)

for cfg in "${configs[@]}"; do
  name="${cfg%%|*}"
  args="${cfg#*|}"
  echo "Submitting ${name}: ${args}"
  # --wandb_group keeps the sweep together; tags make runs searchable.
  sbatch scripts/run_vgdl_bait.sh \
    --wandb_group "${SWEEP_GROUP}" \
    --wandb_tags sweep "${name}" \
    ${args}
done
