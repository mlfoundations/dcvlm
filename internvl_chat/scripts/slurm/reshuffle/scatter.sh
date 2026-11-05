#!/bin/bash
# Runs inside the SLURM allocation: one srun task per rank. Pass 1 (scatter) of the
# global reshuffle. Env (RELEASE_DIR, ROOT, NUM_BUCKETS, SAMPLES_PER_SHARD, SEED)
# is provided by launcher.sh via sbatch --export.

echo "Sourcing software stack from scripts/slurm/system_setup.sh train"
source scripts/slurm/system_setup.sh train
echo "Using python interpreter from $(which python)"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HUB_OFFLINE=1
export TF_CPP_MIN_LOG_LEVEL=3

echo "RELEASE_DIR=${RELEASE_DIR}  ROOT=${ROOT}"
echo "NUM_BUCKETS=${NUM_BUCKETS}  SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD}  SEED=${SEED}"
echo "NTASKS=${SLURM_NTASKS}  NODES=${SLURM_JOB_NUM_NODES}"

srun --wait=0 \
    --kill-on-bad-exit=1 \
    --jobid "$SLURM_JOBID" \
    bash -c '
        export RANK=$SLURM_PROCID
        export WORLD_SIZE=$SLURM_NTASKS
        python -m internvl.train.data.reshuffle_scatter \
            --release_dir "${RELEASE_DIR}" \
            --root "${ROOT}" \
            --num_buckets "${NUM_BUCKETS}" \
            --samples_per_shard "${SAMPLES_PER_SHARD}" \
            --seed "${SEED}"
    '
