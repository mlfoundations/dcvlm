#!/bin/bash
# Runs inside the SLURM allocation: one srun task per rank. Pass 2 (gather) of the
# global reshuffle; each rank owns buckets b where b % WORLD_SIZE == RANK. Env
# (ROOT, NUM_BUCKETS, SAMPLES_PER_SHARD, SHUFFLE_BUFSIZE, SEED) from launcher.sh.

echo "Sourcing software stack from scripts/slurm/system_setup.sh train"
source scripts/slurm/system_setup.sh train
echo "Using python interpreter from $(which python)"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HUB_OFFLINE=1
export TF_CPP_MIN_LOG_LEVEL=3

echo "ROOT=${ROOT}"
echo "NUM_BUCKETS=${NUM_BUCKETS}  SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD}  SHUFFLE_BUFSIZE=${SHUFFLE_BUFSIZE}  SEED=${SEED}"
echo "NTASKS=${SLURM_NTASKS}  NODES=${SLURM_JOB_NUM_NODES}"

srun --wait=0 \
    --kill-on-bad-exit=1 \
    --jobid "$SLURM_JOBID" \
    bash -c '
        export RANK=$SLURM_PROCID
        export WORLD_SIZE=$SLURM_NTASKS
        python -m internvl.train.data.reshuffle_gather \
            --root "${ROOT}" \
            --num_buckets "${NUM_BUCKETS}" \
            --samples_per_shard "${SAMPLES_PER_SHARD}" \
            --shuffle_bufsize "${SHUFFLE_BUFSIZE}" \
            --seed "${SEED}"
    '
