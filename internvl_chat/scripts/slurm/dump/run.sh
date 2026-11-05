#!/bin/bash
# Runs inside the SLURM allocation: one srun task per rank (no torchrun, no model,
# no GPU compute). Each rank dumps its share of the DCVLM-Baseline pre-mixed release.
# Env (META_DIR, META_JSON, OUT, TOT_TOKENS, SEED, PACKED_IMAGES, SAMPLES_PER_SHARD,
# DUMP_NUM_WORKERS, COLLATE_DTYPES) is provided by launcher.sh via sbatch --export.

echo "Sourcing software stack from scripts/slurm/system_setup.sh train"
source scripts/slurm/system_setup.sh train
echo "Using python interpreter from $(which python)"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"   # so `import internvl` works
export HF_HUB_OFFLINE=1
export TF_CPP_MIN_LOG_LEVEL=3
# each rank drives DUMP_NUM_WORKERS dataloader workers; spread CPU threads across them
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / (DUMP_NUM_WORKERS > 0 ? DUMP_NUM_WORKERS : 1) ))
[ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1

mkdir -p "${OUT}"

echo "OUT=${OUT}"
echo "META=${META_DIR}/${META_JSON}"
echo "TOT_TOKENS=${TOT_TOKENS}  SEED=${SEED}  PACKED_IMAGES=${PACKED_IMAGES}"
echo "SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD}  DUMP_NUM_WORKERS=${DUMP_NUM_WORKERS}  COLLATE_DTYPES=${COLLATE_DTYPES}"
echo "NTASKS=${SLURM_NTASKS}  NODES=${SLURM_JOB_NUM_NODES}  NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

# one task per rank; RANK/WORLD_SIZE derived from SLURM per task (single-quoted so
# $SLURM_PROCID / $SLURM_NTASKS expand inside each task, not once on the launch node)
srun --wait=0 \
    --kill-on-bad-exit=1 \
    --jobid "$SLURM_JOBID" \
    bash -c '
        export RANK=$SLURM_PROCID
        export WORLD_SIZE=$SLURM_NTASKS
        python -m internvl.train.data.dump_release \
            --meta_path "${META_DIR}/${META_JSON}" \
            --output_folder "${OUT}" \
            --global_token_budget "${TOT_TOKENS}" \
            --num_workers "${DUMP_NUM_WORKERS}" \
            --samples_per_shard "${SAMPLES_PER_SHARD}" \
            --seed "${SEED}" \
            --num_images_expected "${PACKED_IMAGES}" \
            --collate_dtypes "${COLLATE_DTYPES}" \
            --allow_distributed_replacement True
    '
