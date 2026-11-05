#!/bin/bash
# Launcher for the DCVLM-Baseline pre-mixed dataset dump.
#
# Usage (from internvl_chat/):
#   SCALE=small   META_JSON=small-pool/10c-5m-15t-70i.json  NODES=8   OUT=/path/to/dcvlm_release/small  ./scripts/slurm/dump/launcher.sh
#   SCALE=xlarge_1bmodel META_JSON=large-pool/10c-5m-15t-70i.json NODES=128 OUT=/path/to/dcvlm_release/xlarge ./scripts/slurm/dump/launcher.sh
#
# Produces staging shards + per-rank .done markers under $OUT; run
# `python -m internvl.train.data.finalize_release --output_folder $OUT` afterwards.
#
# Notes:
#  - No GPU compute is needed (pre-decode / text-decode only). By default we still
#    request whole GPU nodes on $PARTITION for reliable scheduling; override
#    PARTITION/ACCOUNT/DUMP_GPUS_PER_NODE to use a CPU partition (e.g. dcgp).
#  - Sampling mirrors production (replacement ON, collate_dtypes=instruction), so
#    intended instruction-oversampling duplicates are materialized.

set -u

USER_TIME="${TIME:-}"
USER_QOS="${QOS:-}"

source .env

# ---- required / overridable knobs ---------------------------------------
export SCALE=${SCALE:?Set SCALE (e.g. small, xlarge_1bmodel, debug)}
export META_JSON=${META_JSON:?Set META_JSON (path under META_DIR, e.g. small-pool/10c-5m-15t-70i.json)}
export NODES=${NODES:-8}
export NTASKS_PER_NODE=${NTASKS_PER_NODE:-4}          # ranks per node (no GPU dependence)
export DUMP_NUM_WORKERS=${DUMP_NUM_WORKERS:-8}         # dataloader workers per rank
export SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-10000}
export PACKED_IMAGES=${PACKED_IMAGES:-24}             # max_images cap (matches training)
export COLLATE_DTYPES=${COLLATE_DTYPES:-instruction}  # match production
export SEED=${SEED:-40}
export DUMP_GPUS_PER_NODE=${DUMP_GPUS_PER_NODE:-${GPUS_PER_NODE}}

# ---- token budget by SCALE (mirrors train launcher; override via TOT_TOKENS) ----
if [ -z "${TOT_TOKENS:-}" ]; then
    case "$SCALE" in
        debug)           export TOT_TOKENS=50000000 ;;      # 50M (smoke)
        small|small_2bmodel) export TOT_TOKENS=6250000000 ;;  # 6.25B
        medium|medium_2bmodel) export TOT_TOKENS=25000000000 ;; # 25B
        large_1bmodel)   export TOT_TOKENS=50000000000 ;;   # 50B
        large2x_1bmodel) export TOT_TOKENS=100000000000 ;;  # 100B
        xlarge_1bmodel|xlarge) export TOT_TOKENS=200000000000 ;; # 200B
        *) echo "Unknown SCALE '$SCALE'; set TOT_TOKENS explicitly." ; exit 1 ;;
    esac
fi
export TOT_TOKENS

# ---- output dir (needs space for the full release; a scratch filesystem is ideal) ----
export OUT=${OUT:?Set OUT to the release output dir}

export JOB_NAME=${JOB_NAME:-dcvlm_dump_${SCALE}}
CPUS_PER_TASK=$(( CPUS_PER_NODE / NTASKS_PER_NODE ))
export TIME=${USER_TIME:-${TIME}}
export QOS=${USER_QOS:-${QOS}}

echo "=== DCVLM-Baseline dump ==="
echo "SCALE=$SCALE  META_JSON=$META_JSON  TOT_TOKENS=$TOT_TOKENS"
echo "NODES=$NODES  NTASKS_PER_NODE=$NTASKS_PER_NODE  -> WORLD_SIZE=$((NODES*NTASKS_PER_NODE)) ranks"
echo "DUMP_NUM_WORKERS=$DUMP_NUM_WORKERS  -> total workers=$((NODES*NTASKS_PER_NODE*DUMP_NUM_WORKERS))"
echo "OUT=$OUT"
mkdir -p "${SLURM_LOGS_DIR}"

output=$(sbatch \
    --account=${ACCOUNT} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --nodes=${NODES} \
    --ntasks-per-node=${NTASKS_PER_NODE} \
    --gpus-per-node=${DUMP_GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --mem=${MEMORY} \
    --job-name=${JOB_NAME} \
    --time=${TIME} \
    --output=${SLURM_LOGS_DIR}/${JOB_NAME}.out \
    --error=${SLURM_LOGS_DIR}/${JOB_NAME}.err \
    --wait-all-nodes=1 \
    --dependency=singleton \
    --exclude=${EXCLUDE_NODES} \
    --mail-type=ALL \
    --mail-user=${EMAIL} \
    scripts/slurm/dump/run.sh)

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Submitted job: $JOB_NAME ($output)"
    echo "Monitor: tail -f ${SLURM_LOGS_DIR}/${JOB_NAME}.out"
    echo "After all ranks finish: python -m internvl.train.data.finalize_release --output_folder $OUT"
else
    echo "Job submission failed with exit code $EXIT_CODE."
fi
