#!/bin/bash
# Launcher for the two-level GLOBAL RESHUFFLE of a finalized DCVLM-Baseline release.
#
# The dump writes each shard as one worker's live draw-order window, so per-shard
# SOURCE COMPOSITION is skewed (measured Deff ~= 633). This reshuffle makes every
# output shard independently representative (Deff -> ~1). See reshuffle.py.
#
# Two phases (scatter MUST finish before gather):
#   PHASE=scatter OUT=/path/to/dcvlm_release/xlarge NODES=64 ./scripts/slurm/reshuffle/launcher.sh
#   # wait for all WORLD_SIZE scatter markers, then:
#   PHASE=gather  OUT=/path/to/dcvlm_release/xlarge NODES=64 ./scripts/slurm/reshuffle/launcher.sh
#   # then finalize (single process, login OK):
#   python -m internvl.train.data.finalize_reshuffle --root $OUT/reshuffle --orig_release $OUT/release
#
# Idempotent per task (scatter) / per bucket (gather); resubmit re-runs only the
# missing ones (delete their partial *.tar first). No GPU compute is needed.

set -u

USER_TIME="${TIME:-}"
USER_QOS="${QOS:-}"

source .env

export PHASE=${PHASE:?Set PHASE=scatter or PHASE=gather}
export OUT=${OUT:?Set OUT (the finalized release root that holds release/)}
export RELEASE_DIR=${RELEASE_DIR:-${OUT}/release}
export ROOT=${ROOT:-${OUT}/reshuffle}

export NODES=${NODES:-64}
export NTASKS_PER_NODE=${NTASKS_PER_NODE:-4}
export NUM_BUCKETS=${NUM_BUCKETS:-256}
export SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-10000}
export SHUFFLE_BUFSIZE=${SHUFFLE_BUFSIZE:-20000}    # gather only; bounded streaming buffer, ~230KB/sample => 20k ~= 4.6GB RAM/rank
export SEED=${SEED:-40}
export DUMP_GPUS_PER_NODE=${DUMP_GPUS_PER_NODE:-${GPUS_PER_NODE}}

case "$PHASE" in
    scatter) RUN_SCRIPT=scripts/slurm/reshuffle/scatter.sh ;;
    gather)  RUN_SCRIPT=scripts/slurm/reshuffle/gather.sh ;;
    *) echo "PHASE must be scatter or gather (got '$PHASE')"; exit 1 ;;
esac

export JOB_NAME=${JOB_NAME:-dcvlm_reshuffle_${PHASE}}
CPUS_PER_TASK=$(( CPUS_PER_NODE / NTASKS_PER_NODE ))
export TIME=${USER_TIME:-${TIME}}
export QOS=${USER_QOS:-${QOS}}

echo "=== DCVLM reshuffle: PHASE=$PHASE ==="
echo "OUT=$OUT  RELEASE_DIR=$RELEASE_DIR  ROOT=$ROOT"
echo "NODES=$NODES NTASKS_PER_NODE=$NTASKS_PER_NODE -> WORLD_SIZE=$((NODES*NTASKS_PER_NODE))"
echo "NUM_BUCKETS=$NUM_BUCKETS SAMPLES_PER_SHARD=$SAMPLES_PER_SHARD SHUFFLE_BUFSIZE=$SHUFFLE_BUFSIZE SEED=$SEED"
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
    ${RUN_SCRIPT})

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Submitted job: $JOB_NAME ($output)"
    echo "Monitor: tail -f ${SLURM_LOGS_DIR}/${JOB_NAME}.out"
    if [ "$PHASE" = "scatter" ]; then
        echo "When all $((NODES*NTASKS_PER_NODE)) scatter markers present -> run PHASE=gather."
    else
        echo "When all $NUM_BUCKETS gather markers present -> python -m internvl.train.data.finalize_reshuffle --root $ROOT --orig_release $RELEASE_DIR"
    fi
else
    echo "Job submission failed with exit code $EXIT_CODE."
fi
