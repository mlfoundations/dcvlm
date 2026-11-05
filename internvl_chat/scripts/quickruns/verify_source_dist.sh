#!/bin/bash
# Scaled per-source faithfulness check for a finalized DCVLM-Baseline release.
#
# Verifies the full 166-source mixing (not just the 4 coarse buckets that check_release.py
# covers): observed per-source sampling fraction vs expected mixing weight, over many shards
# spread across all ranks. Emits a source histogram + faithfulness plot + CSV.
#
# Run from internvl_chat/ on the login node; it submits to $INTERNET_PARTITION (a CPU
# partition, see .env) via srun (reads hundreds of shards in parallel, too heavy for login):
#   OUT=/path/to/dcvlm_release/xlarge ./scripts/quickruns/verify_source_dist.sh
#   OUT=/path/to/dcvlm_release/xlarge NUM_SHARDS=1500 NUM_WORKERS=16 ./scripts/quickruns/verify_source_dist.sh
#
# Knobs (env):
#   OUT          release root         (required)
#   META         source-pool meta     (default $META_DIR/large-pool/10c-5m-15t-70i.json)
#   NUM_SHARDS   shards to sample      (default 800; each ~= one independent window)
#   NUM_WORKERS  parallel readers      (default 8)
#   SEED         RNG seed for shard subset (default 0)
#   HEAD_K       0=whole shard (unbiased); >0=first K samples/shard (faster) (default 0)
#   SPREAD       1=evenly spaced shards instead of random subset (default 0)
#   WALLTIME     srun walltime         (default 02:00:00)
#   ACCOUNT, INTERNET_PARTITION  slurm account / CPU partition (from .env)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."          # -> internvl_chat/
[ -f .env ] && source .env

: "${OUT:?set OUT to the release root}"
: "${ACCOUNT:?set ACCOUNT in .env}"
: "${INTERNET_PARTITION:?set INTERNET_PARTITION in .env}"
export META="${META:-${META_DIR}/large-pool/10c-5m-15t-70i.json}"
export NUM_SHARDS="${NUM_SHARDS:-800}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SEED="${SEED:-0}"
export HEAD_K="${HEAD_K:-0}"
export SPREAD="${SPREAD:-0}"
# NB: intentionally NOT keyed on $TIME -- that name collides with a common inherited env var.
export WALLTIME="${WALLTIME:-02:00:00}"
export RELEASE_DIR="${OUT}/release"
export OUT_PNG="${OUT}/source_dist_verify.png"
export OUT_CSV="${OUT}/source_dist_verify.csv"

if ! ls "${RELEASE_DIR}"/*.tar >/dev/null 2>&1; then
    echo "ERROR: no *.tar shards in ${RELEASE_DIR} (finalize first)." >&2
    exit 1
fi

[ "${SPREAD}" = "1" ] && SPREAD_FLAG="--spread" || SPREAD_FLAG=""
export SPREAD_FLAG

echo "=== per-source faithfulness verify ==="
echo "RELEASE_DIR = ${RELEASE_DIR}"
echo "META        = ${META}"
echo "NUM_SHARDS  = ${NUM_SHARDS}  NUM_WORKERS = ${NUM_WORKERS}  HEAD_K = ${HEAD_K}  SPREAD = ${SPREAD}"
echo "outputs     -> ${OUT_PNG}"
echo "               ${OUT_CSV}"
echo "account     = ${ACCOUNT}  partition = ${INTERNET_PARTITION}  walltime = ${WALLTIME}"
echo

srun -p "${INTERNET_PARTITION}" -A "${ACCOUNT}" -N1 -n1 -c "${NUM_WORKERS}" -t "${WALLTIME}" \
     --job-name=dcvlm_srcdist \
     bash -c '
        source scripts/slurm/system_setup.sh train
        python scripts/quickruns/verify_source_dist.py \
            --release_dir "${RELEASE_DIR}" \
            --meta "${META}" \
            --num_shards "${NUM_SHARDS}" \
            --num_workers "${NUM_WORKERS}" \
            --seed "${SEED}" \
            --head_k "${HEAD_K}" \
            ${SPREAD_FLAG} \
            --out_png "${OUT_PNG}" \
            --out_csv "${OUT_CSV}"
    '
