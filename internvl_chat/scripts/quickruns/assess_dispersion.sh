#!/bin/bash
# Measure per-shard source-composition dispersion (design effect Deff) of a
# finalized release. Deff~1 => shards are already representative random draws;
# Deff>>1 => samples cluster within shards (a global reshuffle would help).
#
# Use it BEFORE and AFTER the reshuffle to confirm Deff drops toward 1:
#   OUT=/path/to/dcvlm_release/small ./scripts/quickruns/assess_dispersion.sh              # before
#   OUT=/path/to/dcvlm_release/small RELEASE_DIR=/path/to/dcvlm_release/small/reshuffle/release_shuffled \
#       ./scripts/quickruns/assess_dispersion.sh                                           # after
#
# Run from internvl_chat/ on the login node; it submits to $INTERNET_PARTITION (a CPU
# partition, see .env) via srun (reads many shards in parallel; too heavy for login).
#
# Knobs (env):
#   OUT          release root         (required)
#   RELEASE_DIR  dir holding *.tar    (default $OUT/release)
#   META         source-pool meta     (default $META_DIR/small-pool/10c-5m-15t-70i.json; loaded only, unused for Deff)
#   NUM_SHARDS   shards to sample      (default 60; each ~= one independent window)
#   NUM_WORKERS  parallel readers      (default 8)
#   SPREAD       1=evenly spaced shards (default 1); 0=random subset
#   MIN_GLOBAL   min total draws for a source to be assessed (default 300)
#   WALLTIME     srun walltime         (default 00:30:00)
#   ACCOUNT, INTERNET_PARTITION  slurm account / CPU partition (from .env)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."          # -> internvl_chat/
[ -f .env ] && source .env

: "${OUT:?set OUT to the release root}"
: "${ACCOUNT:?set ACCOUNT in .env}"
: "${INTERNET_PARTITION:?set INTERNET_PARTITION in .env}"
export RELEASE_DIR="${RELEASE_DIR:-${OUT}/release}"
export META="${META:-${META_DIR}/small-pool/10c-5m-15t-70i.json}"
export NUM_SHARDS="${NUM_SHARDS:-60}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SPREAD="${SPREAD:-1}"
export MIN_GLOBAL="${MIN_GLOBAL:-300}"
export WALLTIME="${WALLTIME:-00:30:00}"          # NB: not $TIME (that name collides w/ inherited env)
export SCRIPT="$(pwd)/scripts/quickruns/assess_shard_dispersion.py"

if ! ls "${RELEASE_DIR}"/*.tar >/dev/null 2>&1; then
    echo "ERROR: no *.tar shards in ${RELEASE_DIR}" >&2
    exit 1
fi

SPREAD_FLAG=""; [ "${SPREAD}" = "1" ] && SPREAD_FLAG="--spread"
export SPREAD_FLAG

echo "=== dispersion (Deff) assessment ==="
echo "RELEASE_DIR = ${RELEASE_DIR}"
echo "NUM_SHARDS = ${NUM_SHARDS}  NUM_WORKERS = ${NUM_WORKERS}  SPREAD = ${SPREAD}  MIN_GLOBAL = ${MIN_GLOBAL}"
echo "account = ${ACCOUNT}  partition = ${INTERNET_PARTITION}  walltime = ${WALLTIME}"
echo

srun -p "${INTERNET_PARTITION}" -A "${ACCOUNT}" -N1 -n1 -c"${NUM_WORKERS}" -t "${WALLTIME}" --job-name=dcvlm_deff \
    bash -c '
        source scripts/slurm/system_setup.sh train
        export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
        python "${SCRIPT}" \
            --release_dir "${RELEASE_DIR}" \
            --meta "${META}" \
            --num_shards "${NUM_SHARDS}" \
            --num_workers "${NUM_WORKERS}" \
            --min_global "${MIN_GLOBAL}" \
            ${SPREAD_FLAG}
    '
