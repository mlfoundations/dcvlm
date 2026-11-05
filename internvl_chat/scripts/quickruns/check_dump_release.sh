#!/bin/bash
# Structural-check a finalized DCVLM-Baseline release (offline, no GPU/model).
#
# Run from internvl_chat/ on the login node; it submits the check to
# $INTERNET_PARTITION (a CPU partition, see .env) via srun, since it decodes images:
#
#   OUT=/path/to/dcvlm_release/smoke ./scripts/quickruns/check_dump_release.sh
#
# Prereq: the release must already be finalized (run this first, it's instant):
#   python -m internvl.train.data.finalize_release --output_folder $OUT
#
# Overridable knobs (env vars):
#   OUT          release root (required)
#   MAX_SAMPLES  samples to inspect (default 20000)
#   TOL          allowed abs deviation from target data_type fractions (default 0.03)
#   ACCOUNT, INTERNET_PARTITION  slurm account / CPU partition (from .env)

set -euo pipefail

# resolve repo internvl_chat/ regardless of where we're invoked from
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

source .env

: "${OUT:?set OUT to the release root}"
: "${ACCOUNT:?set ACCOUNT in .env}"
: "${INTERNET_PARTITION:?set INTERNET_PARTITION in .env}"
export MAX_SAMPLES="${MAX_SAMPLES:-20000}"
export TOL="${TOL:-0.03}"
export RELEASE_DIR="${OUT}/release"

if [ ! -d "${RELEASE_DIR}" ]; then
    echo "ERROR: ${RELEASE_DIR} not found." >&2
    echo "       Finalize first: python -m internvl.train.data.finalize_release --output_folder ${OUT}" >&2
    exit 1
fi
if ! ls "${RELEASE_DIR}"/*.tar >/dev/null 2>&1; then
    echo "ERROR: no *.tar shards in ${RELEASE_DIR} (did finalize run?)." >&2
    exit 1
fi

echo "=== structural check ==="
echo "RELEASE_DIR = ${RELEASE_DIR}"
echo "MAX_SAMPLES = ${MAX_SAMPLES}  TOL = ${TOL}"
echo "account     = ${ACCOUNT}  partition = ${INTERNET_PARTITION}"
echo

# env vars above propagate via srun's default --export=ALL
srun -p "${INTERNET_PARTITION}" -A "${ACCOUNT}" -N1 -n1 -c4 -t 00:30:00 --job-name=dcvlm_check \
    bash -c '
        source scripts/slurm/system_setup.sh train
        export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
        python -m internvl.train.data.check_release \
            --release_dir "${RELEASE_DIR}" \
            --max_samples "${MAX_SAMPLES}" \
            --tol "${TOL}"
    '
