#!/bin/bash
# Submits N chained MINT-1T-HTML download jobs (each resumes after the previous one).
# Usage: PARQUET_ROOT=/path/to/parquets OUTPUT_DIR=/path/to/mint START=0 END=200 ./download_multi_mint.sh
set -x
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../../.env  # ACCOUNT, INTERNET_PARTITION

: "${PARQUET_ROOT:?}" "${OUTPUT_DIR:?}" "${START:?}" "${END:?}"
export PARQUET_ROOT OUTPUT_DIR START END
N=30

LOGDIR=slurm_logs/download_mint
mkdir -p $LOGDIR
echo "Logs will be saved in $LOGDIR"

jobid=""
for i in $(seq 1 $N); do
    dependency=""
    [ -n "$jobid" ] && dependency="--dependency=afterany:$jobid"
    jobid=$(sbatch --account=$ACCOUNT \
        --parsable \
        --job-name=download_mint_$i \
        --partition=$INTERNET_PARTITION \
        --ntasks=4 \
        --cpus-per-task=2 \
        --time=04:00:00 \
        $dependency \
        --output=$LOGDIR/download_mint_$i.log \
        --error=$LOGDIR/download_mint_$i.err download_mint.sh)
done
