#!/bin/bash
# Submits N chained GRIT-20M download jobs (each resumes after the previous one).
# Usage: PARQUETS_DIR=/path/to/grit/parquets/grit-20m SHARDS_DIR=/path/to/grit/shards ./download_multi_grit_caption.sh
set -x
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../../.env  # ACCOUNT, INTERNET_PARTITION

: "${PARQUETS_DIR:?}" "${SHARDS_DIR:?}"
export PARQUETS_DIR SHARDS_DIR
N=30

LOGDIR=slurm_logs/download_gritcaption
mkdir -p $LOGDIR
echo "Logs will be saved in $LOGDIR"

jobid=""
for i in $(seq 1 $N); do
    dependency=""
    [ -n "$jobid" ] && dependency="--dependency=afterany:$jobid"
    jobid=$(sbatch --account=$ACCOUNT \
        --parsable \
        --job-name=download_gritcaption_$i \
        --partition=$INTERNET_PARTITION \
        --ntasks=4 \
        --cpus-per-task=2 \
        --time=04:00:00 \
        $dependency \
        --output=$LOGDIR/download_gritcaption_$i.log \
        --error=$LOGDIR/download_gritcaption_$i.err download_grit_caption_shards.sh)
done
