#!/bin/bash
# Submits N chained SA-1B download jobs (each skips tars that already exist).
# Usage: LINKS_FILE=/path/to/sa1b_links.tsv OUTPUT_DIR=/path/to/sa1b/tars ./download_multi_sa1b.sh
set -x
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../../../.env  # ACCOUNT, INTERNET_PARTITION

: "${LINKS_FILE:?}" "${OUTPUT_DIR:?}"
export LINKS_FILE OUTPUT_DIR
N=30

LOGDIR=slurm_logs/download_sa1b
mkdir -p $LOGDIR
echo "Logs will be saved in $LOGDIR"

jobid=""
for i in $(seq 1 $N); do
    dependency=""
    [ -n "$jobid" ] && dependency="--dependency=afterany:$jobid"
    jobid=$(sbatch --account=$ACCOUNT \
        --parsable \
        --job-name=download_sa1b_$i \
        --partition=$INTERNET_PARTITION \
        --ntasks=4 \
        --cpus-per-task=2 \
        --time=04:00:00 \
        $dependency \
        --output=$LOGDIR/download_sa1b_$i.log \
        --error=$LOGDIR/download_sa1b_$i.err download_sa1b.sh)
done
