#!/bin/bash
# Submits N chained WanJuan1.0 download jobs (each resumes after the previous one).
# Usage: URL_LIST=/path/to/part-XXXX.jsonl.tar.gz OUTPUT_FOLDER=/path/to/wanjuan/shards ./download_multi_wanjuan.sh
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../../.env  # ACCOUNT, INTERNET_PARTITION

: "${URL_LIST:?}" "${OUTPUT_FOLDER:?}"
export URL_LIST OUTPUT_FOLDER
N=30

LOGDIR=slurm_logs/download_wanjuan
mkdir -p $LOGDIR
echo "Logs will be saved in $LOGDIR"

jobid=""
for i in $(seq 1 $N); do
    dependency=""
    [ -n "$jobid" ] && dependency="--dependency=afterany:$jobid"
    jobid=$(sbatch --account=$ACCOUNT \
        --parsable \
        --job-name=download_wanjuan_$i \
        --partition=$INTERNET_PARTITION \
        --ntasks=1 \
        --mem=30G \
        --cpus-per-task=8 \
        --time=04:00:00 \
        $dependency \
        --output=$LOGDIR/download_wanjuan_$i.log \
        --error=$LOGDIR/download_wanjuan_$i.err download_wanjuan.sh)
done
