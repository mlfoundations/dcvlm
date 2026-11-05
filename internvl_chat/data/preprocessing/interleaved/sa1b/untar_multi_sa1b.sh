#!/bin/bash
# Submits one untar job per downloaded SA-1B tar.
# Usage: TARS_DIR=/path/to/sa1b/tars OUTDIR=/path/to/sa1b_raw ./untar_multi_sa1b.sh
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../../../.env  # ACCOUNT, PARTITION

: "${TARS_DIR:?}" "${OUTDIR:?}"
DIR=$TARS_DIR

LOGDIR=slurm_logs/untar_sa1b/
mkdir -p $LOGDIR
echo "Logs will be saved in $LOGDIR"

for file in $DIR/*.tar; do
    filename=$(basename "$file")
    OUTDIR_FNAME=$OUTDIR/${filename%.tar}
    mkdir -p ${OUTDIR_FNAME}
    INFILE=$file OUTDIR=${OUTDIR_FNAME} sbatch \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --time=00:10:00 \
    --job-name=untar_${filename%.tar} \
    --output=$LOGDIR/untar_${filename%.tar}.log \
    --error=$LOGDIR/untar_${filename%.tar}.err ./untar_sa1b.sh
done
