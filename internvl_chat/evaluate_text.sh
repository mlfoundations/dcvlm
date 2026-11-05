#!/bin/bash

# forces MODEL, CKPT and DATASET to be set from cmd-line or crashes
set -x

# setup devices
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

# log meaningful vars
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CKPT: ${CKPT}"

# Make sure the environment for text evals is loaded
source scripts/slurm/system_setup.sh text
echo Using python bin from $(which python)

# Move to the text_evals folder
cd text_evals

# Generate the experiment config on the fly
python make_config.py \
  --model ${MODEL} \
  --ckpt ${CKPT} \
  --dataset ${DATASET}

# ---> Now the config will automatically be located at
# configs/$MODEL/$CKPT/$DATASET_config.py
CONFIG="configs/${MODEL}/${CKPT}/${DATASET}_config.py"

# Launch distributed eval with opencompass
# FIXME: for some reason, opencompass sometimes fails silently when running 
# without --debug mode. This flag will force 1 GPU only, and it's not ideal
python opencompass/run.py $CONFIG \
  --max-num-workers $GPUS \
  --seed $SEED \
  --debug \
  -w ${EVAL_DIR}/seed_${SEED}/text_evals/$MODEL/$CKPT/$DATASET

# Cleanup: some unneeded folders like tmp and icl_inference_outputs spawn
# out of nowhere
# rmdir ./icl_inference_outputs
# rm -rf ./tmp