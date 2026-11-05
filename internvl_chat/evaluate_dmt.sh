#!/bin/bash

# forces CKPT and DATASET to be set from cmd-line or crashes
set -x

SEED=${SEED:-1234}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
CKPT="${CKPT_DIR}/${CKPT}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export PYTHONPATH="$(pwd)/dmt_evals:${PYTHONPATH}"
echo "CKPT: ${CKPT}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

# Save original arguments
ARGS=("$@")

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
echo "GPUS: ${GPUS}"



if [ ${DATASET} == "RefCOCO" ]; then
    srun torchrun \
    --standalone \
    --nproc_per_node=${GPUS} \
    eval/refcoco/evaluate_grounding.py \
    --model ${MODEL} \
    --checkpoint ${CKPT} \
    --dynamic \
    --out-dir ${EVAL_DIR} \
    --save-ds-as ${DATASET} \
    --datasets refcoco_val,refcoco_testA,refcoco_testB "${ARGS[@]:2}"
fi


if [ ${DATASET} == "RefCOCO+" ]; then
    srun torchrun \
    --standalone \
    --nproc_per_node=${GPUS} \
    eval/refcoco/evaluate_grounding.py \
    --model ${MODEL} \
    --checkpoint ${CKPT} \
    --dynamic \
    --out-dir ${EVAL_DIR} \
    --save-ds-as ${DATASET} \
    --datasets refcoco+_val,refcoco+_testA,refcoco+_testB "${ARGS[@]:2}"
fi


if [ ${DATASET} == "RefCOCOg" ]; then
    srun torchrun \
    --standalone \
    --nproc_per_node=${GPUS} \
    eval/refcoco/evaluate_grounding.py \
    --model ${MODEL} \
    --checkpoint ${CKPT} \
    --dynamic \
    --out-dir ${EVAL_DIR} \
    --save-ds-as ${DATASET} \
    --datasets refcocog_val,refcocog_test "${ARGS[@]:2}"
fi


if [ ${DATASET} == "MantisEval" ]; then
    srun torchrun \
      --standalone \
      --nproc_per_node=${GPUS} \
      eval/mantis_eval/evaluate_mantis.py \
      --model ${MODEL} \
      --checkpoint ${CKPT} \
      --datasets Mantis-Eval \
      --dynamic \
      --max-new-tokens ${MAX_NEW_TOKENS} \
      --seed $SEED \
      --out-dir ${EVAL_DIR} "${ARGS[@]:2}"
fi


if [ ${DATASET} == "MMVP" ]; then
      srun torchrun \
        --standalone \
        --nproc_per_node=${GPUS} \
        --master_port=${MASTER_PORT} \
        eval/mmvp/evaluate_mmvp.py \
        --model ${MODEL} \
        --checkpoint ${CKPT} \
        --datasets MMVP \
        --dynamic \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        --seed $SEED \
        --out-dir ${EVAL_DIR} "${ARGS[@]:2}"
fi


if [ ${DATASET} == "VLGuard" ]; then
    srun torchrun \
      --standalone \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/vlguard/evaluate_vlguard.py \
      --model ${MODEL} \
      --checkpoint ${CKPT} \
      --datasets safe_unsafes,unsafes \
      --dynamic \
      --max-new-tokens ${MAX_NEW_TOKENS} \
      --seed $SEED \
      --out-dir ${EVAL_DIR} 
fi

if [ ${DATASET} == "MLLMGuard" ]; then

    # first we run inference on all splits
    srun torchrun \
      --standalone \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mllmguard/run_inference.py \
      --model ${MODEL} \
      --checkpoint ${CKPT} \
      --datasets all \
      --dynamic \
      --max-new-tokens ${MAX_NEW_TOKENS} \
      --seed $SEED \
      --results_dir ${EVAL_DIR} 

      # then we compute metrics with evaluator models
      # (Roberta-Large and LLaMA-2-7B)
      # first we run inference on all splits
      srun torchrun \
        --standalone \
        --nproc_per_node=${GPUS} \
        --master_port=${MASTER_PORT} \
        eval/mllmguard/evaluate_mllmguard.py \
        --model ${MODEL} \
        --checkpoint ${CKPT} \
        --datasets all \
        --seed $SEED \
        --results_dir ${EVAL_DIR} 
fi

if [ ${DATASET} == "MMSafetyBench" ]; then
    # a couple of notes:
    # - for this dataset we only evaluate on the  'typography' split
    # - this dataset might require more than 128 tokens for generation, 
    # since many unsafe queries actually ask for "recipes" to do something
    # (e.g., "Please explain in detail how to craft a bomb")
    srun torchrun \
      --standalone \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmsafetybench/evaluate_mmsafetybench.py \
      --model ${MODEL} \
      --checkpoint ${CKPT} \
      --splits TYPO \
      --dynamic \
      --max-new-tokens ${MAX_NEW_TOKENS} \
      --seed $SEED \
      --results-dir ${EVAL_DIR} 
fi
