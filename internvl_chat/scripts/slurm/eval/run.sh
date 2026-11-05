#!/bin/bash

# setup the software stack for training
# NOTE: this is machine specific; on Leonardo, the CUDA and GCC Compilers for training
# can be loaded as simple modules; on other machines one may need additional commands;
# to keep these details out of version control and as modular as possible, please create your own system_setup.sh
# (see the template system_setup.sh.template for details) 
echo Sourcing software stack from scripts/slurm/system_setup.sh eval
source scripts/slurm/system_setup.sh eval
echo Checking everything is in place
echo "Using python interpreter from $(which python)"
gcc --version
nvcc --version
sleep 3

# NOTE: patch to remove *_local.tsv files from the LMUData folder of VLMEvalKit
# --> they inject abspaths that do not transfer across machines
# find $LMUData -name "*_local.tsv" -delete

# variables for distributed run
export PYTHONPATH="${PYTHONPATH}:$(pwd)" # import internvl would fail otherwise

# list of datasets we extended / integrated from the official InternVL2.5 repo
LOCAL_DATASETS=(
  "MantisEval" 
  "MMVP" 
  "VLGuard" 
  "MLLMGuard" 
  "MMSafetyBench"
)

# list of datasets to execute with --mode infer
INFER_DATASETS=(
  "MMVet" 
  "MathVista_MINI" 
  "MathVision_MINI" 
  "MathVerse_MINI" 
  "WeMath" 
  "WildVision" 
  "LogicVista" 
  "DynaMath" 
  "CharXiv_descriptive_val" 
  "CharXiv_reasoning_val"
)

# list of text-only evals from opencompass
TEXT_DATASETS=(
  "c3"
  "ceval"
  "cmmlu"
  "gaokao"
  "gsm8k"
  "hellaswag"
  "humaneval"
  "math"
  "mbpp_cn"
  "mbpp"
  "mmlu"
  "nq"
  "race"
  "theoremqa"
  "triviaqa"
  "winogrande"
)

export DIST_TIMEOUT=7200

# give slurm some time to warmup
sleep 10

# submit with opencompass
case " ${TEXT_DATASETS[*]} " in
  *" $DATASET "*) 
  MODEL=$MODEL \
  GPUS=${GPUS_PER_NODE} \
  GPUS_PER_NODE=${GPUS_PER_NODE} \
  MAX_NEW_TOKENS=${MAX_NEW_TOKENS} \
  bash evaluate_text.sh ${CKPT} $DATASET
  exit 0
  ;;
esac


case " ${LOCAL_DATASETS[*]} " in
  *" $DATASET "*) 
  # submit with local scripts
  MODEL=$MODEL \
  GPUS=${GPUS_PER_NODE} \
  GPUS_PER_NODE=${GPUS_PER_NODE} \
  MAX_NEW_TOKENS=${MAX_NEW_TOKENS} \
  bash evaluate_dmt.sh ${CKPT} $DATASET 
  ;;
  
  *) 

  # set correct mode for vlmevalkit
  case " ${INFER_DATASETS[*]} " in
  *" $DATASET "*)
    MODE="infer"
    echo "$DATASET needs --mode infer"
    ;;
  *)
    MODE="all"
    ;;
  esac

  # submit with vlmevalkit
  torchrun \
    --standalone \
    --nproc_per_node=${GPUS} \
    dmt_evals/run.py \
    --data $DATASET \
    --model $MODEL \
    --load-dmt-pretrained ${CKPT_DIR}/${CKPT} \
    --work-dir ${EVAL_DIR} \
    --seed ${SEED} \
    --verbose \
    --mode $MODE \
    --overwrite 

  ;;
esac

