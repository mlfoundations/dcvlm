#!/bin/bash

# forces META_JSON to be set from cmd-line or avoids launching
set -u 

# save user-provided TIME before sourcing .env (so it can override .env)
USER_TIME="${TIME:-}"
# save user-provided QOS before sourcing .env (so it can override .env)
# needed e.g. for large jobs that require a different QOS than the default one
USER_QOS="${QOS:-}"

# source environment vars for this project
source .env

# check that the meta JSON exists before doing anything else
if [ ! -f "${META_DIR}/${META_JSON}" ]; then
    echo "Error: meta file not found: ${META_DIR}/${META_JSON}"
    exit 1
fi

# some cmd-line training params (other params are hard-coded in the trainer for now)
export SCALE=${SCALE:-None}
export TRAIN_STAGE=${TRAIN_STAGE:-3}
export MODEL_SIZE=${MODEL_SIZE:-1B}
export NODES=${NODES:-1}
export BATCH_SIZE=${BATCH_SIZE:-512}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
export PACKED_IMAGES=${PACKED_IMAGES:-48}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-16384}
export ESTIMATED_TOKENS=${ESTIMATED_TOKENS:-0}
export CALIBRATION_BATCHES=${CALIBRATION_BATCHES:-4096}
export MAX_PACKED_TOKENS=${MAX_PACKED_TOKENS:-8192}
export LR=${LR:-2e-5}
export WARMUP_RATIO=${WARMUP_RATIO:-0.03}
export MAX_STEPS=${MAX_STEPS:-0}
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-}
export LLM_INIT=${LLM_INIT:-base}
export JOB_NAME=${JOB_NAME:-test_${TRAIN_STAGE}-train-stage_${MODEL_SIZE}-model_${NODES}-nodes}
if [ $SCALE != "None" ]; then
    JOB_NAME="${JOB_NAME}_scale-${SCALE}"
    # check that per-device batch size is 1
    if [ $PER_DEVICE_BATCH_SIZE -ne 1 ]; then
        echo "When using SCALE, PER_DEVICE_BATCH_SIZE must be 1. Please set PER_DEVICE_BATCH_SIZE=1."
        exit 1
    fi
fi

# some SLURM configs (enabling debugging mode)
export DEBUG=${DEBUG:-0}
if [ "$DEBUG" -eq 1 ]; then
    echo "Enabling debugging mode (quick and high-priority access)."
    export JOB_NAME=DEBUG_${JOB_NAME}
    export ACCOUNT=${DEBUG_ACCOUNT}
    export PARTITION=${DEBUG_PARTITION}
    QOS=${DEBUG_QOS}
    TIME=${DEBUG_TIME}
else
    echo "Debugging not set. Will run with standard SLURM settings."
    TIME=${TIME:-"24:00:00"}
fi

# override TIME with user-provided value if set
if [ -n "$USER_TIME" ]; then
    TIME=$USER_TIME
fi

# override QOS with user-provided value if set
if [ -n "$USER_QOS" ]; then
    QOS=$USER_QOS
fi

# log current slurm setup
echo ACCOUNT: $ACCOUNT
echo PARTITION: $PARTITION
echo QOS: $QOS
echo TIME: $TIME
echo JOB_NAME: ${JOB_NAME}
sleep 3

# make sure slurm logs are visible
mkdir -p ${SLURM_LOGS_DIR}

# SCALE CONFIGS
if [[ "$SCALE" == "debug" ]]; then
    # set models
    export MODEL_SIZE="1B"
    # set training params
    export BATCH_SIZE=64
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=50000000 # 50M tokens

elif [[ "$SCALE" == "debug10x" ]]; then
    # set models
    export MODEL_SIZE="1B"
    # set training params
    export BATCH_SIZE=64
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=500000000 # 500M tokens

elif [[ "$SCALE" == "small" ]]; then
    # set models
    export MODEL_SIZE="1B"
    # set training params
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=6250000000 # 6.25B tokens

elif [[ "$SCALE" == "small_2bmodel" ]]; then
    # same hparams as "small" but with 2B model for small-scale 2B experiments
    export MODEL_SIZE="2B"
    # set training params (identical to "small")
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=6250000000 # 6.25B tokens

elif [[ "$SCALE" == "small_4bmodel" ]]; then
    # same hparams as "small" but with 4B model for small-scale 4B experiments
    export MODEL_SIZE="4B"
    # set training params (identical to "small")
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=6250000000 # 6.25B tokens

elif [[ "$SCALE" == "small2x" ]]; then
    # set models
    export MODEL_SIZE="1B"
    # set training params
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=12500000000 # 12.5B tokens

elif [[ "$SCALE" == "small2x_2bmodel" ]]; then
    # same hparams as "small2x" but with 2B model for small-scale-2x 2B experiments
    export MODEL_SIZE="2B"
    # set training params (identical to "small2x")
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=12500000000 # 12.5B tokens

elif [[ "$SCALE" == "small2x_4bmodel" ]]; then
    # same hparams as "small2x" but with 4B model for small-scale-2x 4B experiments
    export MODEL_SIZE="4B"
    # set training params (identical to "small2x")
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=12500000000 # 12.5B tokens

elif [[ "$SCALE" == "medium" ]]; then
    # set models
    export MODEL_SIZE="2B"
    # set training params
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=25000000000 # 25B tokens

elif [[ "$SCALE" == "medium_1bmodel" ]]; then
    # same hparams as "medium" but with 1B model for medium-scale 1B experiments
    export MODEL_SIZE="1B"
    # set training params
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=25000000000 # 25B tokens

elif [[ "$SCALE" == "medium_4bmodel" ]]; then
    # same hparams as "medium" but with 4B model for medium-scale 4B experiments
    export MODEL_SIZE="4B"
    # set training params
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=25000000000 # 25B tokens

elif [[ "$SCALE" == "large_1bmodel" ]]; then
    # same hparams as "large" but with 1B model for large-scale 1B experiments
    export MODEL_SIZE="1B"
    # set training params (identical to "large")
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=50000000000 # 50B tokens

elif [[ "$SCALE" == "large2x_1bmodel" ]]; then
    # 1B model at 100B tokens — second derisk pair (2x the 50B derisk budget)
    export MODEL_SIZE="1B"
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=100000000000 # 100B tokens

elif [[ "$SCALE" == "xlarge_1bmodel" ]]; then
    # 1B model at the 200B-token production budget (derisk-winning mix)
    export MODEL_SIZE="1B"
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=200000000000 # 200B tokens

elif [[ "$SCALE" == "large" ]]; then
    # set models
    export MODEL_SIZE="4B"
    # set training params
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=100000000000 # 100B tokens

elif [[ "$SCALE" == "xlarge" ]]; then
    # 8B model at the 200B-token x-large budget.
    # Identical hparams to "large" (bs=1024, lr=2e-5); only model size + tokens change.
    export MODEL_SIZE="8B"
    # set training params
    export BATCH_SIZE=1024
    export LR=2e-5
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=200000000000 # 200B tokens

# ---- SFT scale configs ----
# SFT token budget = 0.29 * pretraining token budget
# 6.25B PT -> 1.8125B SFT, 12.5B PT -> 3.625B SFT, 25B PT -> 7.25B SFT, 100B PT -> 29B SFT
# MODEL_SIZE is not set here — it's unused for SFT (MODEL_NAME_OR_PATH loads the full checkpoint)

elif [[ "$SCALE" == "sft_1_8b" ]]; then
    export BATCH_SIZE=1024
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=1812500000 # 0.29 * 6.25B

elif [[ "$SCALE" == "sft_3_6b" ]]; then
    export BATCH_SIZE=1024
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=3625000000 # 0.29 * 12.5B

elif [[ "$SCALE" == "sft_7_2b" ]]; then
    export BATCH_SIZE=1024
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=7250000000 # 0.29 * 25B

elif [[ "$SCALE" == "sft_29b" ]]; then
    export BATCH_SIZE=1024
    export WARMUP_RATIO=0.03
    export MAX_PACKED_TOKENS=8192
    export MAX_SEQ_LEN=8192
    export PACKED_IMAGES=24
    export TOT_TOKENS=29000000000 # 0.29 * 100B

elif [[ "$SCALE" == "None" ]]; then
    echo "No SCALE set, using command-line default parameters."
    if [[ "$MAX_STEPS" -le 0 ]]; then
        echo "When SCALE=None, MAX_STEPS must be set to a positive integer. Please set MAX_STEPS."
        exit 1
    fi
else 
    echo "Unknown scale: $SCALE"
    exit 1
fi

# Compute toks_per_batch
if [[ "$SCALE" != "None" ]]; then
    echo "SCALE is set to $SCALE, overriding MAX_STEPS and computing it based on TOT_TOKENS=$TOT_TOKENS"
    if [[ "$ESTIMATED_TOKENS" -gt 0 ]]; then
        TOKS_PER_BATCH=$((ESTIMATED_TOKENS * BATCH_SIZE))
    else
        TOKS_PER_BATCH=$((MAX_PACKED_TOKENS * BATCH_SIZE))
    fi
    echo "Estimated tokens per batch: $TOKS_PER_BATCH"
    export MAX_STEPS=$((TOT_TOKENS / TOKS_PER_BATCH))
    echo "Setting MAX_STEPS to $MAX_STEPS"
fi


# export model paths according to model size var
# (skipped when MODEL_NAME_OR_PATH is set, as the full checkpoint is loaded directly)
if [ -n "$MODEL_NAME_OR_PATH" ]; then
    # resolve relative paths against CKPT_DIR
    if [[ "$MODEL_NAME_OR_PATH" != /* ]]; then
        MODEL_NAME_OR_PATH="${CKPT_DIR}/${MODEL_NAME_OR_PATH}"
        export MODEL_NAME_OR_PATH
        echo "Resolved MODEL_NAME_OR_PATH to: $MODEL_NAME_OR_PATH"
    fi
    # check that the checkpoint exists on disk
    if [ ! -d "$MODEL_NAME_OR_PATH" ]; then
        echo "ERROR: MODEL_NAME_OR_PATH does not exist: $MODEL_NAME_OR_PATH"
        exit 1
    fi
    echo "MODEL_NAME_OR_PATH is set, skipping vision/llm model path resolution."
    export VISION_MODEL=""
    export LANGUAGE_MODEL=""
else
    export VISION_MODEL="${OFFLINE_MODEL_DIR}/InternViT-300M-448px-V2_5"
    if [ ${MODEL_SIZE} == "1B" ]; then
        export LANGUAGE_MODEL="${OFFLINE_MODEL_DIR}/Qwen2.5-0.5B"
    elif [ ${MODEL_SIZE} == "2B" ]; then
        export LANGUAGE_MODEL="${OFFLINE_MODEL_DIR}/Qwen2.5-1.5B"
    elif [ ${MODEL_SIZE} == "4B" ]; then
        export LANGUAGE_MODEL="${OFFLINE_MODEL_DIR}/Qwen2.5-3B"
    elif [ ${MODEL_SIZE} == "8B" ]; then
        export LANGUAGE_MODEL="${OFFLINE_MODEL_DIR}/Qwen2.5-7B"
    else
        echo "Unknown model size: ${MODEL_SIZE}. Exiting."
        exit 1
    fi

    # Apply LLM init variant (e.g., instruct models, alternate LM family)
    if [ "${LLM_INIT}" == "instruct" ]; then
        export LANGUAGE_MODEL="${LANGUAGE_MODEL}-Instruct"
    elif [ "${LLM_INIT}" == "internlm2_5" ]; then
        if [ ${MODEL_SIZE} == "2B" ]; then
            export LANGUAGE_MODEL="${OFFLINE_MODEL_DIR}/internlm2_5-1_8b-chat"
        else
            echo "LLM_INIT=internlm2_5 only supports MODEL_SIZE=2B (no matching InternLM2.5 ckpt for ${MODEL_SIZE})"
            exit 1
        fi
    elif [ "${LLM_INIT}" != "base" ]; then
        echo "Unknown LLM_INIT: ${LLM_INIT}. Supported: base, instruct, internlm2_5"
        exit 1
    fi

    # Verify the resolved language model path exists
    if [ ! -d "$LANGUAGE_MODEL" ]; then
        echo "ERROR: LANGUAGE_MODEL path does not exist: $LANGUAGE_MODEL"
        exit 1
    fi
fi

# export param freezing flags according to TRAIN_STAGE
if [ "${TRAIN_STAGE}" == "1" ]; then
    # this is called "TRAIN_STAGE 1" in the paper, and is the
    # MLP warm-up TRAIN_STAGE
    export FREEZE_LLM=True
    export FREEZE_MLP=False
    export FREEZE_BACKBONE=True
    export LR=2e-4
elif [ "${TRAIN_STAGE}" == "2" ]; then
    # this is called "TRAIN_STAGE 1.5" in the paper, and is the 
    # warm-up for the vision encoder
    export FREEZE_LLM=True
    export FREEZE_MLP=True
    export FREEZE_BACKBONE=False
    export LR=2e-4
elif [ "${TRAIN_STAGE}" == "3" ]; then
    # this is called "TRAIN_STAGE 2" in the paper, and is the
    # pretraining TRAIN_STAGE with all components trainable;
    # THIS IS THE DEFAULT TRAIN_STAGE
    export FREEZE_LLM=False
    export FREEZE_MLP=False
    export FREEZE_BACKBONE=False
    export LR=2e-5
elif [ "${TRAIN_STAGE}" == "sft" ]; then
    # sft TRAIN_STAGE, nothing really changes, it's the
    # meta-json that specifies the data that does most of the work
    export FREEZE_LLM=False
    export FREEZE_MLP=False
    export FREEZE_BACKBONE=False
    # SFT requires either SCALE=None (manual) or one of the 4 sft scales
    if [[ "$SCALE" != "None" && "$SCALE" != "sft_1_8b" && "$SCALE" != "sft_3_6b" && "$SCALE" != "sft_7_2b" && "$SCALE" != "sft_29b" ]]; then
        echo "SFT requires SCALE=None or one of: sft_1_8b, sft_3_6b, sft_7_2b, sft_29b. Got SCALE=$SCALE"
        exit 1
    fi
    # SFT requires a pretrained checkpoint path
    if [ -z "$MODEL_NAME_OR_PATH" ]; then
        echo "SFT requires MODEL_NAME_OR_PATH to be set (path to pretrained checkpoint)."
        exit 1
    fi
else
    echo "Unknown TRAIN_STAGE: ${TRAIN_STAGE}. Exiting."
    exit 1
fi

# submit the job
output=$(sbatch \
    --account=${ACCOUNT} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --nodes=${NODES} \
    --ntasks-per-node=1 \
    --gpus-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_NODE} \
    --mem=${MEMORY} \
    --job-name=${JOB_NAME} \
    --time=${TIME} \
    --output=${SLURM_LOGS_DIR}/${JOB_NAME}.out \
    --error=${SLURM_LOGS_DIR}/${JOB_NAME}.err \
    --wait-all-nodes=1 \
    --dependency=singleton \
    --exclude=${EXCLUDE_NODES} \
    --mail-type=ALL \
    --mail-user=${EMAIL} \
    scripts/slurm/train/run.sh)

# log and exit
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo Submitted job: $JOB_NAME
    echo Monitor the output file with: tail -f ${SLURM_LOGS_DIR}/${JOB_NAME}.out
    echo Monitor the error file with: tail -f ${SLURM_LOGS_DIR}/${JOB_NAME}.err
else
    echo "Job submission failed with exit code $EXIT_CODE."
    exit $EXIT_CODE
fi


