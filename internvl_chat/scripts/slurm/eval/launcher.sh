#!/bin/bash

# forces MODEL, DATASET and CKPT to be set from cmd-line or crashes 
set -u

# source the .env file for slurm, hw and system configs
source .env

# default 1234 seed (as per VLMEvalKit) if not provided
export SEED=${SEED:-1234}
export SUBMIT_API=${SUBMIT_API:-0}
export API_ONLY=${API_ONLY:-0}

# create the name of the run
CKPT_NOSLASH="${CKPT//\//-}"
DATASET_NAMES_NO_WHITESPACE="${DATASETS// /-}"
export RUN_NAME=${DATASET_NAMES_NO_WHITESPACE}_${MODEL}_${CKPT_NOSLASH}

# some SLURM configs (enabling debugging mode)
export DEBUG=${DEBUG:-0}
if [ "$DEBUG" -eq 1 ]; then
    echo "Enabling debugging mode (quick and high-priority access)."
    export RUN_NAME=DEBUG_${RUN_NAME}
    export ACCOUNT=${DEBUG_ACCOUNT}
    export PARTITION=${DEBUG_PARTITION}
    QOS=${DEBUG_QOS}
    TIME=${DEBUG_TIME}
else
    echo "Debugging not set. Will run with standard SLURM settings."
    TIME=${TIME:-"24:00:00"}
fi

# log current slurm setup
echo ACCOUNT: $ACCOUNT
echo PARTITION: $PARTITION
echo QOS: $QOS
echo TIME: $TIME
echo RUN_NAME: ${RUN_NAME}

# system config (machine specific)
export GPUS=${GPUS:-${GPUS_PER_NODE}}
export CPUS=${CPUS:-${CPUS_PER_NODE}}
export OMP_NUM_THREADS=$((CPUS/GPUS)) # by default we enable core-wise threading
mkdir -p ${SLURM_LOGS_DIR}/zs_evals/

# a constant list of datasets that do not need api calls
NO_APIEVAL=(
    "gsm8k" "theoremqa" "math" 
    "c3" "ceval" "cmmlu" 
    "gaokao" "hellaswag" "mmlu" 
    "nq" "race" "triviaqa" 
    "winogrande" "MMVP" "MantisEval" 
    "MLLMGuard" "VLGuard"
)
# list of text evals s.t. we can submit on 1gpu only if slurm allows so
# (opencompass often fails silently in distributed mode)
TEXT_ONLY_EVALS=(
    "gsm8k" "theoremqa" "math" 
    "c3" "ceval" "cmmlu" 
    "gaokao" "hellaswag" "mmlu" 
    "nq" "race" "triviaqa" 
)

# submit our distributed evals =) (cannot distribute on > 1 nodes for now)
DEPENDENCY_FLAG="--dependency=afterok"
for DATASET in $DATASETS; do
    # check if it's a text only dataset
    textonly=false
    for item in "${TEXT_ONLY_EVALS[@]}"; do
        if [[ "$item" == "$DATASET" ]]; then
            textonly=true
            break
        fi
    done
    # quick edit of hw-related vars for opencompass evals
    if $textonly; then
        GPUS_PER_RUN=1
        CPUS_PER_RUN=$((CPUS_PER_NODE/GPUS))
        MEMORY_PER_RUN=80G
    else
        GPUS_PER_RUN=$GPUS
        CPUS_PER_RUN=$CPUS_PER_NODE
        MEMORY_PER_RUN=$MEMORY
    fi
    echo "Dataset=$DATASET, GPUS_PER_RUN=${GPUS_PER_RUN}, CPUS_PER_RUN=${CPUS_PER_RUN}, MEMORY_PER_RUN=${MEMORY_PER_RUN}"
    export DATASET=$DATASET
    export INFER_RUN_NAME=${DATASET}_${MODEL}_${CKPT_NOSLASH}
    if [ $API_ONLY -eq 0 ]; then
        sleep 1
        jobid=$(sbatch \
            --account=${ACCOUNT} \
            --partition=${PARTITION} \
            --time=${TIME} \
            --qos=${QOS} \
            --ntasks-per-node=1 \
            --nodes=1 \
            --job-name=${INFER_RUN_NAME}_seed${SEED} \
            --gpus-per-node=${GPUS_PER_RUN} \
            --cpus-per-task=${CPUS_PER_RUN} \
            --mem=${MEMORY_PER_RUN} \
            --output=${SLURM_LOGS_DIR}/zs_evals/${MODEL}/${CKPT_NOSLASH}/${DATASET}_seed${SEED}.out \
            --error=${SLURM_LOGS_DIR}/zs_evals/${MODEL}/${CKPT_NOSLASH}/${DATASET}_seed${SEED}.err \
            --mail-type=FAIL \
            --mail-user=${EMAIL} \
            $(pwd)/scripts/slurm/eval/run.sh | awk '{print $4}')
        
        echo Submitted eval job: ${INFER_RUN_NAME}
        echo Inspect the output with: tail -f ${SLURM_LOGS_DIR}/zs_evals/${MODEL}/${CKPT_NOSLASH}/${DATASET}_seed${SEED}.out
        echo Inspect the error with: tail -f ${SLURM_LOGS_DIR}/zs_evals/${MODEL}/${CKPT_NOSLASH}/${DATASET}_seed${SEED}.err    
        echo ""
        in_list=false
        for item in "${NO_APIEVAL[@]}"; do
            if [[ "$item" == "$DATASET" ]]; then
                in_list=true
                break
            fi
        done
        
        if ${in_list}; then
            echo "Benchmark is $DATASET, omitting job dependency."
        else
            DEPENDENCY_FLAG="${DEPENDENCY_FLAG}:$jobid"
        fi
    else
        echo "API-only mode is enabled (API_ONLY=$API_ONLY), skipping eval job submission."
        DEPENDENCY_FLAG=""
    fi
done

# log and exit
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    # among submitted ones, collect benchmarks
    # that need api calls
    API_DATASETS=()
    for DATASET in $DATASETS; do
        in_list=false
        for item in "${NO_APIEVAL[@]}"; do
            if [[ "$item" == "$DATASET" ]]; then
                in_list=true
                break
            fi
        done
        
        if ${in_list}; then
            echo "Benchmark is $DATASET, so skipping eval on ${INTERNET_PARTITION}"
        else
            API_DATASETS+=($DATASET)
        fi
    done
    # FOLLOW-UP CMD TO USE API CALLS FOR LLMs:
    # 1 - this will use LLM-based answer extraction
    # 2 - as well as LLM-as-a-judge for benchmarks that need it
    # NOTE: we have to filter out benchmarks that are not from VLMEvalKit
    API_DATASETS="${API_DATASETS[*]}" # creates whitespace separated string
    if [[ ${SUBMIT_API} -eq 1 && -n "${API_DATASETS}" ]]; then
        mkdir -p ${SLURM_LOGS_DIR}/zs_evals/${MODEL}/${CKPT_NOSLASH}/
        DATE_RANDOM="$(date -I)_$RANDOM"
        sbatch \
            --account=${ACCOUNT} \
            --chdir=${DMT_ROOT}/internvl_chat \
            --job-name=APIEVAL_${MODEL}_${CKPT_NOSLASH}_seed${SEED} \
            $DEPENDENCY_FLAG \
            --partition=${INTERNET_PARTITION} \
            --mem=${INTERNET_PARTITION_MEMORY} \
            --time=${INTERNET_PARTITION_TIME} \
            --ntasks=1 \
            --cpus-per-task=${INTERNET_PARTITION_CPUS} \
            --output=${SLURM_LOGS_DIR}/zs_evals/${MODEL}/${CKPT_NOSLASH}/APIEVAL_${DATE_RANDOM}_seed${SEED}.out \
            --error=${SLURM_LOGS_DIR}/zs_evals/${MODEL}/${CKPT_NOSLASH}/APIEVAL_${DATE_RANDOM}_seed${SEED}.err \
            --mail-type=FAIL \
            --mail-user=${EMAIL} \
            scripts/slurm/eval/api.sh "${API_DATASETS}"
        
        echo "Submitted API Eval for ${API_DATASETS}"
        echo "Monitor the output logs with tail -f ${SLURM_LOGS_DIR}/zs_evals/${MODEL}/${CKPT_NOSLASH}/APIEVAL_${DATE_RANDOM}_seed${SEED}.out"
        echo "Monitor the error logs with tail -f ${SLURM_LOGS_DIR}/zs_evals/${MODEL}/${CKPT_NOSLASH}/APIEVAL_${DATE_RANDOM}_seed${SEED}.err"
        echo ""
    else
        if [ ${SUBMIT_API} -eq 0 ]; then
            echo "Submit API flag is set to $SUBMIT_API, so skipping eval on $INTERNET_PARTITION for $API_DATASETS"
        fi
        if [ ! -n "${API_DATASETS}" ]; then
            echo "API_DATASETS is empty, not submitting anything to ${INTERNET_PARTITION}"
        fi
    fi

else
    echo "Job submission failed with exit code $EXIT_CODE."
    exit $EXIT_CODE
fi
