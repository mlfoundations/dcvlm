#!/bin/bash

# typical commands to set stuff up
source .env 
source scripts/slurm/system_setup.sh eval
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)" # import internvl would fail otherwise
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/dmt_evals" # code for tagging ckpts is here
export OPENAI_API_KEY=${OPENAI_API_KEY}

echo Submitting API Evals for $1

for DS in $1; do
    # then compute metrics using GPT-4o-mini as a judge 
    # NOTE: we might use a lighter one model
    if [ ${DS} == "MMSafetyBench" ]; then
        python eval/mmsafetybench/calculate_score.py \
            --model ${MODEL} \
            --checkpoint ${CKPT_DIR}/${CKPT} \
            --seed $SEED \
            --splits TYPO \
            --scenarios all \
            --results-dir ${EVAL_DIR}
    else 
        # launch a single process eval (--mode eval) with no gpus, will only perform
        # answer-extraction + llm-scoring if needed 
        # NOTE: not passing --overwrite to avoid duplicate API calls if compute nodes have Internet connection
        echo Launching API Eval for $DS
        python -u dmt_evals/run.py \
            --data ${DS} \
            --model ${MODEL} \
            --load-dmt-pretrained ${CKPT_DIR}/${CKPT} \
            --work-dir ${EVAL_DIR}/ \
            --verbose \
            --mode eval
    fi
done
