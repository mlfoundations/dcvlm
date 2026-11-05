#!/bin/bash

# run within a container if an image is specified
if [ -z "$IMAGE" ]; then
    APPTAINER_ARGS=""
    echo "No container image specified. Running directly."
else
    APPTAINER_ARGS="
    apptainer exec \
    --nv \
    $IMAGE 
    "
fi

source .env
# FIXME: temporarily setting this var, but should be set by slurm automatically
SLURM_GPUS_PER_NODE=4
BATCH_SIZE=4
PER_DEVICE_BATCH_SIZE=1

# setup the software stack for training
# NOTE: this is machine specific; on Leonardo, the CUDA and GCC Compilers for training
# can be loaded as simple modules; on other machines one may need additional commands;
# to keep these details out of version control and as modular as possible, please create your own system_setup.sh
# (see the template system_setup.sh.template for details) 
echo Sourcing software stack from scripts/slurm/system_setup.sh train
source scripts/slurm/system_setup.sh train
echo Checking everything is in place
echo "Using python interpreter from $(which python)"
gcc --version
nvcc --version
sleep 3

# setup checkpointing & logging 
export WANDB_PROJECT="dcvlm-train"
export RUN_NAME=$JOB_NAME # export is needed by the code
OUTPUT_DIR=${CKPT_DIR}/${RUN_NAME}
mkdir -p $OUTPUT_DIR

# variables for distributed run
export LAUNCHER=pytorch # this is needed by the InternVL repo
export PYTHONPATH="${PYTHONPATH}:$(pwd)" # import internvl would fail otherwise

# some constants needed by torchrun
export TF_CPP_MIN_LOG_LEVEL=3

# NETWORKING
export NCCL_NET_GDR_LEVEL=PIX # Use GPU Direct RDMA when GPU and NIC are on the same PCI switch
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_TIMEOUT=120
# export NCCL_DEBUG=INFO

# setup infiniband on JSC
HOSTNAME=$(hostname)
echo "Running on host: $HOSTNAME"
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
if [[ $HOSTNAME == *"jr"* ]]; then
    export MASTER_ADDR="${MASTER_ADDR}i"
    export NCCL_ALGO="Ring"
elif [[ $HOSTNAME == *"jw"* ]]; then
    export MASTER_ADDR="${MASTER_ADDR}i"
    export NCCL_ALGO="Ring"
else
    export MASTER_ADDR="${MASTER_ADDR}"
fi

# TORCHRUN SETTINGS
export WORLD_SIZE=$((SLURM_GPUS_PER_NODE*SLURM_JOB_NUM_NODES))
export GRADIENT_ACC=$((BATCH_SIZE/PER_DEVICE_BATCH_SIZE/WORLD_SIZE))
export MASTER_PORT=20043

# HUGGINGFACE 
export HF_HUB_OFFLINE=1
export HF_HUB_VERBOSITY="info"

# LOGS
# export LOGLEVEL=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL


# MISC
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_PER_NODE - 1)))
DEEPSPEED_FILE=zero_stage1_config.json
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# log to make sure everything is set correctly
echo BATCH_SIZE: $BATCH_SIZE
echo PER_DEVICE_BATCH_SIZE: $PER_DEVICE_BATCH_SIZE
echo GRADIENT_ACC: $GRADIENT_ACC
echo SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
echo RUN_NAME: $RUN_NAME
echo OUTPUT_DIR: $OUTPUT_DIR


VISION_MODEL="${OFFLINE_MODEL_DIR}/InternViT-300M-448px-V2_5"
LANGUAGE_MODEL="${OFFLINE_MODEL_DIR}/Qwen2.5-0.5B"
FREEZE_LLM=False
FREEZE_MLP=False
FREEZE_BACKBONE=False
PACKED_IMAGES=24
MAX_SEQ_LEN=8192
ESTIMATED_TOKENS=0
LR=2e-5

# ARGS FOR TRAINING
TRAINING_ARGS="--vision_path "${VISION_MODEL}" \
    --llm_path "${LANGUAGE_MODEL}" \
    --model_root "${OFFLINE_MODEL_DIR}" \
    --conv_style "internvl2_5" \
    --use_fast_tokenizer False \
    --output_dir ${OUTPUT_DIR} \
    --meta_path ${META_DIR}/${META_JSON} \
    --overwrite_output_dir False \
    --force_image_size 448 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.0 \
    --min_num_frame 8 \
    --max_num_frame 32 \
    --scale debug \
    --freeze_llm ${FREEZE_LLM} \
    --freeze_mlp ${FREEZE_MLP} \
    --freeze_backbone ${FREEZE_BACKBONE} \
    --vision_select_layer -1 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers False \
    --ignore_data_skip True \
    --seed 40 \
    --bf16 True \
    --max_steps 1250 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.5 \
    --save_total_limit 5 \
    --learning_rate ${LR} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --run_name ${RUN_NAME} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --do_train False \
    --grad_checkpoint True \
    --group_by_length False \
    --dynamic_image_size True \
    --use_thumbnail True \
    --ps_version 'v2' \
    --deepspeed ${DEEPSPEED_FILE} \
    --use_packed_ds True \
    --lazy_packing True \
    --lazy_cleanup_freq 999999 \
    --lazy_allow_distributed_replacement True \
    --num_images_expected ${PACKED_IMAGES} \
    --max_packed_tokens ${MAX_SEQ_LEN} \
    --estimated_tokens ${ESTIMATED_TOKENS} \
    --max_buffer_size 20 \
    --log_freq 1000 \
    --strict_mode False \
    --replacement True \
    --allow_overflow False \
    --remove_unused_columns False \
    --loss_reduction "square" \
    --loss_reduction_all_gather False 
    "

# log the training args
echo "Training arguments: $TRAINING_ARGS"

# MAIN COMMAND ARGS
LAUNCHER_CMD="python -m torch.distributed.run \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

# RUN THE MAIN COMMAND
bash -c "$LAUNCHER_CMD internvl/train/internvl_chat_pretrain.py $TRAINING_ARGS"