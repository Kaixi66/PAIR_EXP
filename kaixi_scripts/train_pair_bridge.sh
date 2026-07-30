#!/usr/bin/env bash
set -euo pipefail

# PAIR bridge training launcher. Override values at launch, for example:
#   MAX_STEPS=2 BATCH_SIZE=1 GPUS=0 WANDB_MODE=disabled ./train_pair_bridge.sh

#########################
# User-facing settings
#########################

DATASET_NAME="${DATASET_NAME:-libero_object_no_noops}"
GPUS="${GPUS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-64}"
MAX_STEPS="${MAX_STEPS:-50005}"
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-50000}"

LR_SCHEDULER="${LR_SCHEDULER:-multistep}" # cosine  multistep  constant
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"

SAVE_FREQ="${SAVE_FREQ:-50000}"
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-False}"
USE_PRO_VERSION="${USE_PRO_VERSION:-False}"

PAIR_ALIGN_WEIGHT="${PAIR_ALIGN_WEIGHT:-0.1}"
PAIR_BRIDGE_DIM="${PAIR_BRIDGE_DIM:-1024}"
PAIR_GATE_MLP_DIM="${PAIR_GATE_MLP_DIM:-256}"
PAIR_GATE_NUM_LAYERS="${PAIR_GATE_NUM_LAYERS:-1}"

PAIR_INIT_GATE_MODE="${PAIR_INIT_GATE_MODE:-learnable}" # learnable  fixed
PAIR_INIT_GATE_VALUE="${PAIR_INIT_GATE_VALUE:-0}"
PAIR_INIT_GATE_GRANULARITY="${PAIR_INIT_GATE_GRANULARITY:-per_step}"
PAIR_GATE_ACTIVATION="${PAIR_GATE_ACTIVATION:-tanh}"
PAIR_INJECTION_POSITIONS="${PAIR_INJECTION_POSITIONS:-start}"
PAIR_LOG_DEBUG_METRICS="${PAIR_LOG_DEBUG_METRICS:-True}"

PAIR_ACTION_AE_ENCODER_PATH="${PAIR_ACTION_AE_ENCODER_PATH:-/umd-datapool/kaixi/PAIR/action_ae_runs/conditionedAE_en2_de1_object/encoder_50000-steps.pt}"
# /umd-datapool/kaixi/PAIR/action_ae_runs/conditionedAE_en2_de1/encoder_9000-steps.pt
# /umd-datapool/kaixi/PAIR/action_ae_runs/conditionedAE_en2_de1_spatial/encoder_50000-steps.pt

WANDB_ENTITY="${WANDB_ENTITY:-kaixi-university-of-maryland}"
WANDB_PROJECT="${WANDB_PROJECT:-PAIR}"
export WANDB_MODE="${WANDB_MODE:-online}"
EXP_NAME="${EXP_NAME:-l20_PAIR_v8_learnable0_specialize_object_acc}"

DRY_RUN="${DRY_RUN:-false}"
BACKGROUND="${BACKGROUND:-false}"

#################
# Internal wiring
#################

CONDA_ENV="${CONDA_ENV:-vla-adapter}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-true}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"

PAIR_ROOT="${PAIR_ROOT:-/data/kaixi/PAIR}"
ENV_SH="${ENV_SH:-${PAIR_ROOT}/kaixi_scripts/env.sh}"
REPO_DIR="${REPO_DIR:-${PAIR_ROOT}/VLA-Adapter}"
ACTION_AE_DIR="${ACTION_AE_DIR:-${PAIR_ROOT}/action_ae}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/data/kaixi/dataset}"

VLM_PATH="${VLM_PATH:-pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-pretrained_models/configs}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/umd-datapool/kaixi/PAIR/checkpoints}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"

USE_FILM="${USE_FILM:-False}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
USE_PROPRIO="${USE_PROPRIO:-True}"
USE_LORA="${USE_LORA:-True}"
USE_FZ="${USE_FZ:-False}"
USE_MINIVLM="${USE_MINIVLM:-True}"
IMAGE_AUG="${IMAGE_AUG:-True}"
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-True}"
USE_VAL_SET="${USE_VAL_SET:-False}"
VAL_FREQ="${VAL_FREQ:-10000}"
VAL_TIME_LIMIT="${VAL_TIME_LIMIT:-180}"
WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

############################
# Derived settings
############################

current_time="${CURRENT_TIME:-$(date +%Y%m%d_%H%M%S)}"
gpu_list="${GPUS// /}"
IFS=',' read -r -a gpu_array <<< "${gpu_list}"
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    NPROC_PER_NODE="${#gpu_array[@]}"
fi

if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="pair_bridge_spatial_g${NPROC_PER_NODE}_b${BATCH_SIZE}_s${MAX_STEPS}_${current_time}"
fi

if [[ -z "${DATA_ROOT_DIR:-}" ]]; then
    DATA_ROOT_DIR="${LOCAL_DATA_ROOT}/libero"
fi

log_file="${LOG_FILE:-${LOG_DIR}/PAIR-Bridge--${DATASET_NAME}--${current_time}.log}"

############################
# Environment
############################

source "${ENV_SH}"

if [[ "${ACTIVATE_CONDA}" == "true" ]]; then
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV}"
fi

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}" "${RUN_ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ACTION_AE_DIR}:${REPO_DIR}:${PAIR_LIBERO_DIR}:${PAIR_CALVIN_ROOT}/calvin_env:${PAIR_CALVIN_ROOT}/calvin_models${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-/data/miniconda3/envs/${CONDA_ENV}/bin/python}"
torchrun_bin="${TORCHRUN_BIN:-/data/miniconda3/envs/${CONDA_ENV}/bin/torchrun}"
hash -r

cmd=(
    "${torchrun_bin}"
    --standalone
    --nnodes 1
    --nproc-per-node "${NPROC_PER_NODE}"
    vla-scripts/finetune.py
    --vlm_path "${VLM_PATH}"
    --config_file_path "${CONFIG_FILE_PATH}"
    --data_root_dir "${DATA_ROOT_DIR}"
    --dataset_name "${DATASET_NAME}"
    --run_root_dir "${RUN_ROOT_DIR}"
    --use_film "${USE_FILM}"
    --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
    --use_proprio "${USE_PROPRIO}"
    --use_lora "${USE_LORA}"
    --use_fz "${USE_FZ}"
    --use_minivlm "${USE_MINIVLM}"
    --image_aug "${IMAGE_AUG}"
    --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY}"
    --lr_scheduler "${LR_SCHEDULER}"
    --lr_warmup_ratio "${LR_WARMUP_RATIO}"
    --lr_warmup_steps "${LR_WARMUP_STEPS}"
    --max_steps "${MAX_STEPS}"
    --save_freq "${SAVE_FREQ}"
    --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY}"
    --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING}"
    --batch_size "${BATCH_SIZE}"
    --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --lora_rank "${LORA_RANK}"
    --use_pro_version "${USE_PRO_VERSION}"
    --use_val_set "${USE_VAL_SET}"
    --val_freq "${VAL_FREQ}"
    --val_time_limit "${VAL_TIME_LIMIT}"
    --wandb_entity "${WANDB_ENTITY}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_log_freq "${WANDB_LOG_FREQ}"
    --run_id_override "${EXP_NAME}"
    --use_pair_bridge True
    --pair_action_ae_encoder_path "${PAIR_ACTION_AE_ENCODER_PATH}"
    --pair_align_weight "${PAIR_ALIGN_WEIGHT}"
    --pair_bridge_dim "${PAIR_BRIDGE_DIM}"
    --pair_gate_mlp_dim "${PAIR_GATE_MLP_DIM}"
    --pair_gate_num_layers "${PAIR_GATE_NUM_LAYERS}"
    --pair_init_gate_mode "${PAIR_INIT_GATE_MODE}"
    --pair_init_gate_value "${PAIR_INIT_GATE_VALUE}"
    --pair_init_gate_granularity "${PAIR_INIT_GATE_GRANULARITY}"
    --pair_gate_activation "${PAIR_GATE_ACTIVATION}"
    --pair_injection_positions "${PAIR_INJECTION_POSITIONS}"
    --pair_log_debug_metrics "${PAIR_LOG_DEBUG_METRICS}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args_array=(${EXTRA_ARGS})
    cmd+=("${extra_args_array[@]}")
fi

cat <<EOF
[train_pair_bridge] dataset: ${DATASET_NAME}
[train_pair_bridge] data_root: ${DATA_ROOT_DIR}
[train_pair_bridge] gpus: ${CUDA_VISIBLE_DEVICES}
[train_pair_bridge] nproc_per_node: ${NPROC_PER_NODE}
[train_pair_bridge] effective_batch: $((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUMULATION_STEPS))
[train_pair_bridge] lr: ${LEARNING_RATE}, scheduler=${LR_SCHEDULER}, warmup_ratio=${LR_WARMUP_RATIO}, warmup_steps=${LR_WARMUP_STEPS}
[train_pair_bridge] python: ${python_bin}
[train_pair_bridge] torchrun: ${torchrun_bin}
[train_pair_bridge] action_ae_dir: ${ACTION_AE_DIR}
[train_pair_bridge] action_ae_encoder: ${PAIR_ACTION_AE_ENCODER_PATH}
[train_pair_bridge] pair_align_weight: ${PAIR_ALIGN_WEIGHT}
[train_pair_bridge] pair_dims: bridge=${PAIR_BRIDGE_DIM}, block_mlp=4x_bridge, init_mlp=4x_bridge, gate_layers=${PAIR_GATE_NUM_LAYERS}, gate_mlp=${PAIR_GATE_MLP_DIM}
[train_pair_bridge] pair_init_gate: mode=${PAIR_INIT_GATE_MODE}, actual_value=${PAIR_INIT_GATE_VALUE}, granularity=${PAIR_INIT_GATE_GRANULARITY}, activation=${PAIR_GATE_ACTIVATION}, positions=${PAIR_INJECTION_POSITIONS}
[train_pair_bridge] pair_log_debug_metrics: ${PAIR_LOG_DEBUG_METRICS}
[train_pair_bridge] wandb: ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_MODE})
[train_pair_bridge] exp_name: ${EXP_NAME}
[train_pair_bridge] run_root: ${RUN_ROOT_DIR}
[train_pair_bridge] log_file: ${log_file}
EOF

printf '[train_pair_bridge] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

if [[ "${BACKGROUND}" == "true" ]]; then
    nohup "${cmd[@]}" > "${log_file}" 2>&1 &
    pid="$!"
    echo "[train_pair_bridge] started in background: pid=${pid}"
    echo "[train_pair_bridge] tail log: tail -f ${log_file}"
else
    "${cmd[@]}" 2>&1 | tee "${log_file}"
fi
