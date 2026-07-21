#!/usr/bin/env bash
set -euo pipefail

# VLA-Adapter training launcher. Edit only this block for normal runs.
# You can also override any value when launching:
#   DATASET_NAME=calvin_abc ./train_vla_adapter.sh

#########################
# Official-style settings
#########################

# Choose one:
# libero_spatial_no_noops | libero_object_no_noops | libero_goal_no_noops | libero_10_no_noops | calvin_abc
DATASET_NAME="${DATASET_NAME:-libero_object_no_noops}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-64}"
MAX_STEPS="${MAX_STEPS:-50005}"
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-50000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-True}"
USE_PRO_VERSION="${USE_PRO_VERSION:-False}"

WANDB_ENTITY="${WANDB_ENTITY:-kaixi-university-of-maryland}"
WANDB_PROJECT="${WANDB_PROJECT:-PAIR}"
EXP_NAME="${EXP_NAME:-baseline_object_64}"

# Utility switches.
DRY_RUN="${DRY_RUN:-false}"
BACKGROUND="${BACKGROUND:-false}"

#################
# Internal wiring
#################

CONDA_ENV="${CONDA_ENV:-vla-adapter}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-true}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"

ENV_SH="${ENV_SH:-/data/kaixi/PAIR/kaixi_scripts/env.sh}"
REPO_DIR="${REPO_DIR:-/data/kaixi/PAIR/VLA-Adapter}"
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


WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-10}"
export WANDB_MODE="${WANDB_MODE:-online}"


USE_VAL_SET="${USE_VAL_SET:-False}"
VAL_FREQ="${VAL_FREQ:-10000}"
VAL_TIME_LIMIT="${VAL_TIME_LIMIT:-180}"

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
    EXP_NAME="pair_${DATASET_NAME}_g${NPROC_PER_NODE}_b${BATCH_SIZE}_ga${GRAD_ACCUMULATION_STEPS}_s${MAX_STEPS}_pro${USE_PRO_VERSION}_${current_time}"
fi

if [[ -z "${DATA_ROOT_DIR:-}" ]]; then
    case "${DATASET_NAME}" in
        calvin_abc)
            DATA_ROOT_DIR="${LOCAL_DATA_ROOT}"
            ;;
        libero_*)
            DATA_ROOT_DIR="${LOCAL_DATA_ROOT}/libero"
            ;;
        *)
            DATA_ROOT_DIR="${LOCAL_DATA_ROOT}"
            ;;
    esac
fi

log_file="${LOG_FILE:-${LOG_DIR}/VLA-Adapter--${DATASET_NAME}--${current_time}.log}"

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
)

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args_array=(${EXTRA_ARGS})
    cmd+=("${extra_args_array[@]}")
fi

cat <<EOF
[train_vla_adapter] dataset: ${DATASET_NAME}
[train_vla_adapter] data_root: ${DATA_ROOT_DIR}
[train_vla_adapter] gpus: ${CUDA_VISIBLE_DEVICES}
[train_vla_adapter] nproc_per_node: ${NPROC_PER_NODE}
[train_vla_adapter] effective_batch: $((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUMULATION_STEPS))
[train_vla_adapter] python: ${python_bin}
[train_vla_adapter] torchrun: ${torchrun_bin}
[train_vla_adapter] wandb: ${WANDB_ENTITY}/${WANDB_PROJECT}
[train_vla_adapter] exp_name: ${EXP_NAME}
[train_vla_adapter] wandb_run_name: ${EXP_NAME}
[train_vla_adapter] run_root: ${RUN_ROOT_DIR}
[train_vla_adapter] log_file: ${log_file}
EOF

printf '[train_vla_adapter] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

if [[ "${BACKGROUND}" == "true" ]]; then
    nohup "${cmd[@]}" > "${log_file}" 2>&1 &
    pid="$!"
    echo "[train_vla_adapter] started in background: pid=${pid}"
    echo "[train_vla_adapter] tail log: tail -f ${log_file}"
else
    "${cmd[@]}" 2>&1 | tee "${log_file}"
fi
