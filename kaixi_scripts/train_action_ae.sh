#!/usr/bin/env bash
set -euo pipefail

# PAIR Action AE training launcher. Override values at launch, for example:
#   MAX_STEPS=2 BATCH_SIZE=4 WANDB_MODE=disabled ./train_action_ae.sh
#   AE_VERSION=conditioned LATENT_DIM=16 MASK_MODE=chunk MASK_COUNT=4 ./train_action_ae.sh
#   DATASET_NAME=libero_spatial_no_noops GPUS=0,1 BATCH_SIZE=16 ./train_action_ae.sh

#########################
# User-facing settings
#########################

AE_VERSION="${AE_VERSION:-v2}"  # v1/action_only or v2/conditioned
DATASET_NAME="${DATASET_NAME:-libero_10_no_noops}" # libero_4_task_suites_no_noops
GPUS="${GPUS:-4,5,6,7}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-50000}"

LEARNING_RATE="${LEARNING_RATE:-3e-4}"
LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-50000}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SEED="${SEED:-7}"
LATENT_DIM="${LATENT_DIM:-16}"
ENCODER_LAYERS="${ENCODER_LAYERS:-1}"
PERCEPTION_LAYERS="${PERCEPTION_LAYERS:-1}"
DECODER_LAYERS="${DECODER_LAYERS:-1}"

# Corruption settings. random uses MASK_PROB; chunk masks exactly MASK_COUNT consecutive steps.
MASK_MODE="${MASK_MODE:-random}"
MASK_PROB="${MASK_PROB:-0.5}"
MASK_COUNT="${MASK_COUNT:-4}"
NOISE_STD="${NOISE_STD:-0.05}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"

WANDB_ENTITY="${WANDB_ENTITY:-kaixi-university-of-maryland}"
WANDB_PROJECT="${WANDB_PROJECT:-PAIR}"
export WANDB_MODE="${WANDB_MODE:-online}"
EXP_NAME="${EXP_NAME:-conditionedAE_en2_de1_10}"

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
ACTION_AE_DIR="${ACTION_AE_DIR:-${PAIR_ROOT}/action_ae}"
VLA_ADAPTER_DIR="${VLA_ADAPTER_DIR:-${PAIR_ROOT}/VLA-Adapter}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/data/kaixi/dataset/libero}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/umd-datapool/kaixi/PAIR/action_ae_runs}"
LOG_DIR="${LOG_DIR:-${ACTION_AE_DIR}/logs}"
CONFIG_PATH="${CONFIG_PATH:-auto}"

# Only v2 uses the frozen VLA to extract perception tokens.
VLA_PATH="${VLA_PATH:-${VLA_ADAPTER_DIR}/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}"
VLA_CONFIG_FILE_PATH="${VLA_CONFIG_FILE_PATH:-${VLA_ADAPTER_DIR}/pretrained_models/configs}"
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

case "${AE_VERSION}" in
    v1|action_only|only_action)
        AE_VERSION="v1"
        ;;
    v2|conditioned|condition|perception)
        AE_VERSION="v2"
        ;;
    *)
        echo "[train_action_ae] ERROR: unsupported AE_VERSION=${AE_VERSION}" >&2
        echo "[train_action_ae] Use v1/action_only or v2/conditioned." >&2
        exit 2
        ;;
esac

if [[ "${CONFIG_PATH}" == "auto" ]]; then
    if [[ "${AE_VERSION}" == "v2" ]]; then
        CONFIG_PATH="${ACTION_AE_DIR}/configs/libero_all_v2_perception.yaml"
    else
        CONFIG_PATH="${ACTION_AE_DIR}/configs/libero_all.yaml"
    fi
fi

if [[ -z "${EXP_NAME}" ]]; then
    if [[ "${AE_VERSION}" == "v2" ]]; then
        EXP_NAME="ae_v2_${DATASET_NAME}_en${ENCODER_LAYERS}_pe${PERCEPTION_LAYERS}_de${DECODER_LAYERS}_${current_time}"
    else
        EXP_NAME="ae_v1_${DATASET_NAME}_${current_time}"
    fi
fi
log_file="${LOG_FILE:-${LOG_DIR}/ActionAE--${EXP_NAME}.log}"

############################
# Environment
############################

source "${ENV_SH}"

if [[ "${ACTIVATE_CONDA}" == "true" ]]; then
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV}"
fi

mkdir -p "${LOG_DIR}" "${RUN_ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS// /}"
export PYTHONPATH="${ACTION_AE_DIR}:${VLA_ADAPTER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python_bin="${PAIR_PYTHON:-python}"

train_args=(
    --config "${CONFIG_PATH}"
    --ae_version "${AE_VERSION}"
    --data_root_dir "${DATA_ROOT_DIR}"
    --mixture "${DATASET_NAME}"
    --run_root_dir "${RUN_ROOT_DIR}"
    --run_name "${EXP_NAME}"
    --batch_size "${BATCH_SIZE}"
    --max_steps "${MAX_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --log_every "${LOG_EVERY}"
    --eval_every "${EVAL_EVERY}"
    --save_every "${SAVE_EVERY}"
    --eval_batches "${EVAL_BATCHES}"
    --seed "${SEED}"
    --wandb_entity "${WANDB_ENTITY}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_mode "${WANDB_MODE}"
)

if [[ -n "${LATENT_DIM}" ]]; then
    train_args+=(--latent_dim "${LATENT_DIM}")
fi

if [[ "${AE_VERSION}" == "v2" ]]; then
    train_args+=(
        --vlm_path "${VLA_PATH}"
        --vla_config_file_path "${VLA_CONFIG_FILE_PATH}"
        --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
        --encoder_layers "${ENCODER_LAYERS}"
        --perception_layers "${PERCEPTION_LAYERS}"
        --decoder_layers "${DECODER_LAYERS}"
        --mask_mode "${MASK_MODE}"
        --mask_prob "${MASK_PROB}"
        --mask_count "${MASK_COUNT}"
        --noise_std "${NOISE_STD}"
    )
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args_array=(${EXTRA_ARGS})
    train_args+=("${extra_args_array[@]}")
fi

if (( NPROC_PER_NODE > 1 )); then
    cmd=(
        "${python_bin}"
        -m torch.distributed.run
        --standalone
        --nnodes 1
        --nproc_per_node "${NPROC_PER_NODE}"
        --module pair_action_ae.train
        "${train_args[@]}"
    )
else
    cmd=(
        "${python_bin}"
        -m pair_action_ae.train
        "${train_args[@]}"
    )
fi

cat <<EOF
[train_action_ae] ae_version: ${AE_VERSION}
[train_action_ae] dataset_name: ${DATASET_NAME}
[train_action_ae] config: ${CONFIG_PATH}
[train_action_ae] data_root: ${DATA_ROOT_DIR}
[train_action_ae] gpus: ${CUDA_VISIBLE_DEVICES}
[train_action_ae] nproc_per_node: ${NPROC_PER_NODE}
[train_action_ae] batch_size_per_rank: ${BATCH_SIZE}
[train_action_ae] effective_batch_size: $(( BATCH_SIZE * NPROC_PER_NODE ))
[train_action_ae] max_steps: ${MAX_STEPS}
[train_action_ae] learning_rate: ${LEARNING_RATE}
[train_action_ae] latent_dim_override: ${LATENT_DIM:-config default}
[train_action_ae] wandb: ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_MODE})
[train_action_ae] exp_name: ${EXP_NAME}
[train_action_ae] run_root: ${RUN_ROOT_DIR}
[train_action_ae] log_file: ${log_file}
EOF

if [[ "${AE_VERSION}" == "v2" ]]; then
    cat <<EOF
[train_action_ae] vla_path: ${VLA_PATH}
[train_action_ae] vla_config_file_path: ${VLA_CONFIG_FILE_PATH}
[train_action_ae] num_images_in_input: ${NUM_IMAGES_IN_INPUT}
[train_action_ae] encoder_layers: ${ENCODER_LAYERS}
[train_action_ae] perception_layers: ${PERCEPTION_LAYERS}
[train_action_ae] decoder_layers: ${DECODER_LAYERS}
[train_action_ae] mask_mode: ${MASK_MODE}
[train_action_ae] mask_prob: ${MASK_PROB}
[train_action_ae] mask_count: ${MASK_COUNT}
[train_action_ae] noise_std: ${NOISE_STD}
EOF
fi

printf '[train_action_ae] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

if [[ "${BACKGROUND}" == "true" ]]; then
    nohup "${cmd[@]}" > "${log_file}" 2>&1 &
    pid="$!"
    echo "[train_action_ae] started in background: pid=${pid}"
    echo "[train_action_ae] tail log: tail -f ${log_file}"
else
    "${cmd[@]}" 2>&1 | tee "${log_file}"
fi
