#!/usr/bin/env bash
set -euo pipefail

# LIBERO eval launcher for VLA-Adapter/PAIR checkpoints. Edit only eval settings.
# Example:
#   PRETRAINED_CHECKPOINT=/umd-datapool/kaixi/PAIR/checkpoints/pair_bridge_spatial TASK_SUITES=libero_spatial ./eval_libero_vla_adapter.sh

#####################
# Eval settings
#####################

# Choose one or several:
# libero_spatial | libero_object | libero_goal | libero_10
TASK_SUITES="${TASK_SUITES:-libero_spatial}"

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/umd-datapool/kaixi/PAIR/checkpoints/l20_PAIR_v8_learnable0_specialize_spatial_acc--50000_chkpt}"
GPU="${GPU:-3}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SEED="${SEED:-7}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
INITIAL_STATES_PATH="${INITIAL_STATES_PATH:-DEFAULT}"
ENV_IMG_RES="${ENV_IMG_RES:-256}"

# WandB is off by default for eval. Set USE_WANDB=True after `wandb login`
USE_WANDB="${USE_WANDB:-False}"
WANDB_ENTITY="${WANDB_ENTITY:-kaixi-university-of-maryland}"
WANDB_PROJECT="${WANDB_PROJECT:-PAIR}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Leave empty to derive a readable name from the checkpoint and suite.
EXP_NAME="${EXP_NAME:-}"

# Utility switches.
DRY_RUN="${DRY_RUN:-false}"
BACKGROUND="${BACKGROUND:-false}"

#################
# Internal wiring
#################

ENV_SH="${ENV_SH:-/data/kaixi/PAIR/kaixi_scripts/env.sh}"
REPO_DIR="${REPO_DIR:-/data/kaixi/PAIR/VLA-Adapter}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/eval_logs}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${LOG_DIR}/local}"

SAVE_VERSION="${SAVE_VERSION:-}"

############################
# Derived settings
############################

source "${ENV_SH}"

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}" "${LOCAL_LOG_DIR}"

current_time="${CURRENT_TIME:-$(date +%Y%m%d_%H%M%S)}"
checkpoint_name="$(basename "${PRETRAINED_CHECKPOINT}")"

read -r -a task_suite_array <<< "${TASK_SUITES}"

if [[ -z "${EXP_NAME}" ]]; then
    if [[ "${#task_suite_array[@]}" -eq 1 ]]; then
        EXP_NAME="eval_${checkpoint_name}_${task_suite_array[0]}_${current_time}"
    else
        EXP_NAME="eval_${checkpoint_name}_multi_${current_time}"
    fi
fi

if [[ -z "${SAVE_VERSION}" ]]; then
    SAVE_VERSION="${EXP_NAME}"
fi

log_file="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

if [[ ! -d "${PRETRAINED_CHECKPOINT}" ]]; then
    case "${PRETRAINED_CHECKPOINT}" in
        VLA-Adapter/*)
            ;;
        *)
            echo "PRETRAINED_CHECKPOINT does not exist: ${PRETRAINED_CHECKPOINT}" >&2
            echo "Set PRETRAINED_CHECKPOINT to a local checkpoint directory or a supported VLA-Adapter HF repo." >&2
            exit 1
            ;;
    esac
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
PYTHON_BIN="${PYTHON_BIN:-${PAIR_PYTHON:-python}}"

run_preflight() {
    "${PYTHON_BIN}" - <<'PY'
import robosuite
import tensorflow as tf
import experiments.robot.libero.libero_utils
from prismatic.models.pair_bridge import PairBridge
print(f"preflight ok: robosuite={robosuite.__version__}, tensorflow={tf.__version__}, pair_bridge={PairBridge.__name__}")
PY
}

run_suite() {
    local task_suite_name="$1"
    local run_id="${EXP_NAME}"
    if [[ "${#task_suite_array[@]}" -gt 1 ]]; then
        run_id="${EXP_NAME}_${task_suite_name}"
    fi

    local cmd=(
        "${PYTHON_BIN}"
        experiments/robot/libero/run_libero_eval.py
        --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
        --task_suite_name "${task_suite_name}"
        --num_trials_per_task "${NUM_TRIALS_PER_TASK}"
        --num_steps_wait "${NUM_STEPS_WAIT}"
        --initial_states_path "${INITIAL_STATES_PATH}"
        --env_img_res "${ENV_IMG_RES}"
        --seed "${SEED}"
        --save_version "${SAVE_VERSION}"
        --local_log_dir "${LOCAL_LOG_DIR}"
        --use_wandb "${USE_WANDB}"
        --wandb_entity "${WANDB_ENTITY}"
        --wandb_project "${WANDB_PROJECT}"
        --run_id_override "${run_id}"
    )

    echo
    echo "============================================================"
    echo "Running LIBERO eval: ${task_suite_name}"
    echo "checkpoint: ${PRETRAINED_CHECKPOINT}"
    echo "model wiring: loaded from checkpoint manifest/components"
    echo "wandb_run_name: ${run_id}"
    echo "rollouts: ${REPO_DIR}/rollouts/${SAVE_VERSION}"
    echo "============================================================"
    printf '[eval_libero_vla_adapter] command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if [[ "${DRY_RUN}" == "true" ]]; then
        return 0
    fi

    "${cmd[@]}"
}

cat <<EOF
[eval_libero_vla_adapter] checkpoint: ${PRETRAINED_CHECKPOINT}
[eval_libero_vla_adapter] task_suites: ${TASK_SUITES}
[eval_libero_vla_adapter] cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}
[eval_libero_vla_adapter] num_trials_per_task: ${NUM_TRIALS_PER_TASK}
[eval_libero_vla_adapter] seed: ${SEED}
[eval_libero_vla_adapter] model_wiring: checkpoint manifest/components
[eval_libero_vla_adapter] wandb: ${WANDB_ENTITY}/${WANDB_PROJECT}, enabled=${USE_WANDB}
[eval_libero_vla_adapter] exp_name: ${EXP_NAME}
[eval_libero_vla_adapter] log_file: ${log_file}
EOF

if [[ "${DRY_RUN}" == "true" ]]; then
    for task_suite_name in "${task_suite_array[@]}"; do
        run_suite "${task_suite_name}"
    done
    exit 0
fi

if [[ "${BACKGROUND}" == "true" ]]; then
    (
        run_preflight
        for task_suite_name in "${task_suite_array[@]}"; do
            run_suite "${task_suite_name}"
        done
    ) > "${log_file}" 2>&1 &
    pid="$!"
    echo "[eval_libero_vla_adapter] started in background: pid=${pid}"
    echo "[eval_libero_vla_adapter] tail log: tail -f ${log_file}"
else
    run_preflight 2>&1 | tee "${log_file}"
    for task_suite_name in "${task_suite_array[@]}"; do
        run_suite "${task_suite_name}" 2>&1 | tee -a "${log_file}"
    done
fi
