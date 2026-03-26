#!/bin/bash

# Request 数量扩展实验脚本
# 验证假设: 随着 request 数量增加，PD scheduler 的稳态时间占比更大，优势更明显
#
# 用法: ./run_request_scaling.sh [MAX_GPUS]
#
# 环境变量:
#   PD_TB: PD scheduler 的最优 token budget，默认 10240
#   PD_BS: PD scheduler 的最优 max_num_seqs，默认 1536
#   DEFAULT_TB: vLLM 默认 token budget (不设置则使用 vLLM 默认)
#   DEFAULT_BS: vLLM 默认 max_num_seqs (不设置则使用 vLLM 默认)
#   REQUEST_COUNTS_STR: request 数量列表，如 "1000 2000 3000"
#   K_RATIO: PD scheduler 的 k_ratio，默认 0.8
#   SKIP_PD: 设为 1 跳过 PD scheduler 实验
#   DEBUG: 设为 1 打印命令

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载公共函数库
source "${SCRIPT_DIR}/../common.sh"

# 实验参数
MAX_GPUS=${1:-4}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-8200}
SKIP_PD=${SKIP_PD:-0}
DEBUG=${DEBUG:-0}

# 场景固定为 in128_out1024
INPUT_LEN=128
OUTPUT_LEN=1024

# PD scheduler 最优配置
PD_TB=${PD_TB:-10240}
PD_BS=${PD_BS:-1536}

# vLLM 默认配置
DEFAULT_TB=${DEFAULT_TB:-""}
DEFAULT_BS=${DEFAULT_BS:-""}

# Request 数量扫描
if [ -n "$REQUEST_COUNTS_STR" ]; then
    read -ra REQUEST_COUNTS <<< "$REQUEST_COUNTS_STR"
else
    REQUEST_COUNTS=()
    for ((i=1; i<=30; i+=5)); do
        REQUEST_COUNTS+=($((i * 1000)))
    done
fi

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/request_scaling_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env
WORKER_PIDS=()
setup_cleanup $BASE_PORT

echo "========================================"
echo "Request 数量扩展实验"
echo "========================================"
echo "验证假设: 随着 request 数量增加，PD scheduler 优势更明显"
echo ""

# 检测并选择 GPU
select_gpus $MAX_GPUS
echo ""
echo "实验配置:"
echo "  场景: in${INPUT_LEN}_out${OUTPUT_LEN}"
echo "  MODEL: $MODEL"
echo "  K_RATIO: $K_RATIO"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo ""
if [ "$SKIP_PD" = "1" ]; then
    echo "对比线 (SKIP_PD=1, 跳过 PD):"
    echo "  1. Baseline (same):    TB=${PD_TB}, BS=${PD_BS}"
    echo "  2. Baseline (default): TB=${DEFAULT_TB:-vLLM默认}, BS=${DEFAULT_BS:-vLLM默认}"
else
    echo "三条对比线:"
    echo "  1. PD Scheduler:       TB=${PD_TB}, BS=${PD_BS}, k_ratio=${K_RATIO}"
    echo "  2. Baseline (same):    TB=${PD_TB}, BS=${PD_BS}"
    echo "  3. Baseline (default): TB=${DEFAULT_TB:-vLLM默认}, BS=${DEFAULT_BS:-vLLM默认}"
fi
echo ""
echo "Request 数量: ${REQUEST_COUNTS[*]}"
echo ""

# ========================================
# 生成实验队列
# ========================================
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"

for num_requests in "${REQUEST_COUNTS[@]}"; do
    if [ "$SKIP_PD" != "1" ]; then
        echo "pd_optimal|${num_requests}|${PD_TB}|${PD_BS}|1" >> "$QUEUE_FILE"
    fi
    echo "baseline_same|${num_requests}|${PD_TB}|${PD_BS}|0" >> "$QUEUE_FILE"
    echo "baseline_default|${num_requests}|${DEFAULT_TB}|${DEFAULT_BS}|0" >> "$QUEUE_FILE"
done

TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
NUM_CONFIGS=$((SKIP_PD == 1 ? 2 : 3))
echo "总实验数: $TOTAL_EXPERIMENTS"
echo "  = $NUM_CONFIGS 配置 × ${#REQUEST_COUNTS[@]} request数量"
echo ""

# 保存全局配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "request_scaling",
    "hypothesis": "PD scheduler advantage increases with more requests",
    "scenario": {"input_len": ${INPUT_LEN}, "output_len": ${OUTPUT_LEN}},
    "configurations": {
        "pd_optimal": {"scheduler": "pd", "tb": ${PD_TB}, "bs": ${PD_BS}, "k_ratio": ${K_RATIO}},
        "baseline_same": {"scheduler": "baseline", "tb": ${PD_TB}, "bs": ${PD_BS}},
        "baseline_default": {"scheduler": "baseline", "tb": "${DEFAULT_TB:-null}", "bs": "${DEFAULT_BS:-null}"}
    },
    "request_counts": [$(echo "${REQUEST_COUNTS[*]}" | sed 's/ /, /g')],
    "model": "${MODEL}",
    "max_concurrency": ${MAX_CONCURRENCY},
    "gpus_used": [${GPUS_TO_USE[*]}],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

echo "配置已保存: ${OUTPUT_DIR}/experiment_config.json"
echo ""

# ========================================
# Worker 函数
# ========================================
run_experiment() {
    local gpu_id=$1
    local config_name=$2
    local num_requests=$3
    local tb=$4
    local bs=$5
    local use_pd=$6

    local port=$((BASE_PORT + gpu_id))

    check_port_available $port $gpu_id || return 1

    local result_dir="${OUTPUT_DIR}/requests_${num_requests}"
    mkdir -p "${result_dir}/logs"
    local log_file="${result_dir}/logs/${config_name}.log"

    echo "[GPU $gpu_id] 开始: ${config_name} requests=${num_requests}"

    # 构建 serve 参数
    local serve_args="--gpu-memory-utilization 0.9"
    [ -n "$tb" ] && serve_args="$serve_args --max-num-batched-tokens $tb"
    [ -n "$bs" ] && serve_args="$serve_args --max-num-seqs $bs"

    # 保存配置
    cat > "${result_dir}/config_${config_name}.json" << EOF
{
    "config_name": "${config_name}",
    "num_requests": ${num_requests},
    "max_num_batched_tokens": ${tb:-null},
    "max_num_seqs": ${bs:-null},
    "use_pd_scheduler": ${use_pd},
    "k_ratio": ${K_RATIO},
    "input_len": ${INPUT_LEN},
    "output_len": ${OUTPUT_LEN},
    "gpu_id": ${gpu_id},
    "port": ${port},
    "model": "${MODEL}",
    "max_concurrency": ${MAX_CONCURRENCY},
    "timestamp": "$(date -Iseconds)"
}
EOF

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export VLLM_COLLECT_SCHEDULE_STATS=1

    if [ "$use_pd" = "1" ]; then
        # 使用 PD 调度器 (比例模式)
        export VLLM_USE_PD_SCHEDULER=1
        export VLLM_PD_K_MODE=ratio
        export VLLM_PD_K_RATIO=$K_RATIO
    else
        # 使用默认 vLLM 调度器
        export VLLM_USE_PD_SCHEDULER=0
    fi

    [ "$DEBUG" = "1" ] && echo "[DEBUG] vllm serve $MODEL --port $port $serve_args"

    # 启动服务
    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${config_name}_stats.json" \
    vllm serve $MODEL --port $port $serve_args > "$log_file" 2>&1 &

    local server_pid=$!

    if ! wait_for_server $port $server_pid; then
        echo "[GPU $gpu_id] 服务启动失败: ${config_name}"
        kill_server $server_pid
        return 1
    fi

    # 运行 benchmark
    vllm bench serve \
        --model $MODEL \
        --base-url "http://localhost:${port}" \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --random-range-ratio $RANDOM_RANGE_RATIO \
        --num-prompts $num_requests \
        --num-warmups $NUM_WARMUP_REQUESTS \
        --request-rate inf \
        --max-concurrency $MAX_CONCURRENCY \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "bench_${config_name}.json" \
        >> "$log_file" 2>&1

    local bench_status=$?
    kill_server $server_pid

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] 完成: ${config_name} requests=${num_requests}"
    else
        echo "[GPU $gpu_id] 失败: ${config_name} requests=${num_requests}"
    fi

    return $bench_status
}

# ========================================
# 并行调度器
# ========================================
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && { echo "[GPU $gpu_id] 队列为空，退出"; break; }

        IFS='|' read -r config_name num_requests tb bs use_pd <<< "$exp"

        if run_experiment "$gpu_id" "$config_name" "$num_requests" "$tb" "$bs" "$use_pd"; then
            update_progress "OK|${config_name}|${num_requests}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${config_name}|${num_requests}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        fi
    done
}

# ========================================
# 主流程
# ========================================
echo "开始并行执行..."
echo "========================================"

> "$PROGRESS_FILE"

for gpu_id in "${GPUS_TO_USE[@]}"; do
    gpu_worker "$gpu_id" &
    WORKER_PIDS+=($!)
    echo "启动 GPU $gpu_id worker (PID: ${WORKER_PIDS[-1]})"
done

echo ""
echo "监控进度: watch -n 5 'wc -l ${OUTPUT_DIR}/progress.txt'"
echo ""

for pid in "${WORKER_PIDS[@]}"; do
    wait $pid
done

print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"
echo ""
echo "运行分析脚本:"
echo "  python ${SCRIPT_DIR}/analyze_request_scaling.py $OUTPUT_DIR"
