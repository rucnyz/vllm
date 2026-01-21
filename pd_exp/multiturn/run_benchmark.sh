#!/bin/bash

# 多轮对话 Benchmark 脚本 (Prefix Cache 测试)
# 对比 baseline 和 PD scheduler 在多轮对话场景下的性能
#
# 用法: ./run_benchmark.sh <DATASET_PATH> [MAX_GPUS]
#
# 示例:
#   # 先导出数据集
#   python pd_exp/multiturn/export_dataset.py \
#       --dataset wildchat \
#       --model Qwen/Qwen3-8B \
#       --num-conversations 500 \
#       --min-turns 8 \
#       --output ./pd_exp/outputs/wildchat_multiturn.json
#
#   # 运行实验
#   ./pd_exp/multiturn/run_benchmark.sh ./pd_exp/outputs/wildchat_multiturn.json 4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../syn/common.sh"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

# 检查参数
if [ -z "${1:-}" ]; then
    echo "用法: $0 <DATASET_PATH> [MAX_GPUS]"
    echo ""
    echo "必须提供数据集文件路径 (JSON 格式，多轮对话)"
    echo ""
    echo "示例:"
    echo "  # 先导出数据集"
    echo "  python pd_exp/multiturn/export_dataset.py \\"
    echo "      --dataset wildchat \\"
    echo "      --model Qwen/Qwen3-8B \\"
    echo "      --num-conversations 500 \\"
    echo "      --min-turns 8 \\"
    echo "      --output ./pd_exp/outputs/wildchat_multiturn.json"
    echo ""
    echo "  # 运行实验"
    echo "  $0 ./pd_exp/outputs/wildchat_multiturn.json 4"
    exit 1
fi

DATASET_PATH="$1"
MAX_GPUS=${2:-4}

# 检查数据集文件
if [ ! -f "$DATASET_PATH" ]; then
    echo "错误: 数据集文件不存在: $DATASET_PATH"
    exit 1
fi

if [[ "$DATASET_PATH" != *.json ]]; then
    echo "错误: 数据集文件必须是 JSON 格式 (.json)"
    echo "请使用 pd_exp/multiturn/export_dataset.py 导出数据集"
    exit 1
fi

# 获取数据集名称（用于输出目录）
DATASET_NAME=$(basename "$DATASET_PATH" .json)

# 实验参数
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
NUM_CLIENTS=${NUM_CLIENTS:-8}
MAX_TURNS=${MAX_TURNS:-10}
LIMIT_MAX_TOKENS=${LIMIT_MAX_TOKENS:-256}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
BASE_PORT=${BASE_PORT:-10000}
K_RATIO=${K_RATIO:-0.8}

# 硬件校准文件 (必须)
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    DEFAULT_CALIBRATION="${SCRIPT_DIR}/../outputs/pd_calibration.json"
    if [ -f "$DEFAULT_CALIBRATION" ]; then
        export VLLM_PD_CALIBRATION_FILE="$DEFAULT_CALIBRATION"
    else
        echo "错误: 未找到硬件校准文件!"
        echo ""
        echo "PD Scheduler 需要硬件校准参数才能准确调度。"
        echo "请先运行校准:"
        echo "  python -m vllm.v1.core.sched.calibration --model ${MODEL}"
        echo ""
        echo "校准文件默认保存到: ${DEFAULT_CALIBRATION}"
        echo "或手动指定: VLLM_PD_CALIBRATION_FILE=/path/to/file.json $0 ..."
        exit 1
    fi
fi
echo "使用校准文件: $VLLM_PD_CALIBRATION_FILE"

# 从校准文件中读取 alpha/beta 参数
if [ -f "$VLLM_PD_CALIBRATION_FILE" ]; then
    ALPHA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_p'])" 2>/dev/null || echo "null")
    BETA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_p'])" 2>/dev/null || echo "null")
    ALPHA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_d'])" 2>/dev/null || echo "null")
    BETA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_d'])" 2>/dev/null || echo "null")
    echo "  alpha_p: $ALPHA_P, beta_p: $BETA_P"
    echo "  alpha_d: $ALPHA_D, beta_d: $BETA_D"
else
    ALPHA_P="null"
    BETA_P="null"
    ALPHA_D="null"
    BETA_D="null"
fi

# 网格搜索参数
BS_VALUES=(${BS_VALUES:-256 512 1024})
TB_VALUES=(${TB_VALUES:-8192 16384})

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/multiturn_${DATASET_NAME}_Clients_${NUM_CLIENTS}_MaxTurns_${MAX_TURNS}"
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env

echo "========================================"
echo "多轮对话 Benchmark (Prefix Cache 测试)"
echo "========================================"

# 检测并选择 GPU
select_gpus $MAX_GPUS

echo ""
echo "实验配置:"
echo "  DATASET: $DATASET_PATH"
echo "  MODEL: $MODEL"
echo "  NUM_CLIENTS: $NUM_CLIENTS"
echo "  MAX_TURNS: $MAX_TURNS"
echo "  LIMIT_MAX_TOKENS: $LIMIT_MAX_TOKENS"
echo "  K_RATIO (for pd_ratio): $K_RATIO"
echo "  BS_VALUES: ${BS_VALUES[*]}"
echo "  TB_VALUES: ${TB_VALUES[*]}"
echo "  SCHEDULERS: baseline, pd_ratio (θ*=${K_RATIO}), pd_direct"
echo "  CALIBRATION_FILE: ${VLLM_PD_CALIBRATION_FILE:-"(未设置)"}"
echo ""

# 生成实验队列
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"

for tb in "${TB_VALUES[@]}"; do
    for bs in "${BS_VALUES[@]}"; do
        echo "baseline|${bs}|${tb}" >> "$QUEUE_FILE"
        echo "pd_ratio|${bs}|${tb}" >> "$QUEUE_FILE"
        echo "pd_direct|${bs}|${tb}" >> "$QUEUE_FILE"
    done
done

TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
echo "总实验数: $TOTAL_EXPERIMENTS"
echo ""

# 保存全局配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "dataset_path": "${DATASET_PATH}",
    "dataset_name": "${DATASET_NAME}",
    "model": "${MODEL}",
    "num_clients": ${NUM_CLIENTS},
    "max_turns": ${MAX_TURNS},
    "limit_max_tokens": ${LIMIT_MAX_TOKENS},
    "request_timeout": ${REQUEST_TIMEOUT},
    "k_ratio": ${K_RATIO},
    "bs_values": [$(echo "${BS_VALUES[*]}" | sed 's/ /, /g')],
    "tb_values": [$(echo "${TB_VALUES[*]}" | sed 's/ /, /g')],
    "schedulers": ["baseline", "pd_ratio", "pd_direct"],
    "scheduler_descriptions": {
        "baseline": "vLLM default scheduler",
        "pd_ratio": "PD scheduler with ratio mode (θ*=${K_RATIO})",
        "pd_direct": "PD scheduler with direct mode (auto k*)"
    },
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE:-null}",
    "calibration_params": {
        "alpha_p": ${ALPHA_P},
        "beta_p": ${BETA_P},
        "alpha_d": ${ALPHA_D},
        "beta_d": ${BETA_D}
    },
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

# 运行单个实验
run_experiment() {
    local gpu_id=$1 scheduler=$2 bs=$3 tb=$4
    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/tb${tb}/bs${bs}"
    local log_file="${result_dir}/logs/${scheduler}.log"
    local bench_log="${result_dir}/logs/${scheduler}_bench.log"

    mkdir -p "${result_dir}/logs"
    : > "$log_file"
    : > "$bench_log"

    check_port_available $port $gpu_id || return 1

    echo "[GPU $gpu_id] 开始: ${scheduler} tb=${tb} bs=${bs}"

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$gpu_id

    case "$scheduler" in
        baseline)
            export VLLM_USE_PD_SCHEDULER=0
            unset VLLM_PD_K_MODE VLLM_PD_K_STAR VLLM_PD_K_RATIO
            ;;
        pd_ratio)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            unset VLLM_PD_K_STAR
            ;;
        pd_direct)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=direct
            unset VLLM_PD_K_RATIO VLLM_PD_K_STAR
            ;;
    esac

    wait_for_gpu_memory $gpu_id 60 || return 1

    # 启动服务
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$bs" \
        --max-num-batched-tokens "$tb" >> "$log_file" 2>&1 &
    local server_pid=$!

    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "[GPU $gpu_id] 服务启动失败"
        kill_server $server_pid $gpu_id
        return 1
    fi

    # 运行多轮对话 benchmark
    python benchmarks/multi_turn/benchmark_serving_multi_turn_threaded.py \
        --input-file "$DATASET_PATH" \
        --model "$MODEL" \
        --url "http://localhost:${port}" \
        --num-clients "$NUM_CLIENTS" \
        --max-turns "$MAX_TURNS" \
        --limit-max-tokens "$LIMIT_MAX_TOKENS" \
        --request-timeout-sec "$REQUEST_TIMEOUT" \
        --output-file "${result_dir}/${scheduler}_results.json" \
        > "$bench_log" 2>&1
    local bench_status=$?

    # 提取关键指标
    if [ $bench_status -eq 0 ]; then
        # 从 benchmark 输出中提取统计信息
        grep -E "(ttft_ms|tpot_ms|latency_ms|approx_cached_percent)" "$bench_log" > "${result_dir}/${scheduler}_metrics.txt" 2>/dev/null || true
    fi

    kill_server $server_pid $gpu_id

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] 完成: ${scheduler} tb=${tb} bs=${bs}"
    else
        echo "[GPU $gpu_id] 失败: ${scheduler} tb=${tb} bs=${bs}"
    fi

    return $bench_status
}

# 并行调度
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break

        IFS='|' read -r scheduler bs tb <<< "$exp"

        if run_experiment "$gpu_id" "$scheduler" "$bs" "$tb"; then
            update_progress "OK|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        fi
    done
}

# 主流程
echo "开始并行执行..."
echo "========================================"

> "$PROGRESS_FILE"

for gpu_id in "${GPUS_TO_USE[@]}"; do
    gpu_worker "$gpu_id" &
    WORKER_PIDS+=($!)
    echo "启动 GPU $gpu_id worker (PID: ${WORKER_PIDS[-1]})"
    sleep 10
done

echo ""
echo "监控进度: watch -n 5 'wc -l ${PROGRESS_FILE}'"
echo ""

for pid in "${WORKER_PIDS[@]}"; do
    wait $pid || true
done

print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"
echo ""
echo "========================================"
echo "实验完成!"
echo "========================================"
echo ""
echo "结果目录: $OUTPUT_DIR"
echo ""
echo "运行分析脚本:"
echo "  # 结果分析 (scheduler 对比)"
echo "  python ${SCRIPT_DIR}/analyze_results.py $OUTPUT_DIR"
echo ""
echo "  # 导出 CSV"
echo "  python ${SCRIPT_DIR}/analyze_results.py $OUTPUT_DIR --csv ${OUTPUT_DIR}/results.csv"
