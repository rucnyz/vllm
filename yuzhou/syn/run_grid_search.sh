#!/bin/bash

# TB × BS 网格搜索实验脚本
# 对比 baseline 和 PD scheduler 在所有 (TB, BS) 组合下的性能
#
# 用法: ./run_grid_search.sh [MAX_GPUS]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

# 实验参数
MAX_GPUS=${1:-4}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
NUM_PROMPTS=${NUM_PROMPTS:-4000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-10000}

# 网格搜索参数
BS_VALUES=(256 512 1024 1536 2048)
TB_VALUES=(4096 6144 8192 10240 14336 16384)
SCENARIOS=("128 1024" "1024 128" "512 512")

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/grid_search_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env

echo "========================================"
echo "TB × BS 网格搜索实验"
echo "========================================"

# 检测并选择 GPU
select_gpus $MAX_GPUS

echo ""
echo "实验配置:"
echo "  MODEL: $MODEL"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  K_RATIO: $K_RATIO"
echo "  BS_VALUES: ${BS_VALUES[*]}"
echo "  TB_VALUES: ${TB_VALUES[*]}"
echo "  SCENARIOS: ${#SCENARIOS[@]} 个"
echo ""

# 生成实验队列
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"

for tb in "${TB_VALUES[@]}"; do
    for bs in "${BS_VALUES[@]}"; do
        for scenario in "${SCENARIOS[@]}"; do
            read -r input_len output_len <<< "$scenario"
            echo "baseline|${input_len}|${output_len}|${bs}|${tb}" >> "$QUEUE_FILE"
            echo "pd|${input_len}|${output_len}|${bs}|${tb}" >> "$QUEUE_FILE"
        done
    done
done

TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
echo "总实验数: $TOTAL_EXPERIMENTS"
echo ""

# 保存全局配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "k_ratio": ${K_RATIO},
    "bs_values": [$(echo "${BS_VALUES[*]}" | sed 's/ /, /g')],
    "tb_values": [$(echo "${TB_VALUES[*]}" | sed 's/ /, /g')],
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

# 运行单个实验
run_experiment() {
    local gpu_id=$1 scheduler=$2 input_len=$3 output_len=$4 bs=$5 tb=$6
    local port=$((BASE_PORT + gpu_id))
    local scenario_name="in${input_len}_out${output_len}"
    local result_dir="${OUTPUT_DIR}/tb${tb}/bs${bs}/${scenario_name}"
    local log_file="${result_dir}/logs/${scheduler}.log"

    mkdir -p "${result_dir}/logs"
    : > "$log_file"

    check_port_available $port $gpu_id || return 1

    echo "[GPU $gpu_id] 开始: ${scheduler} tb=${tb} bs=${bs} ${scenario_name}"

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export VLLM_COLLECT_SCHEDULE_STATS=1

    if [ "$scheduler" = "baseline" ]; then
        export VLLM_USE_PD_SCHEDULER=0
        unset VLLM_PD_K_RATIO
    else
        export VLLM_USE_PD_SCHEDULER=1
        export VLLM_PD_K_RATIO=$K_RATIO
    fi

    # 等待 GPU 内存可用
    wait_for_gpu_memory $gpu_id 60 || return 1

    # 启动服务
    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${scheduler}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$bs" \
        --max-num-batched-tokens "$tb" >> "$log_file" 2>&1 &
    local server_pid=$!

    # 等待服务启动
    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "[GPU $gpu_id] 服务启动失败"
        kill_server $server_pid $gpu_id
        return 1
    fi

    # 运行 benchmark
    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${port}" \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmups "$NUM_WARMUP_REQUESTS" \
        --request-rate inf \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "bench_${scheduler}.json" >> "$log_file" 2>&1
    local bench_status=$?

    kill_server $server_pid $gpu_id

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] 完成: ${scheduler} tb=${tb} bs=${bs} ${scenario_name}"
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

        IFS='|' read -r scheduler input_len output_len bs tb <<< "$exp"

        if run_experiment "$gpu_id" "$scheduler" "$input_len" "$output_len" "$bs" "$tb"; then
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
    sleep 10  # 错开启动避免 CUDA 初始化竞争
done

echo ""
echo "监控进度: watch -n 5 'wc -l ${PROGRESS_FILE}'"
echo ""

for pid in "${WORKER_PIDS[@]}"; do
    wait $pid || true
done

print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"
echo ""
echo "运行分析脚本:"
echo "  python ${SCRIPT_DIR}/analyze_grid_search.py $OUTPUT_DIR"
