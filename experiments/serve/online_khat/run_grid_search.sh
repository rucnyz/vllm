#!/bin/bash

# TB × BS 网格搜索实验脚本
# 对比 baseline 和 PD scheduler 在所有 (TB, BS) 组合下的性能
#
# 用法: ./run_grid_search.sh [MAX_GPUS]
#   MAX_GPUS: 最大使用的 GPU 数量，默认 4
#
# 环境变量:
#   GPU_MEM_THRESHOLD: GPU 内存使用阈值 (MiB)，低于此值认为空闲，默认 10000
#   NUM_PROMPTS: 每个实验的 prompt 数量，默认 4000
#   K_RATIO: PD scheduler 的 k_ratio，默认 0.8
#   MAX_CONCURRENCY: 最大并发数，默认等于 NUM_PROMPTS (模拟 offline)
#   BS_VALUES: max_num_seqs 扫描值列表
#   TB_VALUES: token_budget 扫描值列表

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载公共函数库
source "${SCRIPT_DIR}/common.sh"

# 实验参数
MAX_GPUS=${1:-4}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
NUM_PROMPTS=${NUM_PROMPTS:-4000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-512}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-8200}

# 网格搜索参数
BS_VALUES=(512 1024 1536 2048)
TB_VALUES=(6144 8192 10240 14336 16384)
SCENARIOS=("128 1024" "1024 128" "512 512")

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/grid_search_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env
WORKER_PIDS=()
setup_cleanup $BASE_PORT

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

# ========================================
# 生成实验队列
# ========================================
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
echo "  = ${#TB_VALUES[@]} TB × ${#BS_VALUES[@]} BS × ${#SCENARIOS[@]} Scenarios × 2 Schedulers"
echo ""

# 保存全局配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "grid_search",
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "k_ratio": ${K_RATIO},
    "random_range_ratio": ${RANDOM_RANGE_RATIO},
    "num_warmup_requests": ${NUM_WARMUP_REQUESTS},
    "bs_values": [$(echo "${BS_VALUES[*]}" | sed 's/ /, /g')],
    "tb_values": [$(echo "${TB_VALUES[*]}" | sed 's/ /, /g')],
    "scenarios": [
$(for s in "${SCENARIOS[@]}"; do
    read -r i o <<< "$s"
    echo "        {\"input_len\": $i, \"output_len\": $o},"
done | sed '$ s/,$//')
    ],
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

echo "配置已保存: ${OUTPUT_DIR}/experiment_config.json"
echo ""

# ========================================
# Worker 函数: 运行单个实验
# ========================================
run_experiment() {
    local gpu_id=$1
    local scheduler=$2
    local input_len=$3
    local output_len=$4
    local bs=$5
    local tb=$6

    local port=$((BASE_PORT + gpu_id))
    local scenario_name="in${input_len}_out${output_len}"

    # 检查端口
    check_port_available $port $gpu_id || return 1

    local result_dir="${OUTPUT_DIR}/tb${tb}/bs${bs}/${scenario_name}"
    local result_prefix="${scheduler}"
    mkdir -p "${result_dir}/logs"
    local log_file="${result_dir}/logs/${result_prefix}.log"

    echo "[GPU $gpu_id] 开始: ${scheduler} tb=${tb} bs=${bs} ${scenario_name}"

    # 保存实验配置
    cat > "${result_dir}/config_${result_prefix}.json" << EOF
{
    "scheduler": "${scheduler}",
    "max_num_seqs": ${bs},
    "max_num_batched_tokens": ${tb},
    "input_len": ${input_len},
    "output_len": ${output_len},
    "gpu_id": ${gpu_id},
    "port": ${port},
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "k_ratio": ${K_RATIO},
    "timestamp": "$(date -Iseconds)"
}
EOF

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

    # 启动服务
    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${result_prefix}_stats.json" \
    vllm serve $MODEL \
        --port $port \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs $bs \
        --max-num-batched-tokens $tb \
        > "$log_file" 2>&1 &

    local server_pid=$!

    # 等待服务启动
    if ! wait_for_server $port $server_pid; then
        echo "[GPU $gpu_id] 服务启动失败: tb=${tb} bs=${bs}"
        kill_server $server_pid
        return 1
    fi

    # 运行 benchmark
    vllm bench serve \
        --model $MODEL \
        --base-url "http://localhost:${port}" \
        --dataset-name random \
        --random-input-len $input_len \
        --random-output-len $output_len \
        --random-range-ratio $RANDOM_RANGE_RATIO \
        --num-prompts $NUM_PROMPTS \
        --num-warmups $NUM_WARMUP_REQUESTS \
        --request-rate inf \
        --max-concurrency $MAX_CONCURRENCY \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "bench_${result_prefix}.json" \
        >> "$log_file" 2>&1

    local bench_status=$?

    # 终止服务
    kill_server $server_pid

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] 完成: ${scheduler} tb=${tb} bs=${bs} ${scenario_name}"
    else
        echo "[GPU $gpu_id] 失败: ${scheduler} tb=${tb} bs=${bs} ${scenario_name}"
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

        if [ -z "$exp" ]; then
            echo "[GPU $gpu_id] 队列为空，退出"
            break
        fi

        IFS='|' read -r scheduler input_len output_len bs tb <<< "$exp"

        if run_experiment "$gpu_id" "$scheduler" "$input_len" "$output_len" "$bs" "$tb"; then
            update_progress "OK|${scheduler}|${input_len}|${output_len}|${bs}|${tb}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${scheduler}|${input_len}|${output_len}|${bs}|${tb}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
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

# 打印统计
print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"
echo ""
echo "运行分析脚本:"
echo "  python ${SCRIPT_DIR}/analyze_grid_search.py $OUTPUT_DIR"
