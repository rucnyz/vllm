#!/bin/bash

# Output Length 方差实验脚本
# 验证假设: Output length 方差越大, PD scheduler 表现越好
#
# 用法: ./run_variance_experiment.sh [MAX_GPUS]
#
# 实验设计:
#   - 三个 scenario (in1024_out128, in128_out1024, in512_out512)
#   - 不同的 RANDOM_RANGE_RATIO (0, 0.25, 0.5, 0.75, 0.9)
#   - 对比三种配置: PD optimal, Baseline same, Baseline default

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载公共函数库
source "${SCRIPT_DIR}/../common.sh"

# 实验参数
MAX_GPUS=${1:-4}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
NUM_PROMPTS=${NUM_PROMPTS:-5000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-8200}

# RANDOM_RANGE_RATIO 值
RATIO_VALUES=(0 0.25 0.5 0.75 0.9)

# 从 grid search 得到的最优配置 (TB, BS)
declare -A OPTIMAL_PD_TB
declare -A OPTIMAL_PD_BS

OPTIMAL_PD_TB["in1024_out128"]=16384
OPTIMAL_PD_BS["in1024_out128"]=1024

OPTIMAL_PD_TB["in128_out1024"]=10240
OPTIMAL_PD_BS["in128_out1024"]=1536

OPTIMAL_PD_TB["in512_out512"]=10240
OPTIMAL_PD_BS["in512_out512"]=1024

# Scenarios: "input_len output_len scenario_name"
SCENARIOS=("1024 128 in1024_out128" "128 1024 in128_out1024" "512 512 in512_out512")

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/variance_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env
WORKER_PIDS=()
setup_cleanup $BASE_PORT

echo "========================================"
echo "Output Length 方差实验"
echo "========================================"
echo "假设: Output length 方差越大, PD scheduler Phase 2 触发越频繁, 性能越好"
echo ""

# 检测并选择 GPU
select_gpus $MAX_GPUS
echo ""
echo "实验配置:"
echo "  MODEL: $MODEL"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  RATIO_VALUES: ${RATIO_VALUES[*]}"
echo "  SCENARIOS: ${#SCENARIOS[@]} 个"
echo ""

# ========================================
# 生成实验队列
# ========================================
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"

for ratio in "${RATIO_VALUES[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
        read -r input_len output_len scenario_name <<< "$scenario"

        tb=${OPTIMAL_PD_TB[$scenario_name]}
        bs=${OPTIMAL_PD_BS[$scenario_name]}

        echo "pd_optimal|${input_len}|${output_len}|${scenario_name}|${bs}|${tb}|${ratio}" >> "$QUEUE_FILE"
        echo "baseline_same|${input_len}|${output_len}|${scenario_name}|${bs}|${tb}|${ratio}" >> "$QUEUE_FILE"
        echo "baseline_default|${input_len}|${output_len}|${scenario_name}|null|null|${ratio}" >> "$QUEUE_FILE"
    done
done

TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
echo "总实验数: $TOTAL_EXPERIMENTS"
echo "  = ${#RATIO_VALUES[@]} ratios × ${#SCENARIOS[@]} scenarios × 3 configs"
echo ""

# 保存全局配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "variance_experiment",
    "hypothesis": "Higher output length variance leads to better PD scheduler performance",
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "k_ratio": ${K_RATIO},
    "ratio_values": [$(echo "${RATIO_VALUES[*]}" | sed 's/ /, /g')],
    "scenarios": [
$(for s in "${SCENARIOS[@]}"; do
    read -r i o n <<< "$s"
    echo "        {\"input_len\": $i, \"output_len\": $o, \"name\": \"$n\", \"optimal_tb\": ${OPTIMAL_PD_TB[$n]}, \"optimal_bs\": ${OPTIMAL_PD_BS[$n]}},"
done | sed '$ s/,$//')
    ],
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
    local input_len=$3
    local output_len=$4
    local scenario_name=$5
    local bs=$6
    local tb=$7
    local ratio=$8

    local port=$((BASE_PORT + gpu_id))

    check_port_available $port $gpu_id || return 1

    local result_dir="${OUTPUT_DIR}/ratio_${ratio}/${scenario_name}"
    mkdir -p "${result_dir}/logs"
    local log_file="${result_dir}/logs/${config_name}.log"

    echo "[GPU $gpu_id] 开始: ${config_name} ratio=${ratio} ${scenario_name}"

    # 保存配置
    cat > "${result_dir}/config_${config_name}.json" << EOF
{
    "config_name": "${config_name}",
    "max_num_seqs": ${bs},
    "max_num_batched_tokens": ${tb},
    "input_len": ${input_len},
    "output_len": ${output_len},
    "random_range_ratio": ${ratio},
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

    # 构建 serve 参数
    local serve_args="$MODEL --port $port --gpu-memory-utilization 0.9"
    [ "$bs" != "null" ] && serve_args="$serve_args --max-num-seqs $bs"
    [ "$tb" != "null" ] && serve_args="$serve_args --max-num-batched-tokens $tb"

    if [ "$config_name" = "pd_optimal" ]; then
        # 使用 PD 调度器 (比例模式)
        export VLLM_USE_PD_SCHEDULER=1
        export VLLM_PD_K_MODE=ratio
        export VLLM_PD_K_RATIO=$K_RATIO
    else
        # 使用默认 vLLM 调度器
        export VLLM_USE_PD_SCHEDULER=0
    fi

    # 启动服务
    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${config_name}_stats.json" \
    vllm serve $serve_args > "$log_file" 2>&1 &

    local server_pid=$!

    if ! wait_for_server $port $server_pid; then
        echo "[GPU $gpu_id] 服务启动失败"
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
        --random-range-ratio $ratio \
        --num-prompts $NUM_PROMPTS \
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
        echo "[GPU $gpu_id] 完成: ${config_name} ratio=${ratio} ${scenario_name}"
    else
        echo "[GPU $gpu_id] 失败: ${config_name} ratio=${ratio} ${scenario_name}"
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

        IFS='|' read -r config_name input_len output_len scenario_name bs tb ratio <<< "$exp"

        if run_experiment "$gpu_id" "$config_name" "$input_len" "$output_len" "$scenario_name" "$bs" "$tb" "$ratio"; then
            update_progress "OK|${config_name}|${scenario_name}|${ratio}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${config_name}|${scenario_name}|${ratio}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
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
echo "  python ${SCRIPT_DIR}/analyze_variance_experiment.py $OUTPUT_DIR"
