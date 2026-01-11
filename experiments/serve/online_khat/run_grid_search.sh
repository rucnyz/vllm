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
#   MAX_CONCURRENCY: 最大并发数，默认 2048
#   BS_VALUES: max_num_seqs 扫描值列表
#   TB_VALUES: token_budget 扫描值列表

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_GPUS=${1:-4}
GPU_MEM_THRESHOLD=${GPU_MEM_THRESHOLD:-10000}

# 实验参数
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
NUM_PROMPTS=${NUM_PROMPTS:-4000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-8200}

# 网格搜索参数 (直接定义数组，避免解析问题)
BS_VALUES=(512 1024 1536 2048)
TB_VALUES=(6144 8192 10240 14336 16384)

# Scenarios
SCENARIOS=("128 1024" "1024 128" "512 512")

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/grid_search_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 激活 vllm 环境
source /scratch/yuzhou/aproj/vllm/.venv/bin/activate

# 增加文件描述符限制
ulimit -n 65535 2>/dev/null || true

echo "========================================"
echo "TB × BS 网格搜索实验"
echo "========================================"

# ========================================
# 检测可用 GPU
# ========================================
detect_available_gpus() {
    local available=()

    while IFS=, read -r gpu_id name mem_total mem_used; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' MiB')

        if [ "$mem_used" -lt "$GPU_MEM_THRESHOLD" ]; then
            available+=("$gpu_id")
        fi
    done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null)

    echo "${available[@]}"
}

AVAILABLE_GPUS=($(detect_available_gpus))
NUM_AVAILABLE=${#AVAILABLE_GPUS[@]}

if [ "$NUM_AVAILABLE" -eq 0 ]; then
    echo "错误: 没有可用的 GPU (内存使用 < ${GPU_MEM_THRESHOLD} MiB)"
    exit 1
fi

# 使用 min(available, MAX_GPUS) 张卡
if [ "$NUM_AVAILABLE" -gt "$MAX_GPUS" ]; then
    GPUS_TO_USE=("${AVAILABLE_GPUS[@]:0:$MAX_GPUS}")
else
    GPUS_TO_USE=("${AVAILABLE_GPUS[@]}")
fi

NUM_GPUS=${#GPUS_TO_USE[@]}

echo "检测到 $NUM_AVAILABLE 张可用 GPU: ${AVAILABLE_GPUS[*]}"
echo "将使用 $NUM_GPUS 张 GPU: ${GPUS_TO_USE[*]}"
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
# 生成实验队列 (TB × BS × Scenario × Scheduler)
# ========================================
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"

for tb in "${TB_VALUES[@]}"; do
    for bs in "${BS_VALUES[@]}"; do
        for scenario in "${SCENARIOS[@]}"; do
            read -r input_len output_len <<< "$scenario"
            # baseline
            echo "baseline|${input_len}|${output_len}|${bs}|${tb}" >> "$QUEUE_FILE"
            # pd
            echo "pd|${input_len}|${output_len}|${bs}|${tb}" >> "$QUEUE_FILE"
        done
    done
done

TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
echo "总实验数: $TOTAL_EXPERIMENTS"
echo "  = ${#TB_VALUES[@]} TB × ${#BS_VALUES[@]} BS × ${#SCENARIOS[@]} Scenarios × 2 Schedulers"
echo ""
echo "实验队列: $QUEUE_FILE"

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
    "gpus_used": [${GPUS_TO_USE[*]}],
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
    local scheduler=$2     # baseline or pd
    local input_len=$3
    local output_len=$4
    local bs=$5
    local tb=$6

    local port=$((BASE_PORT + gpu_id))
    local scenario_name="in${input_len}_out${output_len}"

    # 结果目录: grid_search/tb{TB}/bs{BS}/{scenario}/
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

    # 启动 vllm serve
    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${result_prefix}_stats.json" \
    vllm serve $MODEL \
        --port $port \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs $bs \
        --max-num-batched-tokens $tb \
        > "$log_file" 2>&1 &

    local server_pid=$!

    # 等待服务启动
    local max_wait=180
    local wait_count=0
    while [ $wait_count -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            break
        fi
        # 检查进程是否还存活
        if ! kill -0 $server_pid 2>/dev/null; then
            echo "[GPU $gpu_id] 服务进程意外退出: tb=${tb} bs=${bs}"
            return 1
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done

    if [ $wait_count -ge $max_wait ]; then
        echo "[GPU $gpu_id] 服务启动超时: tb=${tb} bs=${bs}"
        kill -9 $server_pid 2>/dev/null || true
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
    kill -INT $server_pid 2>/dev/null || true
    sleep 2
    kill -9 $server_pid 2>/dev/null || true

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

# 获取下一个实验 (带锁)
get_next_experiment() {
    (
        flock -x 200

        local exp=$(head -n 1 "$QUEUE_FILE" 2>/dev/null)

        if [ -n "$exp" ]; then
            tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp"
            mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
            echo "$exp"
        fi
    ) 200>"$LOCK_FILE"
}

# 更新进度
update_progress() {
    local status=$1
    (
        flock -x 200
        echo "$status" >> "$PROGRESS_FILE"
        local completed=$(wc -l < "$PROGRESS_FILE")
        local remaining=$((TOTAL_EXPERIMENTS - completed))
        echo "进度: $completed / $TOTAL_EXPERIMENTS (剩余 $remaining)"
    ) 200>"$LOCK_FILE"
}

# GPU worker 进程
gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment)

        if [ -z "$exp" ]; then
            echo "[GPU $gpu_id] 队列为空，退出"
            break
        fi

        # 解析实验参数: scheduler|input_len|output_len|bs|tb
        IFS='|' read -r scheduler input_len output_len bs tb <<< "$exp"

        # 运行实验
        if run_experiment "$gpu_id" "$scheduler" "$input_len" "$output_len" "$bs" "$tb"; then
            update_progress "OK|${scheduler}|${input_len}|${output_len}|${bs}|${tb}"
        else
            update_progress "FAIL|${scheduler}|${input_len}|${output_len}|${bs}|${tb}"
        fi
    done
}

# ========================================
# 主流程
# ========================================
echo "开始并行执行..."
echo "========================================"

# 清空进度文件
> "$PROGRESS_FILE"

# 启动 GPU workers
WORKER_PIDS=()
for gpu_id in "${GPUS_TO_USE[@]}"; do
    gpu_worker "$gpu_id" &
    WORKER_PIDS+=($!)
    echo "启动 GPU $gpu_id worker (PID: ${WORKER_PIDS[-1]})"
done

echo ""
echo "所有 worker 已启动，等待实验完成..."
echo "可以用以下命令监控进度:"
echo "  watch -n 5 'wc -l ${OUTPUT_DIR}/progress.txt'"
echo ""

# 等待所有 workers 完成
for pid in "${WORKER_PIDS[@]}"; do
    wait $pid
done

echo ""
echo "========================================"
echo "所有实验完成！"
echo "========================================"

# 统计结果
TOTAL_COMPLETED=$(wc -l < "$PROGRESS_FILE")
TOTAL_OK=$(grep -c "^OK|" "$PROGRESS_FILE" || true)
TOTAL_FAIL=$(grep -c "^FAIL|" "$PROGRESS_FILE" || true)

echo "总计: $TOTAL_COMPLETED / $TOTAL_EXPERIMENTS"
echo "成功: $TOTAL_OK"
echo "失败: $TOTAL_FAIL"
echo ""
echo "结果目录: $OUTPUT_DIR"
echo ""
echo "目录结构:"
echo "  $OUTPUT_DIR/"
echo "  ├── experiment_config.json"
echo "  ├── tb4096/"
echo "  │   ├── bs256/"
echo "  │   │   ├── in128_out1024/"
echo "  │   │   │   ├── bench_baseline.json"
echo "  │   │   │   ├── bench_pd.json"
echo "  │   │   │   ├── config_*.json"
echo "  │   │   │   └── logs/"
echo "  │   │   ├── in1024_out128/"
echo "  │   │   └── in512_out512/"
echo "  │   ├── bs512/"
echo "  │   └── ..."
echo "  ├── tb6144/"
echo "  └── ..."
echo ""
echo "运行分析脚本:"
echo "  python ${SCRIPT_DIR}/analyze_grid_search.py $OUTPUT_DIR"
