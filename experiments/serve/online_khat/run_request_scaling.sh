#!/bin/bash

# Request 数量扩展实验脚本
# 验证假设: 随着 request 数量增加，PD scheduler 的稳态时间占比更大，优势更明显
#
# 实验设计:
#   场景: in128_out1024 (短输入长输出，decode heavy)
#   对比三条线:
#     1. PD scheduler (最优 TB/BS)
#     2. Baseline (与 PD 相同的 TB/BS)
#     3. Baseline (vLLM 默认 TB/BS)
#
# 用法: ./run_request_scaling.sh [MAX_GPUS]
#
# 环境变量:
#   PD_TB: PD scheduler 的最优 token budget，默认 10240
#   PD_BS: PD scheduler 的最优 max_num_seqs，默认 1536
#   DEFAULT_TB: vLLM 默认 token budget (不设置则使用 vLLM 默认)
#   DEFAULT_BS: vLLM 默认 max_num_seqs (不设置则使用 vLLM 默认)
#   REQUEST_COUNTS: request 数量列表，默认 "500 1000 2000 4000 8000 16000"
#   K_RATIO: PD scheduler 的 k_ratio，默认 0.8

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_GPUS=${1:-4}
GPU_MEM_THRESHOLD=${GPU_MEM_THRESHOLD:-10000}

# 实验参数
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-8200}

# 场景固定为 in128_out1024
INPUT_LEN=128
OUTPUT_LEN=1024

# PD scheduler 最优配置 (根据之前 grid search 结果设置)
PD_TB=${PD_TB:-10240}
PD_BS=${PD_BS:-1536}

# vLLM 默认配置 (空表示使用 vLLM 内置默认值)
DEFAULT_TB=${DEFAULT_TB:-""}
DEFAULT_BS=${DEFAULT_BS:-""}

# Request 数量扫描 (直接定义数组)
REQUEST_COUNTS=()
for ((i=5; i<=50; i+=5)); do
    REQUEST_COUNTS+=($((i * 1000)))
done

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/request_scaling_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 激活 vllm 环境
source /scratch/yuzhou/aproj/vllm/.venv/bin/activate

# 增加文件描述符限制
ulimit -n 65535 2>/dev/null || true

echo "========================================"
echo "Request 数量扩展实验"
echo "========================================"
echo "验证假设: 随着 request 数量增加，PD scheduler 优势更明显"
echo ""

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
echo "  场景: in${INPUT_LEN}_out${OUTPUT_LEN}"
echo "  MODEL: $MODEL"
echo "  K_RATIO: $K_RATIO"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo ""
echo "三条对比线:"
echo "  1. PD Scheduler:     TB=${PD_TB}, BS=${PD_BS}, k_ratio=${K_RATIO}"
echo "  2. Baseline (same):  TB=${PD_TB}, BS=${PD_BS}"
echo "  3. Baseline (default): TB=${DEFAULT_TB:-vLLM默认}, BS=${DEFAULT_BS:-vLLM默认}"
echo ""
echo "Request 数量: ${REQUEST_COUNTS[*]}"
echo ""

# ========================================
# 生成实验队列
# ========================================
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"

# 三种配置 × N 个 request 数量
for num_requests in "${REQUEST_COUNTS[@]}"; do
    # 1. PD scheduler (最优配置)
    echo "pd_optimal|${num_requests}|${PD_TB}|${PD_BS}|1" >> "$QUEUE_FILE"

    # 2. Baseline (与 PD 相同配置)
    echo "baseline_same|${num_requests}|${PD_TB}|${PD_BS}|0" >> "$QUEUE_FILE"

    # 3. Baseline (vLLM 默认配置)
    echo "baseline_default|${num_requests}|${DEFAULT_TB}|${DEFAULT_BS}|0" >> "$QUEUE_FILE"
done

TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
echo "总实验数: $TOTAL_EXPERIMENTS"
echo "  = 3 配置 × ${#REQUEST_COUNTS[@]} request数量"
echo ""

# 保存全局配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "request_scaling",
    "hypothesis": "PD scheduler advantage increases with more requests due to longer steady-state ratio",
    "scenario": {
        "input_len": ${INPUT_LEN},
        "output_len": ${OUTPUT_LEN}
    },
    "configurations": {
        "pd_optimal": {
            "scheduler": "pd",
            "tb": ${PD_TB},
            "bs": ${PD_BS},
            "k_ratio": ${K_RATIO}
        },
        "baseline_same": {
            "scheduler": "baseline",
            "tb": ${PD_TB},
            "bs": ${PD_BS}
        },
        "baseline_default": {
            "scheduler": "baseline",
            "tb": "${DEFAULT_TB:-null}",
            "bs": "${DEFAULT_BS:-null}"
        }
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
# Worker 函数: 运行单个实验
# ========================================
run_experiment() {
    local gpu_id=$1
    local config_name=$2    # pd_optimal, baseline_same, baseline_default
    local num_requests=$3
    local tb=$4
    local bs=$5
    local use_pd=$6         # 1 or 0

    local port=$((BASE_PORT + gpu_id))

    # 结果目录
    local result_dir="${OUTPUT_DIR}/requests_${num_requests}"
    mkdir -p "${result_dir}/logs"

    local log_file="${result_dir}/logs/${config_name}.log"

    echo "[GPU $gpu_id] 开始: ${config_name} requests=${num_requests}"

    # 构建 vllm serve 参数
    local serve_args="--gpu-memory-utilization 0.9"
    if [ -n "$tb" ]; then
        serve_args="$serve_args --max-num-batched-tokens $tb"
    fi
    if [ -n "$bs" ]; then
        serve_args="$serve_args --max-num-seqs $bs"
    fi

    # 保存实验配置
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
        export VLLM_USE_PD_SCHEDULER=1
        export VLLM_PD_K_RATIO=$K_RATIO
    else
        export VLLM_USE_PD_SCHEDULER=0
        unset VLLM_PD_K_RATIO
    fi

    # 启动 vllm serve
    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${config_name}_stats.json" \
    vllm serve $MODEL \
        --port $port \
        $serve_args \
        > "$log_file" 2>&1 &

    local server_pid=$!

    # 等待服务启动
    local max_wait=180
    local wait_count=0
    while [ $wait_count -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            break
        fi
        if ! kill -0 $server_pid 2>/dev/null; then
            echo "[GPU $gpu_id] 服务进程意外退出: ${config_name}"
            return 1
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done

    if [ $wait_count -ge $max_wait ]; then
        echo "[GPU $gpu_id] 服务启动超时: ${config_name}"
        kill -9 $server_pid 2>/dev/null || true
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

    # 终止服务
    kill -INT $server_pid 2>/dev/null || true
    sleep 2
    kill -9 $server_pid 2>/dev/null || true

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

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment)

        if [ -z "$exp" ]; then
            echo "[GPU $gpu_id] 队列为空，退出"
            break
        fi

        # 解析: config_name|num_requests|tb|bs|use_pd
        IFS='|' read -r config_name num_requests tb bs use_pd <<< "$exp"

        if run_experiment "$gpu_id" "$config_name" "$num_requests" "$tb" "$bs" "$use_pd"; then
            update_progress "OK|${config_name}|${num_requests}"
        else
            update_progress "FAIL|${config_name}|${num_requests}"
        fi
    done
}

# ========================================
# 主流程
# ========================================
echo "开始并行执行..."
echo "========================================"

> "$PROGRESS_FILE"

WORKER_PIDS=()
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

echo ""
echo "========================================"
echo "所有实验完成！"
echo "========================================"

TOTAL_COMPLETED=$(wc -l < "$PROGRESS_FILE")
TOTAL_OK=$(grep -c "^OK|" "$PROGRESS_FILE" || true)
TOTAL_FAIL=$(grep -c "^FAIL|" "$PROGRESS_FILE" || true)

echo "总计: $TOTAL_COMPLETED / $TOTAL_EXPERIMENTS"
echo "成功: $TOTAL_OK"
echo "失败: $TOTAL_FAIL"
echo ""
echo "结果目录: $OUTPUT_DIR"
echo ""
echo "运行分析脚本:"
echo "  python ${SCRIPT_DIR}/analyze_request_scaling.py $OUTPUT_DIR"
