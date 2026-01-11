#!/bin/bash

MODEL="Qwen/Qwen3-8B"
NUM_PROMPTS=2000
PORT=8000
HOST="localhost"
OUTPUT_DIR="online_concurrency3"

# GPU 设置: 单卡 "5" 或 "6", 双卡 "5,6"
CUDA_DEVICES="5"

# 随机性控制: 0.0=固定长度, 0.5=±50%随机范围
# 实际长度范围: [length*(1-ratio), length*(1+ratio)]
RANDOM_RANGE_RATIO=0.5

# 创建输出目录
mkdir -p $OUTPUT_DIR

# scenario: input_len output_len
# 文件夹名自动生成为: in{input}_out{output}
scenarios=(
    "1024 128"
    "128 1024"
    "512 512"
)

# max_concurrency 设置
concurrencies=(512)

# 启动 vLLM server
start_server() {
    local chunked=$1
    echo "Starting vLLM server (chunked_prefill=$chunked)..."

    if [ "$chunked" = "true" ]; then
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
        VLLM_COLLECT_SCHEDULE_STATS=1 \
        vllm serve $MODEL \
            --port $PORT \
            --enable-chunked-prefill \
            --max-num-batched-tokens 4096 &
    else
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
        VLLM_COLLECT_SCHEDULE_STATS=1 \
        vllm serve $MODEL \
            --port $PORT &
    fi

    SERVER_PID=$!

    # 等待 server 启动
    echo "Waiting for server to start..."
    sleep 60

    # 检查 server 是否启动成功
    for i in {1..30}; do
        if curl -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        sleep 2
    done

    echo "Server failed to start!"
    return 1
}

# 停止 server
stop_server() {
    echo "Stopping server..."
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    # 确保端口释放
    sleep 5
}

# 运行 benchmark
run_benchmark() {
    local name=$1
    local input_len=$2
    local output_len=$3
    local max_concurrency=$4
    local prefix=$5

    echo "Running benchmark: ${prefix}_${name}_c${max_concurrency} (concurrency=${max_concurrency})"

    vllm bench serve \
        --model $MODEL \
        --base-url "http://${HOST}:${PORT}" \
        --dataset-name random \
        --random-input-len $input_len \
        --random-output-len $output_len \
        --random-range-ratio $RANDOM_RANGE_RATIO \
        --num-prompts $NUM_PROMPTS \
        --request-rate inf \
        --max-concurrency $max_concurrency \
        --save-result \
        --result-dir "${OUTPUT_DIR}" \
        --result-filename "results_${prefix}_${name}_c${max_concurrency}.json"
}

# 默认调度
echo "========== Default Scheduling =========="
for scenario in "${scenarios[@]}"; do
    read -r input_len output_len <<< "$scenario"
    name="in${input_len}_out${output_len}"

    start_server "false"

    for concurrency in "${concurrencies[@]}"; do
        run_benchmark "$name" "$input_len" "$output_len" "$concurrency" "default"
    done

    stop_server

    # 重命名 schedule_stats.json
    if [ -f "schedule_stats.json" ]; then
        mv schedule_stats.json "${OUTPUT_DIR}/schedule_stats_default_${name}.json"
    fi
done

# Chunked Prefill
echo "========== Chunked Prefill =========="
for scenario in "${scenarios[@]}"; do
    read -r input_len output_len <<< "$scenario"
    name="in${input_len}_out${output_len}"

    start_server "true"

    for concurrency in "${concurrencies[@]}"; do
        run_benchmark "$name" "$input_len" "$output_len" "$concurrency" "chunked"
    done

    stop_server

    # 重命名 schedule_stats.json
    if [ -f "schedule_stats.json" ]; then
        mv schedule_stats.json "${OUTPUT_DIR}/schedule_stats_chunked_${name}.json"
    fi
done

echo "All benchmarks completed!"
