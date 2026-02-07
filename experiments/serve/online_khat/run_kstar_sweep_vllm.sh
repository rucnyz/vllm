#!/bin/bash

# K* 参数扫描脚本 (使用 vllm bench serve)
# 按 scenario 分类保存结果

set -e

PORT=8124
MODEL="Qwen/Qwen3-8B"
MAX_BATCH_SIZE=${1:-64}
NUM_PROMPTS=2000
CUDA_DEVICES="5"

# 随机性控制
RANDOM_RANGE_RATIO=0.5

# scenario: name input_len output_len
scenarios=(
    "long_in_short_out 1024 128"
    "short_in_long_out 128 1024"
    "balanced 512 512"
)

# max_concurrency 设置
MAX_CONCURRENCY=512

# K_STAR 值列表：动态生成到 MAX_BATCH_SIZE
K_STAR_VALUES=(1)
for ((i=4; i<=MAX_BATCH_SIZE; i+=4)); do
    K_STAR_VALUES+=($i)
done

# 等待服务就绪的函数
wait_for_server() {
    echo "等待 vllm 服务启动..."
    local max_attempts=120
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "服务已就绪！"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    echo "错误：服务启动超时"
    return 1
}

# 终止 vllm 服务的函数
kill_vllm_server() {
    echo "终止 vllm 服务..."

    local pid=$(lsof -t -i:${PORT} 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "发送 SIGINT 到 PID $pid..."
        kill -INT $pid 2>/dev/null || true

        local wait_count=0
        while [ $wait_count -lt 30 ]; do
            if ! kill -0 $pid 2>/dev/null; then
                echo "进程已正常退出"
                break
            fi
            sleep 1
            wait_count=$((wait_count + 1))
        done

        if kill -0 $pid 2>/dev/null; then
            echo "进程未响应，强制终止..."
            kill -9 $pid 2>/dev/null || true
            sleep 2
        fi
    fi

    pkill -INT -f "vllm serve" 2>/dev/null || true
    sleep 3
    echo "服务已终止"
}

# 运行 benchmark 的函数
run_benchmark() {
    local scenario_name=$1
    local input_len=$2
    local output_len=$3
    local result_prefix=$4
    local result_dir=$5

    echo "运行 vllm bench serve (${result_prefix})..."
    vllm bench serve \
        --model $MODEL \
        --base-url "http://localhost:${PORT}" \
        --dataset-name random \
        --random-input-len $input_len \
        --random-output-len $output_len \
        --random-range-ratio $RANDOM_RANGE_RATIO \
        --num-prompts $NUM_PROMPTS \
        --request-rate inf \
        --max-concurrency $MAX_CONCURRENCY \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "bench_${result_prefix}.json"
}

# ========================================
# 主循环：按 scenario 分类
# ========================================

for scenario in "${scenarios[@]}"; do
    read -r scenario_name input_len output_len <<< "$scenario"

    echo ""
    echo "========================================"
    echo "Scenario: $scenario_name (input=$input_len, output=$output_len)"
    echo "========================================"

    # 创建 scenario 目录
    RESULT_DIR="${scenario_name}"
    LOG_DIR="${RESULT_DIR}/logs"
    mkdir -p "$RESULT_DIR"
    mkdir -p "$LOG_DIR"

    # ----------------------------------------
    # Baseline 测试
    # ----------------------------------------
    echo ""
    echo "--- Baseline (默认调度器) ---"

    kill_vllm_server

    echo "启动 vllm serve (Baseline)..."
    VLLM_COLLECT_SCHEDULE_STATS=1 \
    VLLM_SCHEDULE_STATS_FILE="${RESULT_DIR}/baseline.json" \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    VLLM_USE_PD_SCHEDULER=0 \
    vllm serve $MODEL \
        --port $PORT \
        --max-num-seqs $MAX_BATCH_SIZE \
        --gpu-memory-utilization 0.9 \
        > "$LOG_DIR/vllm_baseline.log" 2>&1 &

    VLLM_PID=$!

    if wait_for_server; then
        run_benchmark "$scenario_name" "$input_len" "$output_len" "baseline" "$RESULT_DIR"
        kill_vllm_server
        echo "Baseline 完成"
    else
        echo "Baseline 跳过 (服务启动失败)"
        kill_vllm_server
    fi

    # ----------------------------------------
    # Fixed K* 扫描
    # ----------------------------------------
    for k_star in "${K_STAR_VALUES[@]}"; do
        echo ""
        echo "--- Fixed K* = $k_star ---"

        kill_vllm_server

        echo "启动 vllm serve (K*=$k_star)..."
        VLLM_COLLECT_SCHEDULE_STATS=1 \
        VLLM_SCHEDULE_STATS_FILE="${RESULT_DIR}/fixed${k_star}.json" \
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
        VLLM_USE_PD_SCHEDULER=1 \
        VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
        VLLM_PD_K_STAR=$k_star \
        vllm serve $MODEL \
            --port $PORT \
            --max-num-seqs $MAX_BATCH_SIZE \
            --gpu-memory-utilization 0.9 \
            > "$LOG_DIR/vllm_k${k_star}.log" 2>&1 &

        VLLM_PID=$!

        if wait_for_server; then
            run_benchmark "$scenario_name" "$input_len" "$output_len" "fixed${k_star}" "$RESULT_DIR"
            kill_vllm_server
            echo "K*=$k_star 完成"
        else
            echo "K*=$k_star 跳过 (服务启动失败)"
            kill_vllm_server
        fi
    done

    echo ""
    echo "Scenario $scenario_name 全部完成！"
    echo "结果保存在: $RESULT_DIR/"
done

echo ""
echo "========================================"
echo "所有测试完成！"
echo "========================================"
echo "结果目录结构:"
for scenario in "${scenarios[@]}"; do
    read -r name _ _ <<< "$scenario"
    echo "  - ${name}/"
    echo "      baseline.json, fixed*.json"
    echo "      bench_*.json"
    echo "      logs/"
done
