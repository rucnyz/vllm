#!/bin/bash

# K* 参数扫描脚本
# 自动遍历不同的 K_STAR 值，启动 vllm serve，运行 benchmark，然后终止服务

set -e

PORT=8124
API_KEY="7355608"
MODEL="Qwen/Qwen3-8B"

# K_STAR 值列表：24, 28, 32, 36, ..., 128
K_STAR_VALUES=(24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96 100 104 108 112 116 120 124 128)

# 等待服务就绪的函数
wait_for_server() {
    echo "等待 vllm 服务启动..."
    local max_attempts=120  # 最多等待 2 分钟
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

# 终止 vllm 服务的函数（优雅退出以保存结果）
kill_vllm_server() {
    echo "终止 vllm 服务..."

    # 先发送 SIGINT (相当于 Ctrl+C)，让程序优雅退出并保存数据
    local pid=$(lsof -t -i:${PORT} 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "发送 SIGINT 到 PID $pid..."
        kill -INT $pid 2>/dev/null || true

        # 等待进程退出（最多等 30 秒）
        local wait_count=0
        while [ $wait_count -lt 30 ]; do
            if ! kill -0 $pid 2>/dev/null; then
                echo "进程已正常退出"
                break
            fi
            sleep 1
            wait_count=$((wait_count + 1))
        done

        # 如果还没退出，强制终止
        if kill -0 $pid 2>/dev/null; then
            echo "进程未响应，强制终止..."
            kill -9 $pid 2>/dev/null || true
            sleep 2
        fi
    fi

    # 备用方案
    pkill -INT -f "vllm serve" 2>/dev/null || true
    sleep 3
    echo "服务已终止"
}

# 主循环
for k_star in "${K_STAR_VALUES[@]}"; do
    echo ""
    echo "========================================"
    echo "开始测试 K_STAR = $k_star"
    echo "========================================"

    # 确保没有残留的服务
    kill_vllm_server

    # 启动 vllm serve（后台运行）
    echo "启动 vllm serve (K_STAR=$k_star)..."
    VLLM_COLLECT_SCHEDULE_STATS=1 \
    VLLM_SCHEDULE_STATS_FILE="results/fixed${k_star}.json" \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
    VLLM_PD_K_STAR=$k_star \
    vllm serve $MODEL \
        --port $PORT \
        --max-num-seqs 128 \
        --gpu-memory-utilization 0.9 \
        --api-key "$API_KEY" \
        > "results/vllm_log_k${k_star}.log" 2>&1 &

    VLLM_PID=$!
    echo "vllm PID: $VLLM_PID"

    # 等待服务就绪
    if ! wait_for_server; then
        echo "跳过 K_STAR=$k_star (服务启动失败)"
        kill_vllm_server
        continue
    fi

    # 运行 benchmark
    echo "运行 genai-bench (K_STAR=$k_star)..."
    genai-bench benchmark \
        --api-backend vllm \
        --api-key "$API_KEY" \
        --api-base "http://localhost:${PORT}" \
        --api-model-name "$MODEL" \
        --model-tokenizer "$MODEL" \
        --task text-to-text \
        --experiment-base-dir "./experiment_results/genai/fixed_k${k_star}" \
        --dataset-path ./experiments/serve/alpaca_prompts.csv \
        --dataset-prompt-column prompt \
        --max-time-per-run 60 \
        --max-requests-per-run 500 \
        --num-concurrency 128

    echo "Benchmark 完成 (K_STAR=$k_star)"

    # 终止服务
    kill_vllm_server

    echo "K_STAR=$k_star 测试完成！"
done

echo ""
echo "========================================"
echo "所有测试完成！"
echo "========================================"
echo "结果保存在:"
echo "  - 调度统计: results/fixed*.json"
echo "  - vllm 日志: results/vllm_log_k*.log"
echo "  - Benchmark 结果: experiment_results/genai/fixed_k*/"