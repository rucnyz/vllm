#!/bin/bash

# K* 参数扫描脚本
# 自动遍历不同的 K_STAR 值，启动 vllm serve，运行 benchmark，然后终止服务

set -e

PORT=8124
API_KEY="7355608"
MODEL="Qwen/Qwen3-8B"
MAX_BATCH_SIZE=${1:-64}  # 通过第一个参数控制，默认值为 128
LOG_DIR="results/logs"    # 日志文件夹

# 创建目录
mkdir -p "$LOG_DIR"
mkdir -p "results"

# K_STAR 值列表：动态生成到 MAX_BATCH_SIZE
K_STAR_VALUES=(1)
for ((i=4; i<=MAX_BATCH_SIZE; i+=4)); do
    K_STAR_VALUES+=($i)
done

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

# ========================================
# Baseline 测试（使用默认调度器）
# ========================================
echo ""
echo "========================================"
echo "开始测试 Baseline (默认调度器)"
echo "========================================"

# 确保没有残留的服务
kill_vllm_server

# 启动 vllm serve（后台运行）- 不使用 P/D scheduler
echo "启动 vllm serve (Baseline - 默认调度器)..."
VLLM_COLLECT_SCHEDULE_STATS=1 \
VLLM_SCHEDULE_STATS_FILE="results/baseline.json" \
CUDA_VISIBLE_DEVICES=5 \
VLLM_USE_PD_SCHEDULER=0 \
vllm serve $MODEL \
    --port $PORT \
    --max-num-seqs $MAX_BATCH_SIZE \
    --gpu-memory-utilization 0.9 \
    --api-key "$API_KEY" \
    > "$LOG_DIR/vllm_log_baseline.log" 2>&1 &

VLLM_PID=$!
echo "vllm PID: $VLLM_PID"

# 等待服务就绪
if ! wait_for_server; then
    echo "Baseline 测试跳过 (服务启动失败)"
    kill_vllm_server
else
    # 运行 benchmark
    echo "运行 genai-bench (Baseline)..."
    genai-bench benchmark \
        --api-backend vllm \
        --api-key "$API_KEY" \
        --api-base "http://localhost:${PORT}" \
        --api-model-name "$MODEL" \
        --model-tokenizer "$MODEL" \
        --task text-to-text \
        --experiment-base-dir "./experiment_results/genai/baseline" \
        --dataset-path ../alpaca_prompts.csv \
        --dataset-prompt-column prompt \
        --max-time-per-run 60 \
        --max-requests-per-run 500 \
        --num-concurrency $MAX_BATCH_SIZE

    echo "Benchmark 完成 (Baseline)"

    # 终止服务
    kill_vllm_server

    echo "Baseline 测试完成！"
fi

# ========================================
# K* 参数扫描
# ========================================

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
    CUDA_VISIBLE_DEVICES=5 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
    VLLM_PD_K_STAR=$k_star \
    vllm serve $MODEL \
        --port $PORT \
        --max-num-seqs $MAX_BATCH_SIZE \
        --gpu-memory-utilization 0.9 \
        --api-key "$API_KEY" \
        > "$LOG_DIR/vllm_log_k${k_star}.log" 2>&1 &

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
        --dataset-path ../alpaca_prompts.csv \
        --dataset-prompt-column prompt \
        --max-time-per-run 60 \
        --max-requests-per-run 500 \
        --num-concurrency $MAX_BATCH_SIZE

    echo "Benchmark 完成 (K_STAR=$k_star)"

    # 终止服务
    kill_vllm_server

    echo "K_STAR=$k_star 测试完成！"
done

# ========================================
# Dynamic K* 测试（在线学习自适应 k*）
# ========================================

# Dynamic K* 参数配置
INITIAL_K_STAR=64           # 初始 k* 值
EMA_ALPHA=0.2               # EMA 平滑因子
UPDATE_INTERVAL=32          # 每隔多少请求更新一次 k*

export VLLM_PD_ALPHA_P=0.002528784356021418
export VLLM_PD_BETA_P=6.498792400220424e-06
export VLLM_PD_ALPHA_D=0.001
export VLLM_PD_BETA_D=0.00023557651251992446

# 测试不同的 EMA alpha 值
EMA_ALPHA_VALUES=(0.1 0.2 0.3 0.5)

for ema_alpha in "${EMA_ALPHA_VALUES[@]}"; do
    echo ""
    echo "========================================"
    echo "开始测试 Dynamic K* (EMA_ALPHA=$ema_alpha)"
    echo "========================================"

    # 确保没有残留的服务
    kill_vllm_server

    # 启动 vllm serve（后台运行）- 启用 dynamic k*
    echo "启动 vllm serve (Dynamic K*, EMA_ALPHA=$ema_alpha)..."
    VLLM_COLLECT_SCHEDULE_STATS=1 \
    VLLM_SCHEDULE_STATS_FILE="results/dynamic_ema${ema_alpha}.json" \
    CUDA_VISIBLE_DEVICES=5 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=1 \
    VLLM_PD_K_STAR=$INITIAL_K_STAR \
    VLLM_PD_EMA_ALPHA=$ema_alpha \
    VLLM_PD_UPDATE_INTERVAL=$UPDATE_INTERVAL \
    vllm serve $MODEL \
        --port $PORT \
        --max-num-seqs $MAX_BATCH_SIZE \
        --gpu-memory-utilization 0.9 \
        --api-key "$API_KEY" \
        > "$LOG_DIR/vllm_log_dynamic_ema${ema_alpha}.log" 2>&1 &

    VLLM_PID=$!
    echo "vllm PID: $VLLM_PID"

    # 等待服务就绪
    if ! wait_for_server; then
        echo "跳过 Dynamic K* EMA_ALPHA=$ema_alpha (服务启动失败)"
        kill_vllm_server
        continue
    fi

    # 运行 benchmark
    echo "运行 genai-bench (Dynamic K*, EMA_ALPHA=$ema_alpha)..."
    genai-bench benchmark \
        --api-backend vllm \
        --api-key "$API_KEY" \
        --api-base "http://localhost:${PORT}" \
        --api-model-name "$MODEL" \
        --model-tokenizer "$MODEL" \
        --task text-to-text \
        --experiment-base-dir "./experiment_results/genai/dynamic_ema${ema_alpha}" \
        --dataset-path ../alpaca_prompts.csv \
        --dataset-prompt-column prompt \
        --max-time-per-run 60 \
        --max-requests-per-run 500 \
        --num-concurrency $MAX_BATCH_SIZE

    echo "Benchmark 完成 (Dynamic K*, EMA_ALPHA=$ema_alpha)"

    # 终止服务
    kill_vllm_server

    echo "Dynamic K* EMA_ALPHA=$ema_alpha 测试完成！"
done

# ========================================
# 测试不同的 Update Interval
# ========================================
# 0.5x, 1x, 2x MAX_BATCH_SIZE
UPDATE_INTERVAL_VALUES=($((MAX_BATCH_SIZE / 2)) $MAX_BATCH_SIZE $((MAX_BATCH_SIZE * 2)))

for update_interval in "${UPDATE_INTERVAL_VALUES[@]}"; do
    echo ""
    echo "========================================"
    echo "开始测试 Dynamic K* (UPDATE_INTERVAL=$update_interval)"
    echo "========================================"

    # 确保没有残留的服务
    kill_vllm_server

    # 启动 vllm serve（后台运行）
    echo "启动 vllm serve (Dynamic K*, UPDATE_INTERVAL=$update_interval)..."
    VLLM_COLLECT_SCHEDULE_STATS=1 \
    VLLM_SCHEDULE_STATS_FILE="results/dynamic_interval${update_interval}.json" \
    CUDA_VISIBLE_DEVICES=5 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=1 \
    VLLM_PD_K_STAR=$INITIAL_K_STAR \
    VLLM_PD_EMA_ALPHA=0.2 \
    VLLM_PD_UPDATE_INTERVAL=$update_interval \
    vllm serve $MODEL \
        --port $PORT \
        --max-num-seqs $MAX_BATCH_SIZE \
        --gpu-memory-utilization 0.9 \
        --api-key "$API_KEY" \
        > "$LOG_DIR/vllm_log_dynamic_interval${update_interval}.log" 2>&1 &

    VLLM_PID=$!
    echo "vllm PID: $VLLM_PID"

    # 等待服务就绪
    if ! wait_for_server; then
        echo "跳过 Dynamic K* UPDATE_INTERVAL=$update_interval (服务启动失败)"
        kill_vllm_server
        continue
    fi

    # 运行 benchmark
    echo "运行 genai-bench (Dynamic K*, UPDATE_INTERVAL=$update_interval)..."
    genai-bench benchmark \
        --api-backend vllm \
        --api-key "$API_KEY" \
        --api-base "http://localhost:${PORT}" \
        --api-model-name "$MODEL" \
        --model-tokenizer "$MODEL" \
        --task text-to-text \
        --experiment-base-dir "./experiment_results/genai/dynamic_interval${update_interval}" \
        --dataset-path ../alpaca_prompts.csv \
        --dataset-prompt-column prompt \
        --max-time-per-run 60 \
        --max-requests-per-run 500 \
        --num-concurrency $MAX_BATCH_SIZE

    echo "Benchmark 完成 (Dynamic K*, UPDATE_INTERVAL=$update_interval)"

    # 终止服务
    kill_vllm_server

    echo "Dynamic K* UPDATE_INTERVAL=$update_interval 测试完成！"
done

echo ""
echo "========================================"
echo "所有测试完成！"
echo "========================================"
echo "结果保存在:"
echo "  - 调度统计: results/*.json"
echo "  - 日志文件: $LOG_DIR/"
echo "  - Benchmark: experiment_results/genai/*/"