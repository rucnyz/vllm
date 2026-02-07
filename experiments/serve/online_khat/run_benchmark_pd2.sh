#!/bin/bash
# =============================================================================
# Benchmark script for comparing Baseline vs PD2 Scheduler
# =============================================================================

set -e

# =============================================================================
# 参数配置
# =============================================================================
MAX_NUM_SEQS=${MAX_NUM_SEQS:-896}          # 2N: max decode batch size
PD2_PREFILL_BATCH=${PD2_PREFILL_BATCH:-448} # N: prefill batch size (default = MAX_NUM_SEQS/2)
OUTPUT_DIR=${OUTPUT_DIR:-"./pd2_results"}
NUM_PROMPTS=${NUM_PROMPTS:-4000}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
CUDA_DEVICES=${CUDA_DEVICES:-6}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
PORT=${PORT:-8134}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1024}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}

# 输入输出长度场景 (input_len output_len)
scenarios=(
    "128 1024"
    "1024 128"
    "512 512"
)

# 测试选项
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_PD2=${RUN_PD2:-1}
SWEEP_N=${SWEEP_N:-0}  # 是否 sweep 不同的 N 值
N_VALUES=${N_VALUES:-"256 512"}  # N 值列表 (当 SWEEP_N=1 时使用)

# =============================================================================
# 辅助函数
# =============================================================================
kill_vllm_server() {
    echo "终止 vllm 服务..."
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo "发送 SIGINT 到 PID $VLLM_PID..."
        kill -INT $VLLM_PID 2>/dev/null || true
        for i in {1..30}; do
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "进程已正常退出"
                break
            fi
            sleep 1
        done
        if kill -0 $VLLM_PID 2>/dev/null; then
            echo "强制终止进程..."
            kill -9 $VLLM_PID 2>/dev/null || true
        fi
    fi
    pkill -f "vllm serve.*$MODEL" 2>/dev/null || true
    sleep 2
    echo "服务已终止"
}

wait_for_server() {
    echo "等待 vllm 服务启动..."
    for i in {1..120}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "服务已就绪！"
            return 0
        fi
        sleep 2
    done
    echo "服务启动超时！"
    return 1
}

run_benchmark() {
    local input_len=$1
    local output_len=$2
    local label=$3
    local result_dir=$4

    echo "运行 benchmark ($label, in=$input_len, out=$output_len, warmup=$NUM_WARMUP_REQUESTS)..."
    vllm bench serve \
        --model $MODEL \
        --backend openai \
        --base-url "http://localhost:$PORT" \
        --endpoint /v1/completions \
        --dataset-name random \
        --random-input-len $input_len \
        --random-output-len $output_len \
        --random-range-ratio $RANDOM_RANGE_RATIO \
        --num-prompts $NUM_PROMPTS \
        --max-concurrency $MAX_CONCURRENCY \
        --num-warmups $NUM_WARMUP_REQUESTS \
        --save-result \
        --result-dir "$result_dir" \
        --result-filename "bench_${label}.json"
}

# =============================================================================
# 主程序
# =============================================================================
echo "========================================"
echo "Baseline vs PD2 Benchmark"
echo "========================================"
echo "参数配置:"
echo "  MAX_NUM_SEQS (2N): $MAX_NUM_SEQS"
echo "  PD2_PREFILL_BATCH (N): $PD2_PREFILL_BATCH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  CUDA_DEVICES: $CUDA_DEVICES"
echo "  MODEL: $MODEL"
echo ""
echo "Scenarios:"
for scenario in "${scenarios[@]}"; do
    echo "  $scenario"
done
echo ""
echo "测试类型:"
echo "  RUN_BASELINE: $RUN_BASELINE"
echo "  RUN_PD2: $RUN_PD2"
echo "  SWEEP_N: $SWEEP_N"
if [ "$SWEEP_N" = "1" ]; then
    echo "  N_VALUES: $N_VALUES"
fi
echo "========================================"

# Trap to cleanup on exit
trap kill_vllm_server EXIT

# =============================================================================
# 遍历所有场景
# =============================================================================
for scenario in "${scenarios[@]}"; do
    read -r INPUT_LEN OUTPUT_LEN <<< "$scenario"
    scenario_name="in${INPUT_LEN}_out${OUTPUT_LEN}"

    echo ""
    echo "========================================"
    echo "Scenario: $scenario_name (input=$INPUT_LEN, output=$OUTPUT_LEN)"
    echo "========================================"

    # 创建输出目录
    RESULT_DIR="${OUTPUT_DIR}/${scenario_name}"
    LOG_DIR="${RESULT_DIR}/logs"
    mkdir -p "$RESULT_DIR" "$LOG_DIR"

    # =========================================================================
    # 1. Baseline 测试
    # =========================================================================
    if [ "$RUN_BASELINE" = "1" ]; then
        echo ""
        echo "--- Baseline (默认调度器) ---"
        kill_vllm_server

        echo "启动 vllm serve (Baseline, max_seqs=$MAX_NUM_SEQS)..."
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
        VLLM_USE_PD_SCHEDULER=0 \
        vllm serve $MODEL \
            --port $PORT \
            --max-num-seqs $MAX_NUM_SEQS \
            --gpu-memory-utilization 0.9 \
            > "$LOG_DIR/vllm_baseline.log" 2>&1 &

        VLLM_PID=$!

        if wait_for_server; then
            run_benchmark "$INPUT_LEN" "$OUTPUT_LEN" "baseline" "$RESULT_DIR"
            echo "Baseline 完成"
        else
            echo "Baseline 跳过 (服务启动失败)"
        fi

        kill_vllm_server
    fi

    # =========================================================================
    # 2. PD2 测试
    # =========================================================================
    if [ "$RUN_PD2" = "1" ]; then
        if [ "$SWEEP_N" = "1" ]; then
            # Sweep 不同的 N 值
            for n_val in $N_VALUES; do
                echo ""
                echo "--- PD2 (N=$n_val, 2N=$MAX_NUM_SEQS) ---"
                kill_vllm_server

                echo "启动 vllm serve (PD2, N=$n_val)..."
                VLLM_COLLECT_SCHEDULE_STATS=1 \
                VLLM_SCHEDULE_STATS_FILE="${RESULT_DIR}/pd2_n${n_val}_stats.json" \
                CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
                VLLM_USE_PD_SCHEDULER=2 \
                VLLM_PD2_PREFILL_BATCH=$n_val \
                vllm serve $MODEL \
                    --port $PORT \
                    --max-num-seqs $MAX_NUM_SEQS \
                    --gpu-memory-utilization 0.9 \
                    > "$LOG_DIR/vllm_pd2_n${n_val}.log" 2>&1 &

                VLLM_PID=$!

                if wait_for_server; then
                    run_benchmark "$INPUT_LEN" "$OUTPUT_LEN" "pd2_n${n_val}" "$RESULT_DIR"
                    echo "PD2 (N=$n_val) 完成"
                else
                    echo "PD2 (N=$n_val) 跳过 (服务启动失败)"
                fi

                kill_vllm_server
            done
        else
            # 单次 PD2 测试
            echo ""
            echo "--- PD2 (N=$PD2_PREFILL_BATCH, 2N=$MAX_NUM_SEQS) ---"
            kill_vllm_server

            echo "启动 vllm serve (PD2)..."
            VLLM_COLLECT_SCHEDULE_STATS=1 \
            VLLM_SCHEDULE_STATS_FILE="${RESULT_DIR}/pd2_stats.json" \
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
            VLLM_USE_PD_SCHEDULER=2 \
            VLLM_PD2_PREFILL_BATCH=$PD2_PREFILL_BATCH \
            vllm serve $MODEL \
                --port $PORT \
                --max-num-seqs $MAX_NUM_SEQS \
                --gpu-memory-utilization 0.9 \
                > "$LOG_DIR/vllm_pd2.log" 2>&1 &

            VLLM_PID=$!

            if wait_for_server; then
                run_benchmark "$INPUT_LEN" "$OUTPUT_LEN" "pd2" "$RESULT_DIR"
                echo "PD2 完成"
            else
                echo "PD2 跳过 (服务启动失败)"
            fi

            kill_vllm_server
        fi
    fi

    echo ""
    echo "Scenario $scenario_name 完成！"
    echo "结果保存在: $RESULT_DIR/"
done

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "========================================"
echo "所有 Benchmark 完成！"
echo "结果保存在: $OUTPUT_DIR/"
echo "========================================"

# 打印结果摘要
echo ""
echo "结果文件:"
find "$OUTPUT_DIR" -name "*.json" -type f 2>/dev/null | head -20 || echo "  (无结果文件)"
