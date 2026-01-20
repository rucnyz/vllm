#!/bin/bash

# 统一的 K* 参数扫描 + 调度策略对比脚本
# 合并 online_test.sh 和 run_kstar_sweep_vllm.sh 的功能
#
# 用法: ./run_benchmark_unified.sh [MAX_BATCH_SIZE] [OUTPUT_DIR] [NUM_PROMPTS]
#   MAX_BATCH_SIZE: Baseline 的 max_num_seqs，为空使用 vLLM 默认值
#   OUTPUT_DIR: 结果保存目录，默认 ../outputs/
#   NUM_PROMPTS: 每个实验的 prompt 数量，默认 5000
#
# 环境变量 (可选):
#   CUDA_DEVICES: GPU 设备，默认 "5"
#   MODEL: 模型名称，默认 "Qwen/Qwen3-8B"
#   PORT: 服务端口，默认 8104
#   MAX_CONCURRENCY: 最大并发数，默认 2048
#   RANDOM_RANGE_RATIO: 随机范围比例，默认 0.5
#   PD_MAX_BATCH_SIZE: P/D 分离的 max_num_seqs (固定 K* 模式使用)
#   K_RATIO_VALUES: K ratio 扫描值列表，默认 "0.2 0.4 0.6 0.8"
#   RUN_BASELINE: 是否运行 baseline，默认 1
#   RUN_KSTAR: 是否运行固定 K* 扫描 (需要 PD_MAX_BATCH_SIZE)，默认 1
#   RUN_KRATIO: 是否运行 K ratio 扫描 (自适应 N)，默认 1
#   SKIP_EXISTING: 跳过已有结果的测试 (检测 bench_*.json)，默认 0
#   TOKEN_BUDGET_DEFAULT: Baseline 的 max_num_batched_tokens，默认 8192
#   TOKEN_BUDGET_PD: P/D Scheduler 的 max_num_batched_tokens，默认 16384
#   PD_MAX_NUM_SEQS: K Ratio 模式的 max_num_seqs，默认 2048
#
# 示例:
#   # 只测试 baseline 和 K ratio (自适应 N)
#   ./run_benchmark_unified.sh
#
#   # 自定义 K ratio 值
#   K_RATIO_VALUES="0.2 0.4 0.6 0.8" ./run_benchmark_unified.sh
#
#   # 固定 K* 扫描 (需要指定 PD_MAX_BATCH_SIZE)
#   PD_MAX_BATCH_SIZE=1152 RUN_KRATIO=0 ./run_benchmark_unified.sh
#
#
#   # 自定义 token budget (分别设置默认调度器和 P/D 调度器)
#   TOKEN_BUDGET_DEFAULT=4096 TOKEN_BUDGET_PD=8192 ./run_benchmark_unified.sh

set -e

# 增加文件描述符限制 (高并发需要)
ulimit -n 65535 2>/dev/null || true

# 激活 vllm 虚拟环境
source /scratch/yuzhou/aproj/vllm/.venv/bin/activate

# ========================================
# 命令行参数
# ========================================
MAX_BATCH_SIZE=${1:-""}             # Baseline 使用, 为空则使用 vLLM 默认值
PD_MAX_BATCH_SIZE=${PD_MAX_BATCH_SIZE:-""}    # P/D 分离使用, 为空则让 scheduler 自动决定 N
NUM_PROMPTS=${3:-5000}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================
# 环境变量配置 (可通过环境变量覆盖)
# ========================================
CUDA_DEVICES=${CUDA_DEVICES:-"6"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
PORT=${PORT:-8111}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}  # = PD_MAX_BATCH_SIZE (不要超太多,否则请求会超时)
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}

# 导出公共环境变量 (所有测试类型共用)
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export VLLM_COLLECT_SCHEDULE_STATS=1

# K_STAR 值列表：动态生成到 PD_MAX_BATCH_SIZE (仅当 PD_MAX_BATCH_SIZE 设置时)
K_STAR_VALUES=()
if [ -n "$PD_MAX_BATCH_SIZE" ]; then
    for ((i=128; i<=PD_MAX_BATCH_SIZE; i+=128)); do
        K_STAR_VALUES+=($i)
    done
fi

# K_RATIO 值列表：用于自适应 N 模式下的 k = ratio * N 实验
# 默认测试 0.3, 0.5, 0.7 三个比例
K_RATIO_VALUES=(${K_RATIO_VALUES:-"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"})

# 测试类型开关
RUN_BASELINE=${RUN_BASELINE:-1}    # 运行 baseline (默认调度器)
RUN_KSTAR=${RUN_KSTAR:-0}          # 运行 K* 扫描 (固定 K* 模式)
RUN_KRATIO=${RUN_KRATIO:-1}        # 运行 K ratio 扫描 (自适应 N 模式)
SKIP_EXISTING=${SKIP_EXISTING:-0}  # 跳过已有结果 (检测 bench_*.json)

# Warmup 配置
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}  # 预热请求数

# Token Budget 参数 (max_num_batched_tokens)
# 分别为默认调度器和 P/D 调度器设置不同的 token budget
TOKEN_BUDGET_DEFAULT=${TOKEN_BUDGET_DEFAULT:-}   # Baseline 使用 (schedule_default)
TOKEN_BUDGET_PD=${TOKEN_BUDGET_PD:-10752}             # P/D Scheduler 使用 (schedule_pd)

# P/D Scheduler max_num_seqs 参数 (K Ratio 模式使用)
PD_MAX_NUM_SEQS=${PD_MAX_NUM_SEQS:-1408}

# 输出目录 (必须在 MAX_CONCURRENCY 和 NUM_PROMPTS 之后定义)
if [ -n "$PD_MAX_BATCH_SIZE" ]; then
    OUTPUT_BASE_DIR=${2:-"${SCRIPT_DIR}/../outputs/bs${PD_MAX_BATCH_SIZE}_c${MAX_CONCURRENCY}_n${NUM_PROMPTS}"}
else
    OUTPUT_BASE_DIR=${2:-"${SCRIPT_DIR}/../outputs/bsAuto_max${PD_MAX_NUM_SEQS}_tb${TOKEN_BUDGET_PD}"}
fi

# ========================================
# Scenario 配置
# 格式: "input_len output_len"
# 文件夹名自动生成为: in{input}_out{output}
# ========================================
scenarios=(
    "128 1024"
    "1024 128"
    "512 512"
)


# 创建输出根目录
mkdir -p "$OUTPUT_BASE_DIR"

# 保存实验配置
cat > "${OUTPUT_BASE_DIR}/experiment_config.json" << EOF
{
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "num_warmup_requests": ${NUM_WARMUP_REQUESTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "random_range_ratio": ${RANDOM_RANGE_RATIO},
    "cuda_devices": "${CUDA_DEVICES}",
    "port": ${PORT},
    "max_batch_size_baseline": "${MAX_BATCH_SIZE:-null}",
    "pd_max_batch_size": "${PD_MAX_BATCH_SIZE:-null}",
    "pd_max_num_seqs": ${PD_MAX_NUM_SEQS},
    "token_budget_default": "${TOKEN_BUDGET_DEFAULT:-null}",
    "token_budget_pd": ${TOKEN_BUDGET_PD},
    "k_star_values": [$(IFS=,; echo "${K_STAR_VALUES[*]}")],
    "k_ratio_values": [$(echo "${K_RATIO_VALUES[*]}" | sed 's/ /, /g')],
    "scenarios": ["128 1024", "1024 128", "512 512"],
    "run_baseline": ${RUN_BASELINE},
    "run_kstar": ${RUN_KSTAR},
    "run_kratio": ${RUN_KRATIO},
    "timestamp": "$(date -Iseconds)"
}
EOF

echo "========================================"
echo "统一 Benchmark 脚本"
echo "========================================"
echo "参数配置:"
echo "  MAX_BATCH_SIZE (Baseline): ${MAX_BATCH_SIZE:-"(vLLM默认)"}"
echo "  PD_MAX_BATCH_SIZE (P/D分离): ${PD_MAX_BATCH_SIZE:-"(自动)"}"
echo "  OUTPUT_DIR: $OUTPUT_BASE_DIR"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  NUM_WARMUP_REQUESTS: $NUM_WARMUP_REQUESTS"
echo "  CUDA_DEVICES: $CUDA_DEVICES"
echo "  MODEL: $MODEL"
echo "  PORT: $PORT"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  RANDOM_RANGE_RATIO: $RANDOM_RANGE_RATIO"
echo "  TOKEN_BUDGET_DEFAULT: $TOKEN_BUDGET_DEFAULT (Baseline)"
echo "  TOKEN_BUDGET_PD: $TOKEN_BUDGET_PD (P/D Scheduler)"
echo "  PD_MAX_NUM_SEQS: $PD_MAX_NUM_SEQS (K Ratio 模式)"
echo ""
echo "测试类型:"
echo "  RUN_BASELINE: $RUN_BASELINE"
if [ -n "$PD_MAX_BATCH_SIZE" ]; then
    echo "  RUN_KSTAR: $RUN_KSTAR (固定 K* 值: ${K_STAR_VALUES[*]})"
else
    echo "  RUN_KSTAR: $RUN_KSTAR (需要设置 PD_MAX_BATCH_SIZE)"
fi
echo "  RUN_KRATIO: $RUN_KRATIO (K ratio 值: ${K_RATIO_VALUES[*]})"
echo "  SKIP_EXISTING: $SKIP_EXISTING"
echo "========================================"

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

    # 只杀死当前端口的进程，不影响其他端口的 vllm 服务
    pkill -INT -f "vllm serve.*--port $PORT" 2>/dev/null || true
    sleep 3
    echo "服务已终止"
}

# 运行 benchmark 的函数
run_benchmark() {
    local input_len=$1
    local output_len=$2
    local result_prefix=$3
    local result_dir=$4

    echo "运行 vllm bench serve (${result_prefix}, warmup=${NUM_WARMUP_REQUESTS})..."
    vllm bench serve \
        --model $MODEL \
        --base-url "http://localhost:${PORT}" \
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
        --result-filename "bench_${result_prefix}.json"
}

# 检查结果是否已存在的函数
check_result_exists() {
    local result_dir=$1
    local result_prefix=$2
    local result_file="${result_dir}/bench_${result_prefix}.json"
    [ -f "$result_file" ]
}

# 统一的启动服务并运行 benchmark 的函数
# 参数:
#   $1: test_name     - 测试名称 (用于日志显示)
#   $2: result_prefix - 结果文件前缀 (bench_xxx.json)
#   $3: log_name      - 日志文件名
#   $4: env_vars      - 额外环境变量 (空格分隔的 KEY=VALUE 对)
#   $5: extra_args    - 额外的 vllm serve 参数
start_vllm_and_benchmark() {
    local test_name=$1
    local result_prefix=$2
    local log_name=$3
    local env_vars=$4
    local extra_args=$5

    # 检查是否跳过已有结果
    if [ "$SKIP_EXISTING" = "1" ] && check_result_exists "$RESULT_DIR" "$result_prefix"; then
        echo "跳过 $test_name (结果已存在: bench_${result_prefix}.json)"
        return 0
    fi

    kill_vllm_server

    echo ""
    echo "--- $test_name ---"
    echo "启动 vllm serve ($test_name)..."

    # 设置额外环境变量
    for var in $env_vars; do
        export "$var"
    done

    VLLM_SCHEDULE_STATS_FILE="${RESULT_DIR}/${result_prefix}.json" \
    vllm serve $MODEL \
        --port $PORT \
        --gpu-memory-utilization 0.9 \
        $extra_args \
        > "$LOG_DIR/vllm_${log_name}.log" 2>&1 &

    # 取消额外环境变量
    for var in $env_vars; do
        unset "${var%%=*}"
    done

    if wait_for_server; then
        run_benchmark "$input_len" "$output_len" "$result_prefix" "$RESULT_DIR"
        kill_vllm_server
        echo "$test_name 完成"
        return 0
    else
        echo "$test_name 跳过 (服务启动失败)"
        kill_vllm_server
        return 1
    fi
}

# ========================================
# 主循环：按 scenario 分类
# ========================================

for scenario in "${scenarios[@]}"; do
    read -r input_len output_len <<< "$scenario"
    scenario_name="in${input_len}_out${output_len}"

    echo ""
    echo "========================================"
    echo "Scenario: $scenario_name (input=$input_len, output=$output_len)"
    echo "========================================"

    # 创建 scenario 目录
    RESULT_DIR="${OUTPUT_BASE_DIR}/${scenario_name}"
    LOG_DIR="${RESULT_DIR}/logs"
    mkdir -p "$RESULT_DIR"
    mkdir -p "$LOG_DIR"

    # ----------------------------------------
    # 1. Baseline 测试 (默认调度器)
    # ----------------------------------------
    if [ "$RUN_BASELINE" = "1" ]; then
        # 构建可选参数
        BASELINE_ARGS=""
        [ -n "$MAX_BATCH_SIZE" ] && BASELINE_ARGS="$BASELINE_ARGS --max-num-seqs $MAX_BATCH_SIZE"
        [ -n "$TOKEN_BUDGET_DEFAULT" ] && BASELINE_ARGS="$BASELINE_ARGS --max-num-batched-tokens $TOKEN_BUDGET_DEFAULT"

        start_vllm_and_benchmark "Baseline (默认调度器)" "baseline" "baseline" \
            "VLLM_USE_PD_SCHEDULER=0" "$BASELINE_ARGS"
    fi

    # ----------------------------------------
    # 2. PD Scheduler 固定 K* 测试 (需要 PD_MAX_BATCH_SIZE)
    # ----------------------------------------
    if [ "$RUN_KSTAR" = "1" ] && [ -n "$PD_MAX_BATCH_SIZE" ] && [ ${#K_STAR_VALUES[@]} -gt 0 ]; then
        KSTAR_ARGS="--max-num-seqs $PD_MAX_BATCH_SIZE --max-num-batched-tokens $TOKEN_BUDGET_PD"

        for k_star in "${K_STAR_VALUES[@]}"; do
            start_vllm_and_benchmark "Fixed K*=$k_star" "fixed${k_star}" "k${k_star}" \
                "VLLM_USE_PD_SCHEDULER=1 VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 VLLM_PD_K_STAR=$k_star" \
                "$KSTAR_ARGS"
        done
    fi

    # ----------------------------------------
    # 3. PD Scheduler K Ratio 测试 (自适应 N, k = ratio * N)
    # ----------------------------------------
    if [ "$RUN_KRATIO" = "1" ]; then
        KRATIO_ARGS="--max-num-seqs $PD_MAX_NUM_SEQS --max-num-batched-tokens $TOKEN_BUDGET_PD"

        for k_ratio in ${K_RATIO_VALUES[@]}; do
            # 格式化 ratio 为文件名友好格式 (0.5 -> 0_5)
            ratio_name=$(echo "$k_ratio" | sed 's/\./_/')

            start_vllm_and_benchmark "K Ratio=$k_ratio (自适应 N)" "kratio_${ratio_name}" "kratio_${ratio_name}" \
                "VLLM_USE_PD_SCHEDULER=1 VLLM_PD_K_RATIO=$k_ratio" "$KRATIO_ARGS"
        done
    fi

    echo ""
    echo "Scenario $scenario_name 全部完成！"
    echo "结果保存在: $RESULT_DIR/"
done

echo ""
echo "========================================"
echo "所有测试完成！"
echo "========================================"
echo "结果保存在: ${OUTPUT_BASE_DIR}/"
echo ""
echo "目录结构:"
echo "  experiment_config.json (实验配置)"
for scenario in "${scenarios[@]}"; do
    read -r input_len output_len <<< "$scenario"
    name="in${input_len}_out${output_len}"
    echo "  - ${name}/"
    echo "      调度统计: baseline.json"
    if [ "$RUN_KSTAR" = "1" ] && [ -n "$PD_MAX_BATCH_SIZE" ]; then
        echo "                fixed*.json (固定 K* sweep)"
    fi
    if [ "$RUN_KRATIO" = "1" ]; then
        echo "                kratio_*.json (K ratio sweep, 自适应 N)"
    fi
    echo "      Benchmark: bench_*.json"
    echo "      日志: logs/"
done
