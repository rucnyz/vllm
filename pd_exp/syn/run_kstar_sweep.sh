#!/bin/bash

# 统一的 K* 参数扫描 + 调度策略对比脚本 (多 GPU 版本)
# 合并 online_test.sh 和 run_kstar_sweep_vllm.sh 的功能
#
# 用法: ./run_kstar_sweep.sh [MAX_GPUS]
#   MAX_GPUS: 使用的 GPU 数量，默认 4
#
# 环境变量 (可选):
#   GPUS: 指定使用的 GPU 列表，如 "0,1,2,3"
#   MODEL: 模型名称，默认 "Qwen/Qwen3-8B"
#   NUM_PROMPTS: 每个实验的 prompt 数量，默认 5000
#   MAX_CONCURRENCY: 最大并发数，默认 2048
#   RANDOM_RANGE_RATIO: 随机范围比例，默认 0.5
#   MAX_BATCH_SIZE: Baseline 的 max_num_seqs，为空使用 vLLM 默认值
#   PD_MAX_BATCH_SIZE: P/D 分离的 max_num_seqs (固定 K* 模式使用)
#   K_RATIO_VALUES: K ratio 扫描值列表，默认 "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
#   RUN_BASELINE: 是否运行 baseline，默认 1
#   RUN_KSTAR: 是否运行固定 K* 扫描 (需要 PD_MAX_BATCH_SIZE)，默认 0
#   RUN_KRATIO: 是否运行 K ratio 扫描 (固定 θ*)，默认 0
#   RUN_RATIO_AUTO: 是否运行 ratio 模式 (动态 θ*)，默认 0
#   RUN_DIRECT: 是否运行 direct 模式 (自动 k*)，默认 1
#   SKIP_EXISTING: 跳过已有结果的测试 (检测 bench_*.json)，默认 0
#   TOKEN_BUDGET_DEFAULT: Baseline 的 max_num_batched_tokens
#   TOKEN_BUDGET_PD: P/D Scheduler 的 max_num_batched_tokens，默认 10752
#   PD_MAX_NUM_SEQS: K Ratio 模式的 max_num_seqs，默认 1408
#   BASE_PORT: 起始端口号，默认 10000 (每个 GPU 使用 BASE_PORT + GPU_ID)
#
# 示例:
#   # 使用 4 张 GPU 运行 baseline 和 K ratio
#   ./run_kstar_sweep.sh 4
#
#   # 指定 GPU 列表
#   GPUS="0,2,4,6" ./run_kstar_sweep.sh
#
#   # 自定义 K ratio 值
#   K_RATIO_VALUES="0.2 0.4 0.6 0.8" ./run_kstar_sweep.sh
#
#   # 固定 K* 扫描 (需要指定 PD_MAX_BATCH_SIZE)
#   PD_MAX_BATCH_SIZE=1152 RUN_KRATIO=0 RUN_KSTAR=1 ./run_kstar_sweep.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# 清理函数
WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

# ========================================
# 命令行参数与环境变量配置
# ========================================
MAX_GPUS=${1:-4}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-""}             # Baseline 使用, 为空则使用 vLLM 默认值
PD_MAX_BATCH_SIZE=${PD_MAX_BATCH_SIZE:-1024}    # P/D 分离使用
NUM_PROMPTS=${NUM_PROMPTS:-4000}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}
BASE_PORT=${BASE_PORT:-10000}

# K_STAR 值列表：动态生成到 PD_MAX_BATCH_SIZE (仅当 RUN_KSTAR=1 时)
K_STAR_VALUES=()
if [ -n "$PD_MAX_BATCH_SIZE" ]; then
    for ((i=128; i<=PD_MAX_BATCH_SIZE; i+=128)); do
        K_STAR_VALUES+=($i)
    done
fi

# K_RATIO 值列表：用于自适应 N 模式下的 k = ratio * N 实验
# 注意：不要用引号，否则会被当作一个字符串
K_RATIO_VALUES=(${K_RATIO_VALUES:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9})

# 测试类型开关
RUN_BASELINE=${RUN_BASELINE:-1}    # 运行 baseline (默认调度器)
RUN_KSTAR=${RUN_KSTAR:-0}          # 运行 K* 扫描 (固定 K* 模式)
RUN_KRATIO=${RUN_KRATIO:-0}        # 运行 K ratio 扫描 (手动指定固定 θ*) - 默认关闭，可手动启用
RUN_RATIO_AUTO=${RUN_RATIO_AUTO:-0}  # 运行 ratio_auto 模式 (自动计算 θ*) - 已禁用，渐近公式严重低估 k*
RUN_DIRECT=${RUN_DIRECT:-1}      # 运行 dynamic 模式
RUN_IFR=${RUN_IFR:-0}            # 运行 IFR 模式 (IFR 校正的在线自适应)
SKIP_EXISTING=${SKIP_EXISTING:-0}  # 跳过已有结果 (检测 bench_*.json)

# Warmup 配置
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}

# Token Budget 参数 (max_num_batched_tokens)
TOKEN_BUDGET_DEFAULT=${TOKEN_BUDGET_DEFAULT:-}   # Baseline 使用
TOKEN_BUDGET_PD=${TOKEN_BUDGET_PD:-16384}        # P/D Scheduler 使用

# P/D Scheduler max_num_seqs 参数 (K Ratio 模式使用)
PD_MAX_NUM_SEQS=${PD_MAX_NUM_SEQS:-1024}

# 硬件校准文件 (必须)
# PD Scheduler 需要硬件校准参数才能准确调度
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    DEFAULT_CALIBRATION="${SCRIPT_DIR}/../outputs/pd_calibration.json"
    if [ -f "$DEFAULT_CALIBRATION" ]; then
        export VLLM_PD_CALIBRATION_FILE="$DEFAULT_CALIBRATION"
    else
        echo "错误: 未找到硬件校准文件!"
        echo ""
        echo "PD Scheduler 需要硬件校准参数才能准确调度。"
        echo "请先运行校准:"
        echo "  python -m vllm.v1.core.sched.calibration --model ${MODEL}"
        echo ""
        echo "校准文件默认保存到: ${DEFAULT_CALIBRATION}"
        echo "或手动指定: VLLM_PD_CALIBRATION_FILE=/path/to/file.json ./run_kstar_sweep.sh"
        exit 1
    fi
fi
echo "使用校准文件: $VLLM_PD_CALIBRATION_FILE"

# 从校准文件中读取 alpha/beta 参数
if [ -f "$VLLM_PD_CALIBRATION_FILE" ]; then
    ALPHA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_p'])" 2>/dev/null || echo "null")
    BETA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_p'])" 2>/dev/null || echo "null")
    ALPHA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_d'])" 2>/dev/null || echo "null")
    BETA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_d'])" 2>/dev/null || echo "null")
    echo "  alpha_p: $ALPHA_P, beta_p: $BETA_P"
    echo "  alpha_d: $ALPHA_D, beta_d: $BETA_D"
else
    ALPHA_P="null"
    BETA_P="null"
    ALPHA_D="null"
    BETA_D="null"
fi

# Scenario 配置
SCENARIOS=("128 1024" "1024 128" "512 512")

# 输出目录
if [ -n "$PD_MAX_BATCH_SIZE" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-"${SCRIPT_DIR}/../outputs/kstar_bs${PD_MAX_BATCH_SIZE}_c${MAX_CONCURRENCY}_n${NUM_PROMPTS}"}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-"${SCRIPT_DIR}/../outputs/kstar_max${PD_MAX_NUM_SEQS}_tb${TOKEN_BUDGET_PD}_c${MAX_CONCURRENCY}_n${NUM_PROMPTS}"}"
fi
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env

echo "========================================"
echo "K* 参数扫描实验 (多 GPU 版本)"
echo "========================================"

# 检测并选择 GPU
select_gpus $MAX_GPUS

echo ""
echo "实验配置:"
echo "  MODEL: $MODEL"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  NUM_WARMUP_REQUESTS: $NUM_WARMUP_REQUESTS"
echo "  MAX_BATCH_SIZE (Baseline): ${MAX_BATCH_SIZE:-"(vLLM默认)"}"
echo "  PD_MAX_BATCH_SIZE: ${PD_MAX_BATCH_SIZE:-"(自动)"}"
echo "  PD_MAX_NUM_SEQS: $PD_MAX_NUM_SEQS"
echo "  TOKEN_BUDGET_DEFAULT: ${TOKEN_BUDGET_DEFAULT:-"(vLLM默认)"}"
echo "  TOKEN_BUDGET_PD: $TOKEN_BUDGET_PD"
echo "  RUN_BASELINE: $RUN_BASELINE"
echo "  RUN_KSTAR: $RUN_KSTAR"
[ "$RUN_KSTAR" = "1" ] && echo "    K_STAR_VALUES: ${K_STAR_VALUES[*]}"
echo "  RUN_KRATIO: $RUN_KRATIO"
[ "$RUN_KRATIO" = "1" ] && echo "    K_RATIO_VALUES: ${K_RATIO_VALUES[*]}"
echo "  RUN_RATIO_AUTO: $RUN_RATIO_AUTO"
echo "  RUN_DIRECT: $RUN_DIRECT"
echo "  RUN_IFR: $RUN_IFR"
echo "  SKIP_EXISTING: $SKIP_EXISTING"
echo "  CALIBRATION_FILE: ${VLLM_PD_CALIBRATION_FILE:-"(未设置，使用默认参数)"}"
echo ""

# ========================================
# 生成实验队列
# ========================================
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"

for scenario in "${SCENARIOS[@]}"; do
    read -r input_len output_len <<< "$scenario"

    # Baseline 实验
    if [ "$RUN_BASELINE" = "1" ]; then
        echo "baseline|${input_len}|${output_len}" >> "$QUEUE_FILE"
    fi

    # 固定 K* 扫描
    if [ "$RUN_KSTAR" = "1" ] && [ ${#K_STAR_VALUES[@]} -gt 0 ]; then
        for k_star in "${K_STAR_VALUES[@]}"; do
            echo "kstar|${input_len}|${output_len}|${k_star}" >> "$QUEUE_FILE"
        done
    fi

    # K Ratio 扫描 (固定 θ*)
    if [ "$RUN_KRATIO" = "1" ]; then
        for k_ratio in "${K_RATIO_VALUES[@]}"; do
            echo "kratio|${input_len}|${output_len}|${k_ratio}" >> "$QUEUE_FILE"
        done
    fi

    # Ratio Auto 模式 (动态 θ*)
    if [ "$RUN_RATIO_AUTO" = "1" ]; then
        echo "ratio_auto|${input_len}|${output_len}" >> "$QUEUE_FILE"
    fi

    # Direct 模式 (自动计算 k*)
    if [ "$RUN_DIRECT" = "1" ]; then
        echo "direct|${input_len}|${output_len}" >> "$QUEUE_FILE"
    fi

    # IFR 模式 (IFR 校正的在线自适应)
    if [ "$RUN_IFR" = "1" ]; then
        echo "ifr|${input_len}|${output_len}" >> "$QUEUE_FILE"
    fi
done

TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
echo "总实验数: $TOTAL_EXPERIMENTS"
echo ""

# 保存实验配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "num_warmup_requests": ${NUM_WARMUP_REQUESTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "random_range_ratio": ${RANDOM_RANGE_RATIO},
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
    "run_ratio_auto": ${RUN_RATIO_AUTO},
    "run_direct": ${RUN_DIRECT},
    "run_ifr": ${RUN_IFR},
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE:-null}",
    "calibration_params": {
        "alpha_p": ${ALPHA_P},
        "beta_p": ${BETA_P},
        "alpha_d": ${ALPHA_D},
        "beta_d": ${BETA_D}
    },
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

# ========================================
# 运行单个实验
# ========================================
run_experiment() {
    local gpu_id=$1
    local exp_type=$2
    local input_len=$3
    local output_len=$4
    local param=$5  # k_star 或 k_ratio 值

    local port=$((BASE_PORT + gpu_id))
    local scenario_name="in${input_len}_out${output_len}"
    local result_dir="${OUTPUT_DIR}/${scenario_name}"
    local log_file result_prefix

    mkdir -p "${result_dir}/logs"

    # 根据实验类型设置参数
    case "$exp_type" in
        baseline)
            result_prefix="baseline"
            ;;
        kstar)
            result_prefix="fixed${param}"
            ;;
        kratio)
            local ratio_name=$(echo "$param" | sed 's/\./_/')
            result_prefix="kratio_${ratio_name}"
            ;;
        ratio_auto)
            result_prefix="ratio_auto"
            ;;
        direct)
            result_prefix="direct"
            ;;
        ifr)
            result_prefix="ifr"
            ;;
    esac

    log_file="${result_dir}/logs/${result_prefix}.log"
    : > "$log_file"

    # 检查是否跳过已有结果
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${result_dir}/bench_${result_prefix}.json" ]; then
        echo "[GPU $gpu_id] 跳过: ${exp_type} ${scenario_name} (结果已存在)"
        return 0
    fi

    check_port_available $port $gpu_id || return 1

    echo "[GPU $gpu_id] 开始: ${exp_type} ${scenario_name} ${param:+param=$param}"

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export VLLM_COLLECT_SCHEDULE_STATS=1

    # 构建 vllm serve 参数
    local serve_args="--gpu-memory-utilization 0.9"

    case "$exp_type" in
        baseline)
            # 使用默认 vLLM 调度器
            export VLLM_USE_PD_SCHEDULER=0
            unset VLLM_PD_K_MODE VLLM_PD_K_STAR VLLM_PD_K_RATIO
            [ -n "$MAX_BATCH_SIZE" ] && serve_args="$serve_args --max-num-seqs $MAX_BATCH_SIZE"
            [ -n "$TOKEN_BUDGET_DEFAULT" ] && serve_args="$serve_args --max-num-batched-tokens $TOKEN_BUDGET_DEFAULT"
            ;;
        kstar)
            # Direct 模式 + 固定 k*: k* = param
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=direct
            export VLLM_PD_K_STAR=$param
            unset VLLM_PD_K_RATIO
            serve_args="$serve_args --max-num-seqs $PD_MAX_BATCH_SIZE --max-num-batched-tokens $TOKEN_BUDGET_PD"
            ;;
        kratio)
            # 比例模式 (固定 θ*): k* = param × N
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$param
            unset VLLM_PD_K_STAR
            serve_args="$serve_args --max-num-seqs $PD_MAX_NUM_SEQS --max-num-batched-tokens $TOKEN_BUDGET_PD"
            ;;
        ratio_auto)
            # 比例模式 (动态 θ*): 不设置 VLLM_PD_K_RATIO，系统自动计算
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            unset VLLM_PD_K_RATIO VLLM_PD_K_STAR
            serve_args="$serve_args --max-num-seqs $PD_MAX_NUM_SEQS --max-num-batched-tokens $TOKEN_BUDGET_PD"
            ;;
        direct)
            # Direct 模式: 自动计算 k*
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=direct
            unset VLLM_PD_K_RATIO VLLM_PD_K_STAR
            serve_args="$serve_args --max-num-seqs $PD_MAX_NUM_SEQS --max-num-batched-tokens $TOKEN_BUDGET_PD"
            ;;
        ifr)
            # IFR 模式: IFR 校正的在线自适应
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ifr
            unset VLLM_PD_K_RATIO VLLM_PD_K_STAR
            serve_args="$serve_args --max-num-seqs $PD_MAX_NUM_SEQS --max-num-batched-tokens $TOKEN_BUDGET_PD"
            ;;
    esac

    # 等待 GPU 内存可用
    wait_for_gpu_memory $gpu_id 60 || return 1

    # 启动服务
    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${result_prefix}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        $serve_args >> "$log_file" 2>&1 &
    local server_pid=$!

    # 等待服务启动
    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "[GPU $gpu_id] 服务启动失败: ${exp_type} ${scenario_name}"
        kill_server $server_pid $gpu_id
        return 1
    fi

    # 运行 benchmark
    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${port}" \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmups "$NUM_WARMUP_REQUESTS" \
        --request-rate inf \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "bench_${result_prefix}.json" >> "$log_file" 2>&1
    local bench_status=$?

    kill_server $server_pid $gpu_id

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] 完成: ${exp_type} ${scenario_name} ${param:+param=$param}"
    else
        echo "[GPU $gpu_id] 失败: ${exp_type} ${scenario_name}"
    fi

    return $bench_status
}

# ========================================
# GPU Worker 并行调度
# ========================================
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break

        # 解析实验参数 (格式: type|input_len|output_len[|param])
        IFS='|' read -r exp_type input_len output_len param <<< "$exp"

        if run_experiment "$gpu_id" "$exp_type" "$input_len" "$output_len" "$param"; then
            update_progress "OK|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
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
    sleep 10  # 错开启动避免 CUDA 初始化竞争
done

echo ""
echo "监控进度: watch -n 5 'wc -l ${PROGRESS_FILE}'"
echo ""

for pid in "${WORKER_PIDS[@]}"; do
    wait $pid || true
done

print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"

echo ""
echo "目录结构:"
echo "  experiment_config.json (实验配置)"
for scenario in "${SCENARIOS[@]}"; do
    read -r input_len output_len <<< "$scenario"
    name="in${input_len}_out${output_len}"
    echo "  - ${name}/"
    echo "      调度统计: *_stats.json"
    echo "      Benchmark: bench_*.json"
    echo "      日志: logs/"
done
