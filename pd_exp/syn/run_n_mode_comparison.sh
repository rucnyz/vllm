#!/bin/bash

# N 模式对比实验脚本
# 对比 reactive (启发式) 和 paper (论文公式 Eq.16) 两种 N 计算方式
#
# 论文公式: N* = (C - κ·ln(1/ε)) / (E[L] + (1-θ)/(θp)·ln(1/(1-θ)))
# - reactive: 基于 KV cache 压力的反应式调整
# - paper: 基于内存约束的理论公式，带 OOM 概率保证
#
# 用法: ./run_n_mode_comparison.sh [MAX_GPUS]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

# 实验参数
MAX_GPUS=${1:-4}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
NUM_PROMPTS=${NUM_PROMPTS:-2000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1024}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-10000}

# 硬件校准文件 (必须)
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
        exit 1
    fi
fi
echo "使用校准文件: $VLLM_PD_CALIBRATION_FILE"

# 从校准文件中读取参数
if [ -f "$VLLM_PD_CALIBRATION_FILE" ]; then
    ALPHA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_p'])" 2>/dev/null || echo "null")
    BETA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_p'])" 2>/dev/null || echo "null")
    ALPHA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_d'])" 2>/dev/null || echo "null")
    BETA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_d'])" 2>/dev/null || echo "null")
    echo "  alpha_p: $ALPHA_P, beta_p: $BETA_P"
    echo "  alpha_d: $ALPHA_D, beta_d: $BETA_D"
fi

# ============== 实验配置 ==============
# 固定 BS 和 TB，聚焦于 N 模式对比
BS=${BS:-1024}
TB=${TB:-16384}

# N 模式配置
# - baseline: vLLM 默认调度器 (对照组)
# - pd_reactive: PD 调度器 + reactive N 模式
# - pd_paper_eps001: PD 调度器 + paper N 模式 (ε=0.001, 保守)
# - pd_paper_eps01: PD 调度器 + paper N 模式 (ε=0.01, 默认)
# - pd_paper_eps05: PD 调度器 + paper N 模式 (ε=0.05, 激进)
# - pd_paper_eps10: PD 调度器 + paper N 模式 (ε=0.10, 非常激进)
N_MODES=("baseline" "pd_reactive" "pd_paper_eps001" "pd_paper_eps01" "pd_paper_eps05" "pd_paper_eps10")

# Workload 场景 (input_len output_len)
# 设计原则:
# - short_output: 短输出 (p 大), 论文假设 p<<1 可能不成立
# - medium_output: 中等输出, 符合论文假设
# - long_output: 长输出 (p 小), κ = 1/(p²·E[L]) 会很大
# - variable_input: 输入长度变化大, 测试 E[L] 估计准确性
SCENARIOS=("512 50" "512 200" "512 500" "512 1000" "256 200" "1024 200")
SCENARIO_NAMES=("short_out" "medium_out" "long_out" "very_long_out" "short_in" "long_in")

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/n_mode_comparison_BS${BS}_TB${TB}"
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env

echo "========================================"
echo "N 模式对比实验 (reactive vs paper)"
echo "========================================"

# 检测并选择 GPU
select_gpus $MAX_GPUS

echo ""
echo "实验配置:"
echo "  MODEL: $MODEL"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  BS: $BS, TB: $TB"
echo "  K_RATIO: $K_RATIO"
echo "  N_MODES: ${N_MODES[*]}"
echo "  SCENARIOS: ${#SCENARIOS[@]} 个"
echo ""
echo "N 模式说明:"
echo "  - baseline: vLLM 默认调度器"
echo "  - pd_reactive: PD + 反应式 N (基于 KV cache 压力)"
echo "  - pd_paper_eps*: PD + 论文公式 N (OOM 概率 ≤ ε)"
echo ""

# 生成实验队列
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"

for i in "${!SCENARIOS[@]}"; do
    scenario="${SCENARIOS[$i]}"
    scenario_name="${SCENARIO_NAMES[$i]}"
    read -r input_len output_len <<< "$scenario"

    for n_mode in "${N_MODES[@]}"; do
        echo "${n_mode}|${input_len}|${output_len}|${scenario_name}" >> "$QUEUE_FILE"
    done
done

TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
echo "总实验数: $TOTAL_EXPERIMENTS"
echo ""

# 保存实验配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "n_mode_comparison",
    "description": "Compare reactive vs paper-based N computation modes",
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "bs": ${BS},
    "tb": ${TB},
    "k_ratio": ${K_RATIO},
    "n_modes": [$(printf '"%s",' "${N_MODES[@]}" | sed 's/,$//')]
    "n_mode_descriptions": {
        "baseline": "vLLM default scheduler (no PD)",
        "pd_reactive": "PD scheduler with reactive N mode (heuristic-based)",
        "pd_paper_eps001": "PD scheduler with paper N mode (ε=0.001, very conservative)",
        "pd_paper_eps01": "PD scheduler with paper N mode (ε=0.01, default)",
        "pd_paper_eps05": "PD scheduler with paper N mode (ε=0.05, aggressive)",
        "pd_paper_eps10": "PD scheduler with paper N mode (ε=0.10, very aggressive)"
    },
    "paper_formula": "N* = (C - κ·ln(1/ε)) / (E[L] + (1-θ)/(θp)·ln(1/(1-θ)))",
    "paper_formula_terms": {
        "C": "Total KV cache capacity in tokens",
        "kappa": "1/(p²·E[L]) - peak memory supremum",
        "epsilon": "OOM probability tolerance",
        "theta": "k/N switching ratio",
        "p": "1/mean_output_length",
        "E_L": "Expected input (prompt) length"
    },
    "scenarios": {
        $(for i in "${!SCENARIOS[@]}"; do
            scenario="${SCENARIOS[$i]}"
            name="${SCENARIO_NAMES[$i]}"
            read -r input_len output_len <<< "$scenario"
            echo "\"${name}\": {\"input_len\": ${input_len}, \"output_len\": ${output_len}},"
        done | sed '$ s/,$//')
    },
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE:-null}",
    "calibration_params": {
        "alpha_p": ${ALPHA_P:-null},
        "beta_p": ${BETA_P:-null},
        "alpha_d": ${ALPHA_D:-null},
        "beta_d": ${BETA_D:-null}
    },
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

# 运行单个实验
run_experiment() {
    local gpu_id=$1 n_mode=$2 input_len=$3 output_len=$4 scenario_name=$5
    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/${scenario_name}"
    local log_file="${result_dir}/logs/${n_mode}.log"

    mkdir -p "${result_dir}/logs"
    : > "$log_file"

    check_port_available $port $gpu_id || return 1

    echo "[GPU $gpu_id] 开始: ${n_mode} ${scenario_name} (in=${input_len}, out=${output_len})"

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export VLLM_COLLECT_SCHEDULE_STATS=1

    case "$n_mode" in
        baseline)
            # vLLM 默认调度器
            export VLLM_USE_PD_SCHEDULER=0
            unset VLLM_PD_K_MODE VLLM_PD_K_RATIO VLLM_PD_N_MODE VLLM_PD_OOM_TOLERANCE
            ;;
        pd_reactive)
            # PD 调度器 + reactive N 模式
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            export VLLM_PD_N_MODE=reactive
            unset VLLM_PD_OOM_TOLERANCE
            ;;
        pd_paper_eps001)
            # PD 调度器 + paper N 模式 (ε=0.001)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            export VLLM_PD_N_MODE=paper
            export VLLM_PD_OOM_TOLERANCE=0.001
            ;;
        pd_paper_eps01)
            # PD 调度器 + paper N 模式 (ε=0.01)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            export VLLM_PD_N_MODE=paper
            export VLLM_PD_OOM_TOLERANCE=0.01
            ;;
        pd_paper_eps05)
            # PD 调度器 + paper N 模式 (ε=0.05)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            export VLLM_PD_N_MODE=paper
            export VLLM_PD_OOM_TOLERANCE=0.05
            ;;
        pd_paper_eps10)
            # PD 调度器 + paper N 模式 (ε=0.10)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            export VLLM_PD_N_MODE=paper
            export VLLM_PD_OOM_TOLERANCE=0.10
            ;;
    esac

    # 等待 GPU 内存可用
    wait_for_gpu_memory $gpu_id 60 || return 1

    # 启动服务
    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${n_mode}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$BS" \
        --max-num-batched-tokens "$TB" >> "$log_file" 2>&1 &
    local server_pid=$!

    # 等待服务启动
    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "[GPU $gpu_id] 服务启动失败: ${n_mode}"
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
        --result-filename "bench_${n_mode}.json" >> "$log_file" 2>&1
    local bench_status=$?

    kill_server $server_pid $gpu_id

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] 完成: ${n_mode} ${scenario_name}"
    else
        echo "[GPU $gpu_id] 失败: ${n_mode} ${scenario_name}"
    fi

    return $bench_status
}

# 并行调度
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break

        IFS='|' read -r n_mode input_len output_len scenario_name <<< "$exp"

        if run_experiment "$gpu_id" "$n_mode" "$input_len" "$output_len" "$scenario_name"; then
            update_progress "OK|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        fi
    done
}

# 主流程
echo "开始并行执行..."
echo "========================================"

> "$PROGRESS_FILE"

for gpu_id in "${GPUS_TO_USE[@]}"; do
    gpu_worker "$gpu_id" &
    WORKER_PIDS+=($!)
    echo "启动 GPU $gpu_id worker (PID: ${WORKER_PIDS[-1]})"
    sleep 10
done

echo ""
echo "监控进度: watch -n 5 'wc -l ${PROGRESS_FILE}'"
echo ""

for pid in "${WORKER_PIDS[@]}"; do
    wait $pid || true
done

print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"
echo ""
echo "运行分析脚本:"
echo "  python ${SCRIPT_DIR}/analyze_n_mode_comparison.py $OUTPUT_DIR"
