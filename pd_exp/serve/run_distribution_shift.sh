#!/bin/bash

# Distribution Shift 实验脚本
# 验证 THETA 的 IFR 在线 controller 在 workload 突变时的行为:
#   1. θ* 能在 ~W 个样本内收敛到新最优值
#   2. 系统保持 memory-safe (无 OOM)
#   3. 吞吐量暂时下降幅度有限
#
# 实验设计:
#   前半段: ShareGPT (中等输出 ~300-500 tokens)
#   后半段: LongBench (短输出 ~50-100 tokens)
#   对比: pd_ifr (自适应 θ*) vs pd_ratio (固定 θ*=0.8)
#
# 用法: ./run_distribution_shift.sh [GPU_ID]
#
# 环境变量:
#   MODEL: 模型路径，默认 Qwen/Qwen3-8B
#   NUM_PROMPTS_PER_PHASE: 每个阶段的请求数，默认 2000
#   MAX_CONCURRENCY: 最大并发，默认 2048
#   IFR_WINDOW_SIZE: IFR 滑动窗口大小，默认 500

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../syn/common.sh"

# 实验参数
GPU_ID=${1:-0}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_PROMPTS_PER_PHASE=${NUM_PROMPTS_PER_PHASE:-2000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:-4000}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-13000}
IFR_WINDOW_SIZE=${IFR_WINDOW_SIZE:-500}

# 最优配置 (H200, 取 ShareGPT 的最优值作为初始配置)
TB=${TB:-18432}
BS=${BS:-2048}

# 数据集路径
SHAREGPT_PATH="${SCRIPT_DIR}/../outputs/sharegpt_prompts.jsonl"
LONGBENCH_PATH="${SCRIPT_DIR}/../outputs/longbench_prefill.jsonl"

if [ ! -f "$SHAREGPT_PATH" ] || [ ! -f "$LONGBENCH_PATH" ]; then
    echo "错误: 需要以下数据集文件:"
    echo "  $SHAREGPT_PATH"
    echo "  $LONGBENCH_PATH"
    echo ""
    echo "请先导出数据集:"
    echo "  python experiments/serve/export_dataset.py --dataset sharegpt --model $MODEL --num-samples $NUM_PROMPTS_PER_PHASE --output $SHAREGPT_PATH"
    echo "  python experiments/serve/export_dataset.py --dataset longbench --model $MODEL --num-samples $NUM_PROMPTS_PER_PHASE --output $LONGBENCH_PATH"
    exit 1
fi

# 硬件校准文件
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    DEFAULT_CALIBRATION="${SCRIPT_DIR}/../outputs/pd_calibration_${MODEL_SHORT}.json"
    if [ -f "$DEFAULT_CALIBRATION" ]; then
        export VLLM_PD_CALIBRATION_FILE="$DEFAULT_CALIBRATION"
    else
        echo "错误: 未找到硬件校准文件: $DEFAULT_CALIBRATION"
        exit 1
    fi
fi

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/distribution_shift_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

# 初始化环境
init_experiment_env

echo "========================================"
echo "Distribution Shift 实验"
echo "========================================"
echo ""
echo "实验配置:"
echo "  MODEL: $MODEL"
echo "  GPU: $GPU_ID"
echo "  TB: $TB, BS: $BS"
echo "  NUM_PROMPTS_PER_PHASE: $NUM_PROMPTS_PER_PHASE"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  IFR_WINDOW_SIZE: $IFR_WINDOW_SIZE"
echo "  Phase 1: ShareGPT ($SHAREGPT_PATH)"
echo "  Phase 2: LongBench ($LONGBENCH_PATH)"
echo ""

# ========================================
# Step 1: 创建拼接数据集
# ========================================
COMBINED_DATASET="${OUTPUT_DIR}/combined_sharegpt_longbench.jsonl"
echo "创建拼接数据集..."

python3 -c "
import json

# 取前 N 条 ShareGPT
count = 0
with open('$SHAREGPT_PATH') as f, open('$COMBINED_DATASET', 'w') as out:
    for line in f:
        out.write(line)
        count += 1
        if count >= $NUM_PROMPTS_PER_PHASE:
            break
    print(f'  ShareGPT: {count} prompts')

    # 追加前 N 条 LongBench
    count2 = 0
    with open('$LONGBENCH_PATH') as f2:
        for line in f2:
            out.write(line)
            count2 += 1
            if count2 >= $NUM_PROMPTS_PER_PHASE:
                break
    print(f'  LongBench: {count2} prompts')
    print(f'  Total: {count + count2} prompts')
"

TOTAL_PROMPTS=$((NUM_PROMPTS_PER_PHASE * 2))

# 保存实验配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "distribution_shift",
    "purpose": "Validate IFR controller convergence under workload distribution shift",
    "model": "${MODEL}",
    "gpu_id": ${GPU_ID},
    "tb": ${TB},
    "bs": ${BS},
    "num_prompts_per_phase": ${NUM_PROMPTS_PER_PHASE},
    "total_prompts": ${TOTAL_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "ifr_window_size": ${IFR_WINDOW_SIZE},
    "k_ratio": ${K_RATIO},
    "phase1_dataset": "ShareGPT",
    "phase2_dataset": "LongBench",
    "schedulers": ["pd_ifr", "pd_ratio"],
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE}",
    "timestamp": "$(date -Iseconds)"
}
EOF

# ========================================
# Step 2: 运行实验
# ========================================
run_single_experiment() {
    local scheduler=$1
    local port=$((BASE_PORT))
    local log_file="${OUTPUT_DIR}/logs/${scheduler}.log"

    echo ""
    echo "========================================"
    echo "运行: ${scheduler}"
    echo "========================================"

    : > "$log_file"

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export VLLM_COLLECT_SCHEDULE_STATS=1

    case "$scheduler" in
        pd_ifr)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ifr
            export VLLM_PD_IFR_WINDOW_SIZE=$IFR_WINDOW_SIZE
            unset VLLM_PD_K_RATIO VLLM_PD_K_STAR
            ;;
        pd_ratio)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            unset VLLM_PD_K_STAR
            ;;
    esac

    wait_for_gpu_memory $GPU_ID 60 || return 1

    # 启动服务
    local dtype_arg=""
    if [ -n "${DTYPE:-}" ]; then
        dtype_arg="--dtype $DTYPE"
    fi

    VLLM_SCHEDULE_STATS_FILE="${OUTPUT_DIR}/${scheduler}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$BS" \
        --max-num-batched-tokens "$TB" \
        $dtype_arg >> "$log_file" 2>&1 &
    local server_pid=$!

    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "服务启动失败: ${scheduler}"
        kill_server $server_pid $GPU_ID
        return 1
    fi

    echo "服务已启动 (PID: $server_pid, port: $port)"
    echo "开始 benchmark..."

    # 运行 benchmark
    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${port}" \
        --dataset-name custom \
        --dataset-path "$COMBINED_DATASET" \
        --custom-output-len "$CUSTOM_OUTPUT_LEN" \
        --num-prompts "$TOTAL_PROMPTS" \
        --num-warmups 0 \
        --request-rate inf \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --save-detailed \
        --result-dir "${OUTPUT_DIR}" \
        --result-filename "bench_${scheduler}.json" \
        >> "$log_file" 2>&1
    local bench_status=$?

    kill_server $server_pid $GPU_ID

    if [ $bench_status -eq 0 ]; then
        echo "完成: ${scheduler}"
    else
        echo "失败: ${scheduler}"
    fi

    return $bench_status
}

# 顺序运行两个 scheduler
run_single_experiment "pd_ifr"
run_single_experiment "pd_ratio"

echo ""
echo "========================================"
echo "实验完成!"
echo "========================================"
echo ""
echo "结果目录: $OUTPUT_DIR"
echo ""
echo "运行分析脚本:"
echo "  python pd_exp/plot_distribution_shift.py $OUTPUT_DIR"
