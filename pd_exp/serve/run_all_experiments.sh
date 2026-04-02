#!/bin/bash

# 运行所有 4-GPU 对比实验 (3 workload × 3 concurrency = 9 组)
#
# 用法: bash pd_exp/serve/run_all_experiments.sh [GPU1] [GPU2] [GPU3] [GPU4]
# 示例: bash pd_exp/serve/run_all_experiments.sh 0 1 2 3
#
# 环境变量:
#   MODEL: 模型路径，默认 Qwen/Qwen3-8B
#   NUM_PROMPTS: 每组请求数，默认 2000
#   SKIP_DISAGG: 设为1跳过所有disagg
#   DISAGG_BENCH_TIMEOUT: disagg bench 超时秒数，默认 600

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU1=${1:-0}
GPU2=${2:-1}
GPU3=${3:-2}
GPU4=${4:-3}

NUM_PROMPTS=${NUM_PROMPTS:-2000}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}

# 实验矩阵
WORKLOADS=(
    "1024:128"   # Prefill-heavy
    "512:512"    # Balanced
    "128:1024"   # Decode-heavy
)
CONCURRENCIES=(128 256 512)

total=$((${#WORKLOADS[@]} * ${#CONCURRENCIES[@]}))
current=0
failed=()
succeeded=()

echo "========================================"
echo "全量实验: ${total} 组"
echo "  GPUs: ${GPU1},${GPU2},${GPU3},${GPU4}"
echo "  MODEL: ${MODEL}"
echo "  NUM_PROMPTS: ${NUM_PROMPTS}"
echo "  Workloads: ${WORKLOADS[*]}"
echo "  Concurrencies: ${CONCURRENCIES[*]}"
echo "========================================"
echo ""

for workload in "${WORKLOADS[@]}"; do
    INPUT_LEN="${workload%%:*}"
    OUTPUT_LEN="${workload##*:}"

    for conc in "${CONCURRENCIES[@]}"; do
        current=$((current + 1))
        label="i${INPUT_LEN}_o${OUTPUT_LEN}_c${conc}"

        echo ""
        echo "========================================"
        echo "[${current}/${total}] ${label}"
        echo "========================================"

        NUM_PROMPTS="$NUM_PROMPTS" \
        MAX_CONCURRENCY="$conc" \
        INPUT_LEN="$INPUT_LEN" \
        OUTPUT_LEN="$OUTPUT_LEN" \
            bash "${SCRIPT_DIR}/run_4gpu_comparison.sh" "$GPU1" "$GPU2" "$GPU3" "$GPU4"

        if [ $? -eq 0 ]; then
            succeeded+=("$label")
        else
            failed+=("$label")
        fi

        echo "[${current}/${total}] ${label} 完成"
    done
done

# 汇总
echo ""
echo "========================================"
echo "全量实验完成: ${#succeeded[@]}/${total} 成功"
echo "========================================"

if [ ${#succeeded[@]} -gt 0 ]; then
    echo "成功:"
    for s in "${succeeded[@]}"; do
        echo "  ✓ $s"
    done
fi

if [ ${#failed[@]} -gt 0 ]; then
    echo "失败:"
    for f in "${failed[@]}"; do
        echo "  ✗ $f"
    done
fi
