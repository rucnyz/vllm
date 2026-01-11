#!/bin/bash

MODEL="Qwen/Qwen3-8B"
NUM_PROMPTS=1000
OUTPUT_DIR="offline"

# 创建输出目录
mkdir -p $OUTPUT_DIR

scenarios=(
    "long_in_short_out 2048 128"
    "short_in_long_out 128 2048"
    "balanced 512 512"
)

# 默认调度
echo "========== Default Scheduling =========="
for scenario in "${scenarios[@]}"; do
    read -r name input_len output_len <<< "$scenario"

    echo "Running benchmark: default_${name}"

    CUDA_VISIBLE_DEVICES=5 \
    VLLM_COLLECT_SCHEDULE_STATS=1 \
    vllm bench throughput \
        --model $MODEL \
        --input-len $input_len \
        --output-len $output_len \
        --num-prompts $NUM_PROMPTS \
        --output-json "${OUTPUT_DIR}/results_default_${name}.json"

    # 重命名 schedule_stats.json 文件
    if [ -f "schedule_stats.json" ]; then
        mv schedule_stats.json "${OUTPUT_DIR}/schedule_stats_default_${name}.json"
    fi
done

# Chunked Prefill
echo "========== Chunked Prefill =========="
for scenario in "${scenarios[@]}"; do
    read -r name input_len output_len <<< "$scenario"

    echo "Running benchmark: chunked_${name}"

    CUDA_VISIBLE_DEVICES=5 \
    VLLM_COLLECT_SCHEDULE_STATS=1 \
    vllm bench throughput \
        --model $MODEL \
        --input-len $input_len \
        --output-len $output_len \
        --num-prompts $NUM_PROMPTS \
        --enable-chunked-prefill \
        --max-num-batched-tokens 4096 \
        --output-json "${OUTPUT_DIR}/results_chunked_${name}.json"

    # 重命名 schedule_stats.json 文件
    if [ -f "schedule_stats.json" ]; then
        mv schedule_stats.json "${OUTPUT_DIR}/schedule_stats_chunked_${name}.json"
    fi
done

echo "All benchmarks completed!"