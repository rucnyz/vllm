

# 1. 启动服务器 (保持运行)
CUDA_VISIBLE_DEVICES=6 \
VLLM_COLLECT_SCHEDULE_STATS=1 \
VLLM_SCHEDULE_STATS_FILE=/scratch/yuzhou/zwf/vllm/experiments/schedule_stats/schedule_stats.json \
    VLLM_USE_PD_SCHEDULER=0 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
    vllm serve Qwen/Qwen3-8B \
        --port 8000 \
        --gpu-memory-utilization 0.9 \
        --api-key "7355608"

# 2. 另一个终端运行 benchmark
python experiments/serve/run_benchmark_with_stats.py
# 或
bash experiments/serve/run_benchmark_with_stats.sh