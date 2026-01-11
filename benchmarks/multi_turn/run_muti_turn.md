
download dataset: download_gutenberg.py
generate multi-turn conversations: generate_conversations_only.py
use the converted dataset: generated_conversations.json



VLLM_COLLECT_SCHEDULE_STATS=1 \
VLLM_SCHEDULE_STATS_FILE=/scratch/yuzhou/zwf/vllm/schedule_stats.json \
vllm serve Qwen/Qwen3-8B \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests

python benchmark_serving_multi_turn_threaded.py \
    --model Qwen/Qwen3-8B \
    --url http://localhost:8000 \
    --input-file generated_conversations.json \
    --num-clients 1024

use plot_schedule_stats.py to plot the schedule stats


