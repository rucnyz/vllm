# Serve-based Throughput Test (Fair Comparison)

Use vllm serve + genai-bench for fair comparison between fixed k and dynamic k*.

## Step 0: Prepare dataset

Export our datasets (alpaca/sharegpt/lmsys) to CSV format:

```shell
python experiments/serve/export_dataset.py \
    --dataset alpaca \
    --model Qwen/Qwen3-8B \
    --num-samples 1000 \
    --output ./experiments/serve/alpaca_prompts.csv
```

## Step 1: Start two vLLM serve instances

```shell
# Terminal 1: Fixed k mode (port 8000)
# Note: Set VLLM_PD_K_STAR_DYNAMIC to your offline-estimated k* value
CUDA_VISIBLE_DEVICES=5 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
    VLLM_PD_K_STAR_DYNAMIC=4 \
    vllm serve Qwen/Qwen3-8B \
        --port 8000 \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.8 \
        --api-key "7355608"

# Terminal 2: Dynamic k* mode (port 8001)
CUDA_VISIBLE_DEVICES=6 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=1 \
    vllm serve Qwen/Qwen3-8B \
        --port 8001 \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.8 \
        --api-key "7355608"

# Terminal 3: Baseline (port 8003)
CUDA_VISIBLE_DEVICES=7 \
    VLLM_USE_PD_SCHEDULER=0 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
    vllm serve Qwen/Qwen3-8B \
        --port 8003 \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.8 \
        --api-key "7355608"
```

## Step 2: Run throughput tests using genai-bench

### Option A: Use exported CSV dataset (recommended)

```shell
# Test fixed k (port 8000)
genai-bench benchmark \
    --api-backend vllm \
    --api-key "7355608" \
    --api-base http://localhost:8000 \
    --api-model-name "Qwen/Qwen3-8B" \
    --model-tokenizer "Qwen/Qwen3-8B" \
    --task text-to-text \
    --experiment-base-dir ./experiment_results/genai/fixed \
    --dataset-path ./experiments/serve/alpaca_prompts.csv \
    --dataset-prompt-column prompt \
    --max-time-per-run 60 \
    --max-requests-per-run 500 \
    --num-concurrency 32 --num-concurrency 64

# Test dynamic k* (port 8001)
genai-bench benchmark \
    --api-backend vllm \
    --api-key "7355608" \
    --api-base http://localhost:8001 \
    --api-model-name "Qwen/Qwen3-8B" \
    --model-tokenizer "Qwen/Qwen3-8B" \
    --task text-to-text \
    --experiment-base-dir ./experiment_results/genai/dynamic \
    --dataset-path ./experiments/serve/alpaca_prompts.csv \
    --dataset-prompt-column prompt \
    --max-time-per-run 60 \
    --max-requests-per-run 500 \
    --num-concurrency 32 --num-concurrency 64
    
# Test baseline
genai-bench benchmark \
    --api-backend vllm \
    --api-key "7355608" \
    --api-base http://localhost:8003 \
    --api-model-name "Qwen/Qwen3-8B" \
    --model-tokenizer "Qwen/Qwen3-8B" \
    --task text-to-text \
    --experiment-base-dir ./experiment_results/genai/baseline \
    --dataset-path ./experiments/serve/alpaca_prompts.csv \
    --dataset-prompt-column prompt \
    --max-time-per-run 60 \
    --max-requests-per-run 500 \
    --num-concurrency 32 --num-concurrency 64

```


## Step 3: Export results to Excel

```shell
genai-bench excel \
    --experiment-folder ./experiment_results/genai \
    --excel-name pd_comparison.xlsx \
    --metric-percentile "mean, p50, p90, p99" \
    --metrics-time-unit ms

genai-bench plot \
  --experiments-folder ./experiment_results/genai \
  --group-key none \
  --preset multi_line_latency \
  --metrics-time-unit ms \
  --filter-criteria '{"model": "Qwen/Qwen3-8B"}'
```

## Output Metrics

genai-bench reports:
- End-to-end latency
- `TTFT` - Time to first token
- `TPOT` - Time per output token
- Throughput at each concurrency level

