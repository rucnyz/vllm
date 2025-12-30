# Serve-based Throughput Test (Fair Comparison)

Use vllm serve + genai-bench for fair comparison between fixed k and dynamic k*.

Attention: make the port numbers consistent throughout the steps below.

## Step 0: Prepare dataset

Export our datasets (alpaca/sharegpt/lmsys) to CSV format:

```shell
python experiments/serve/export_dataset.py \
    --dataset alpaca \
    --model Qwen/Qwen3-8B \
    --num-samples 1000 \
    --output ./experiments/serve/alpaca_prompts.csv
```
Export Environment Variables:
export VLLM_PD_ALPHA_P=0.006528784356021418
export VLLM_PD_BETA_P=6.498792400220424e-06
export VLLM_PD_ALPHA_D=0.004303444935141221
export VLLM_PD_BETA_D=0.00023557651251992446

export VLLM_PD_ALPHA_P=0.002528784356021418
export VLLM_PD_BETA_P=6.498792400220424e-06
export VLLM_PD_ALPHA_D=0.002303444935141221
export VLLM_PD_BETA_D=0.00023557651251992446

export VLLM_PD_ALPHA_P=0.002528784356021418
export VLLM_PD_BETA_P=6.498792400220424e-06
export VLLM_PD_ALPHA_D=0.001
export VLLM_PD_BETA_D=0.00023557651251992446

## Step 1: Start two vLLM serve instances

profile
```shell
VLLM_COLLECT_SCHEDULE_STATS=1 VLLM_SCHEDULE_STATS_FILE="results/pd5.json"  CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=1 \
    vllm serve Qwen/Qwen3-8B \
        --port 8124 \
        --max-num-seqs 128 \
        --gpu-memory-utilization 0.9 \
        --api-key "7355608"
        
python analyze_schedule_stats.py results/pd4.json results/baseline.json --plot
```

```shell
# Terminal 1: Fixed k mode (port 8000)
# Note: Set VLLM_PD_K_STAR_DYNAMIC to your offline-estimated k* value
CUDA_VISIBLE_DEVICES=6 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
    python -m vllm.entrypoints.cli.main serve Qwen/Qwen3-8B \
        --port 8000 \
        --max-num-seqs 128 \
        --gpu-memory-utilization 0.9 \
        --api-key "7355608"

# Terminal 2: Dynamic k* mode (port 8001)
CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=1 \
    vllm serve Qwen/Qwen3-8B \
        --port 8124 \
        --max-num-seqs 128 \
        --gpu-memory-utilization 0.9 \
        --api-key "7355608"

# Terminal 3: Baseline (port 8003)
CUDA_VISIBLE_DEVICES=2 \
    VLLM_USE_PD_SCHEDULER=0 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
    vllm serve Qwen/Qwen3-8B \
        --port 8124 \
        --max-num-seqs 128 \
        --gpu-memory-utilization 0.9 \
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
    --num-concurrency 128

# Test dynamic k* (port 8001)
genai-bench benchmark \
    --api-backend vllm \
    --api-key "7355608" \
    --api-base http://localhost:8124 \
    --api-model-name "Qwen/Qwen3-8B" \
    --model-tokenizer "Qwen/Qwen3-8B" \
    --task text-to-text \
    --experiment-base-dir ./experiment_results/genai/dynamic \
    --dataset-path ./experiments/serve/alpaca_prompts.csv \
    --dataset-prompt-column prompt \
    --max-time-per-run 60 \
    --max-requests-per-run 500 \
    --num-concurrency 128
    
# Test baseline
genai-bench benchmark \
    --api-backend vllm \
    --api-key "7355608" \
    --api-base http://localhost:8124 \
    --api-model-name "Qwen/Qwen3-8B" \
    --model-tokenizer "Qwen/Qwen3-8B" \
    --task text-to-text \
    --experiment-base-dir ./experiment_results/genai/baseline \
    --dataset-path ./experiments/serve/alpaca_prompts.csv \
    --dataset-prompt-column prompt \
    --max-time-per-run 60 \
    --max-requests-per-run 500 \
    --num-concurrency 128

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
  --group-key "" \
  --metrics-time-unit ms \
  --filter-criteria '{"model": "Qwen/Qwen3-8B"}'
```

## Output Metrics

genai-bench reports:
- End-to-end latency
- `TTFT` - Time to first token
- `TPOT` - Time per output token
- Throughput at each concurrency level

