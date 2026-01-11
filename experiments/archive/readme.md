Fixed

```shell
export CUDA_VISIBLE_DEVICES=2,3
for dataset in alpaca lmsys; do
    python experiments/run_fixed_k.py \
        --tensor-parallel-size 2 \
        --use-offline-kstar \
        --dataset $dataset \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/fixed/ \
        --disable-thinking \
        --gpu-memory-utilization 0.90
done
```


Dynamic

```shell
export CUDA_VISIBLE_DEVICES=2,3
for dataset in alpaca lmsys; do
    python experiments/run_dynamic_kstar.py \
        --tensor-parallel-size 2 \
        --ema-alpha 0.3 \
        --dataset $dataset \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/dynamic/ \
        --disable-thinking \
        --gpu-memory-utilization 0.90
done
```

sweep

```shell
export CUDA_VISIBLE_DEVICES=2,3
for dataset in alpaca lmsys; do
      python experiments/run_k_sweep.py \
        --tensor-parallel-size 2 \
        --ema-alpha 0.3 \
        --dataset $dataset \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/sweep/ \
        --disable-thinking \
        --gpu-memory-utilization 0.90 \
done
```



## Yuzhou

```shell

CUDA_VISIBLE_DEVICES=5    python experiments/run_fixed_k.py \
        --tensor-parallel-size 1 \
        --use-offline-kstar \
        --dataset alpaca \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/fixed2/ \
        --disable-thinking \
        --gpu-memory-utilization 0.8
CUDA_VISIBLE_DEVICES=5 python -m cProfile -o ./experiment_results/fixed.prof experiments/run_fixed_k.py \
    --tensor-parallel-size 1 \
    --use-offline-kstar \
    --dataset alpaca \
    --model Qwen/Qwen3-8B \
    --output-dir ./experiment_results/fixed2/ \
    --disable-thinking \
    --gpu-memory-utilization 0.8
    
    
CUDA_VISIBLE_DEVICES=5   python experiments/run_dynamic_kstar.py \
        --tensor-parallel-size 1 \
        --ema-alpha 0.3 \
        --dataset alpaca \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/dynamic2/ \
        --disable-thinking \
        --gpu-memory-utilization 0.8

CUDA_VISIBLE_DEVICES=5 python -m cProfile -o ./experiment_results/dynamic.prof experiments/run_fixed_k.py \
        --tensor-parallel-size 1 \
        --ema-alpha 0.3 \
        --dataset alpaca \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/dynamic2/ \
        --disable-thinking \
        --gpu-memory-utilization 0.8

 CUDA_VISIBLE_DEVICES=5   python experiments/run_k_sweep.py \
        --tensor-parallel-size 1 \
        --dataset alpaca \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/sweep2/ \
        --disable-thinking \
        --gpu-memory-utilization 0.8
```

## Serve-based Throughput Test (Fair Comparison)

The direct LLM.generate() tests above may have unfair warmup conditions.
Use vllm serve + official benchmark tool for fair comparison between fixed k and dynamic k*.

### Step 1: Start two vLLM serve instances

```shell
# Terminal 1: Fixed k mode (port 8000)
# Note: Set VLLM_PD_K_STAR_DYNAMIC to your offline-estimated k* value
CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
    VLLM_PD_K_STAR_DYNAMIC=5 \
    vllm serve Qwen/Qwen3-8B \
        --port 8000 \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.8

# Terminal 2: Dynamic k* mode (port 8001)
CUDA_VISIBLE_DEVICES=1 \
    VLLM_USE_PD_SCHEDULER=1 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=1 \
    vllm serve Qwen/Qwen3-8B \
        --port 8001 \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.8
```

### Step 2: Run throughput tests using official vllm bench

```shell
# Test fixed k (port 8000)
vllm bench serve \
    --model Qwen/Qwen3-8B \
    --port 8000 \
    --dataset-name sharegpt \
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 \
    --num-warmups 100 \
    --max-concurrency 32 \
    --save-result \
    --result-dir ./experiment_results/serve/ \
    --result-filename fixed_k_results.json

# Test dynamic k* (port 8001)
vllm bench serve \
    --model Qwen/Qwen3-8B \
    --port 8001 \
    --dataset-name sharegpt \
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 \
    --num-warmups 100 \
    --max-concurrency 32 \
    --save-result \
    --result-dir ./experiment_results/serve/ \
    --result-filename dynamic_kstar_results.json
```

### Output Metrics

The benchmark will report:
- `Output token throughput (tok/s)` - **Primary metric for comparison**
- `Total token throughput (tok/s)` - Input + output tokens per second
- `Request throughput (req/s)` - Requests per second
- `TTFT` - Time to first token
- `TPOT` - Time per output token
- `ITL` - Inter-token latency

### Why this is fairer

1. Both servers are fully warmed up before testing (100 warmup requests)
2. Same prompts, same order for both tests
3. GPU is in stable boost state for both tests
4. Official tool calculates throughput correctly (tokens/s, not just req/s)