# k* vs k_hat Real Model Experiment

Validates the P/D Competition Scheduler's analytical k* formula against the empirically optimal k_hat on real LLM inference.

## Overview

This experiment script (`kstar_vs_khat_real_model.py`) validates the theoretical k* (optimal switching threshold) derived from the P/D Competition Scheduler paper by comparing it with k_hat (empirically determined optimal threshold) on real LLM inference workloads.

**Key Questions Answered:**
- How close is the analytical k* to the empirically optimal k_hat?
- How does k* behave across different workload characteristics (input/output length distributions)?
- Can online p estimation with EMA achieve good k* values in real-time?

## Experiment Workflow

1. **Load Dataset** - Load prompts from one of the supported datasets
2. **Profile Model** - Measure timing parameters (α_p, α_d, β_p, β_d) via linear regression
3. **Offline p Estimation** - Estimate p from batch samples during profiling
4. **Online p Estimation** - Run dynamic k* inference with EMA-based p updates
5. **Compute k*** - Calculate both offline k* and online k* (final converged value)
6. **Search k_hat** - Test different k values (1 to N) to find the empirical optimum
7. **Compare Results** - Analyze throughput gap between k* and k_hat, plot comparison curves

## Supported Datasets

| Dataset | Source | Characteristics |
|---------|--------|-----------------|
| `alpaca` | HuggingFace: tatsu-lab/alpaca | Instruction-following, short inputs, medium outputs (recommended baseline) |
| `sharegpt` | Local JSON file | Multi-turn conversation, heavy-tail output distribution |
| `longbench` | HuggingFace: THUDM/LongBench-v2 | Long-context QA, very long inputs, short outputs (A/B/C/D choices) |
| `longbench_v1` | HuggingFace: THUDM/LongBench | Long-context QA (original version) |
| `lmsys` | HuggingFace: lmsys/lmsys-chat-1m | Real user conversations, diverse output lengths |
| `processbench` | HuggingFace: Qwen/ProcessBench | Math problem solving with multiple splits |

## Command-Line Arguments

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | HuggingFace model name or local path |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism |

### Dataset Selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `alpaca` | Dataset choice: `alpaca`, `sharegpt`, `longbench`, `longbench_v1`, `lmsys`, `processbench` |
| `--sharegpt-path` | `./ShareGPT_V3_unfiltered_cleaned_split.json` | Path to ShareGPT JSON file (only for `sharegpt`) |
| `--max-input-tokens` | `32000` | Skip samples exceeding this input length |
| `--processbench-split` | `gsm8k` | ProcessBench subset: `gsm8k`, `math`, `olympiadbench`, `omni-math` |

### Experiment Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-requests` | `500` | Number of requests for throughput testing |
| `--batch-size` | `32` | Scheduler batch size N |
| `--max-output-tokens` | `128` | Maximum output tokens per request |

### P Estimation (Offline + Online)

The experiment now performs BOTH offline and online p estimation simultaneously:
- **Offline**: Batch estimation from profiling samples
- **Online**: EMA update during dynamic k* inference

| Argument | Default | Description |
|----------|---------|-------------|
| `--offline-p-multiplier` | `2.0` | Offline samples = multiplier × kstar-update-interval |
| `--ema-alpha` | `0.3` | EMA smoothing factor for online p updates (higher = more weight on recent observations) |
| `--kstar-update-interval` | `32` | Number of completed requests between k* updates during online learning |

### Profiling Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-prefill-samples` | `30` | Number of samples for prefill time measurement |
| `--num-decode-repeats` | `3` | Repetitions per batch size for decode profiling |

### Memory-Constrained Batch Sizing

Automatically compute optimal batch size N* based on GPU memory constraints.

| Argument | Default | Description |
|----------|---------|-------------|
| `--optimize-batch-size` | `false` | Enable automatic N* computation (overrides `--batch-size`) |
| `--gpu-memory-gb` | `80` | Total GPU memory in GB (e.g., 80 for A100, 40 for A6000) |
| `--model-memory-gb` | auto | Model weights memory in GB (auto-detected if not set) |
| `--kv-cache-bytes` | auto | KV cache bytes per token (auto-computed if not set) |
| `--oom-tolerance` | `0.05` | OOM probability tolerance ε (lower = more conservative) |

### Output Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `./experiment_results_real` | Directory for saving results |

## Output Files

Each experiment generates the following files (prefixed with dataset name):

| File | Description |
|------|-------------|
| `{dataset}_results.json` | Complete experiment results in JSON format |
| `{dataset}_throughput_vs_k.pdf` | Throughput curve with k*, k_hat markers, and online k* throughput line |
| `{dataset}_profiling_results.pdf` | Prefill/Decode time fitting plots |
| `{dataset}_input_length_distribution.pdf` | Input token length histogram |
| `{dataset}_output_length_distribution.pdf` | Output token length histogram |
| `{dataset}_online_kstar_convergence.pdf` | k* convergence over time |
| `truncated_samples_{dataset}.json` | Samples that reached max output tokens (for analysis) |
| `pd_timeline_{dataset}.json` | P/D scheduler timeline data (if multiprocessing disabled) |

## Usage Examples

### 1. Basic Usage (Alpaca)

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca \
    --num-requests 500 \
    --batch-size 32
```

### 2. ShareGPT Dataset

Requires downloading `ShareGPT_V3_unfiltered_cleaned_split.json` first.

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset sharegpt \
    --sharegpt-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-requests 1000
```

### 3. LongBench Dataset

LongBench has very long contexts. Set `--max-input-tokens` to filter samples.

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset longbench \
    --max-input-tokens 16000 \
    --num-requests 200
```

### 4. LMSYS-Chat-1M Dataset

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset lmsys \
    --num-requests 500
```

### 5. ProcessBench (Math Problems)

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset processbench \
    --processbench-split math \
    --num-requests 300
```

### 6. Custom EMA Parameters

Adjust EMA smoothing factor and k* update interval for online learning.

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset sharegpt \
    --sharegpt-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --ema-alpha 0.3 \
    --kstar-update-interval 64
```

### 7. Custom Offline Sample Size

Control how many samples are used for offline p estimation.

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca \
    --offline-p-multiplier 3.0 \
    --kstar-update-interval 32
```

This uses 3.0 × 32 = 96 samples for offline estimation.

### 8. Large Batch Size Experiment

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca \
    --batch-size 100 \
    --num-requests 2000
```

### 9. Multi-GPU Tensor Parallelism

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --dataset alpaca
```

### 10. High-Precision Profiling

Increase profiling samples for more accurate timing parameter estimation.

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca \
    --num-prefill-samples 50 \
    --num-decode-repeats 5 \
    --offline-p-multiplier 4.0
```

### 11. Memory-Constrained Batch Sizing (N* Optimization)

Automatically compute optimal batch size N* based on GPU memory constraints.
This uses the probabilistic bound from Section 3.3 of the P/D Competition paper
to ensure OOM probability stays below ε.

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca \
    --optimize-batch-size \
    --gpu-memory-gb 80 \
    --model-memory-gb 16 \
    --oom-tolerance 0.05
```

For smaller GPUs (e.g., A6000 with 48GB):

```bash
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca \
    --optimize-batch-size \
    --gpu-memory-gb 48 \
    --oom-tolerance 0.01
```

## Dependencies

```bash
pip install vllm transformers datasets matplotlib numpy tqdm
```

## Notes

1. **ShareGPT dataset** requires manually downloading the JSON file
2. **LongBench** samples exceeding the model's context length are automatically skipped
3. First-time usage of HuggingFace datasets requires internet access for download
4. Use `--max-output-tokens` to control output length truncation behavior
5. **Dual Estimation**: The experiment runs both offline and online p estimation simultaneously:
   - Offline k* is computed from batch samples (controlled by `--offline-p-multiplier`)
   - Online k* is computed from dynamic inference with EMA updates
   - The throughput plot shows both offline k* marker and online k* throughput line for comparison
6. **Memory-Constrained Batch Sizing** (`--optimize-batch-size`) computes the optimal batch size N* using the probabilistic bound from Section 3.3. The formula accounts for:
   - Memory at decode start (input tokens + partial outputs from old requests)
   - Peak memory dynamics during decode phase
   - Safety margin based on OOM tolerance ε
7. **Degenerate Output Detection**: The experiment detects and flags repetition-loop outputs (e.g., "aaaa..."). Use `--exclude-degenerate-throughput` to exclude these from throughput calculations.
8. **Timeline Recording**: Use `--disable-multiprocessing` to enable P/D scheduler timeline recording with tensor parallelism.

## Specifying GPUs

To run on specific GPUs, use the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# Use GPU 0 and 1
CUDA_VISIBLE_DEVICES=0,1 python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --dataset alpaca

# Use GPU 5 and 6
CUDA_VISIBLE_DEVICES=5,6 python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --dataset alpaca
```

## Running Multiple Datasets Sequentially

Chain commands with `&&` to run experiments on multiple datasets:

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --dataset alpaca \
    --ema-alpha 0.3 && \
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --dataset sharegpt \
    --sharegpt-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --ema-alpha 0.3 && \
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --dataset lmsys \
    --ema-alpha 0.3 && \
python experiments/kstar_vs_khat_real_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --dataset longbench \
    --max-input-tokens 16000 \
    --ema-alpha 0.3
```

## Key Concepts

### Analytical k* (Proposition 1)
The optimal switching threshold k* is the smallest integer k satisfying:
```
k * τ(N-k) - Σ_{j=N-k+1}^{N} τ(j) >= α_p
```
where τ(j) is the expected time per completion with batch size j.

### Termination Probability p
The experiment estimates p using both methods simultaneously:
- **Offline estimation**: Run a batch of samples during profiling, compute p = 1/mean_output_length
- **Online EMA**: Update p incrementally during inference: `p_new = α * (1/L) + (1-α) * p_old`

The throughput plot shows:
- **Online k\* horizontal line**: Throughput achieved using dynamic k* during inference (varies as p updates)
- **Offline k\* marker**: Where offline k* falls on the empirical throughput curve
- **k_hat marker**: The empirically optimal k value

### Timing Parameters
- **α_p**: Fixed prefill overhead (seconds)
- **β_p**: Per-token prefill cost (seconds/token)
- **α_d**: Fixed decode overhead per step (seconds)
- **β_d**: Per-request decode cost (seconds/request)
