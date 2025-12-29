# Memory Model Validation Experiment

Validates the theoretical memory model from Section 3.3 of the P/D Competition paper by running **real LLM inference** with vLLM and measuring actual GPU memory usage.

## Theoretical Predictions Being Validated

### 1. Expected Initial Memory E[X₀]

At decode start, the KV cache contains:
- **New requests (k)**: input tokens only
- **Old requests (N-k)**: input tokens + partial outputs from previous cycles

The formula:
```
E[X₀] = N × E[L] + N × (1-θ)²/θp × ln(1/(1-θ))
```

Where:
- `N` = batch size
- `θ = k/N` = normalized switching threshold
- `p` = termination probability per step
- `E[L]` = mean input length

### 2. Expected Peak Memory E[Xₘₐₓ]

Peak memory occurs during decode phase due to token generation before completions:

```
E[Xₘₐₓ] = E[X₀] + κ
```

Where `κ = 1/(p² × E[L])` is the supremum constant.

### 3. OOM Probability Bound

The tail probability satisfies:

```
P(sup Yₜ > x) ≤ exp(-2|d|x/v)
```

Where:
- `Yₜ = Xₜ - X₀` = memory change from start
- `d_N = -N × p × E[L]` = expected drift (negative)
- `v_N ≈ 2N/p` = per-step variance

## Experiment Workflow

1. **Load Dataset** - Load prompts from Alpaca or ShareGPT
2. **Initialize Model** - Load LLM with vLLM
3. **Profile Parameters** - Estimate `p` from actual model outputs
4. **Compute Theory** - Calculate theoretical `E[X₀]`, `E[Xₘₐₓ]`, `κ`
5. **Run Real Inference** - Execute multiple decode cycles:
   - Run batch inference with vLLM
   - Measure GPU memory via `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()`
   - Record initial, peak, and final memory for each cycle
6. **Validate** - Compare empirical measurements against theory

## Command-Line Arguments

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | HuggingFace model name or local path |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism |

### Dataset Selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `alpaca` | Dataset: `alpaca` or `sharegpt` |
| `--sharegpt-path` | `./ShareGPT_V3_...` | Path to ShareGPT JSON |
| `--max-samples` | `5000` | Max samples to load from dataset |

### Experiment Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | `32` | Batch size N |
| `--switching-threshold` | `N//5` | Switching threshold k |
| `--num-cycles` | `100` | Number of decode cycles to run |
| `--max-output-tokens` | `256` | Maximum output tokens per request |

### Output Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `./memory_validation_results` | Output directory |
| `--run-both` | `false` | Run on both Alpaca and ShareGPT |

## Usage Examples

### 1. Basic Validation (Alpaca)

```bash
python experiments/memory_model_validation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca \
    --batch-size 32 \
    --num-cycles 100
```

### 2. ShareGPT (Heavy-tailed outputs)

```bash
python experiments/memory_model_validation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset sharegpt \
    --sharegpt-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --batch-size 32 \
    --num-cycles 100
```

### 3. Compare Both Datasets

```bash
python experiments/memory_model_validation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --run-both \
    --batch-size 32 \
    --num-cycles 50
```

### 4. Large Batch Size

```bash
python experiments/memory_model_validation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca \
    --batch-size 64 \
    --switching-threshold 12 \
    --num-cycles 100
```

### 5. Multi-GPU Tensor Parallelism

```bash
python experiments/memory_model_validation.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --dataset alpaca \
    --batch-size 64
```

## Output Files

| File | Description |
|------|-------------|
| `{dataset}_memory_validation.pdf` | Four-panel validation plot |
| `{dataset}_memory_validation.json` | Detailed results in JSON |

### Validation Plot Contents

1. **Top-left**: X₀ distribution vs theoretical E[X₀]
2. **Top-right**: Xₘₐₓ distribution vs theoretical E[Xₘₐₓ]
3. **Bottom-left**: sup(Y) distribution vs κ
4. **Bottom-right**: OOM probability bound (log scale)

## Interpreting Results

### Error Metrics

- **X₀ error < 10%**: Initial memory model is accurate
- **Xₘₐₓ error < 15%**: Peak memory model is accurate
- **κ error < 20%**: Supremum constant is useful approximation

### OOM Bound Validation

- **≥ 90% valid**: The bound `P(sup Y > x) ≤ exp(-2|d|x/v)` holds
- **< 90% valid**: The bound may be too loose or assumptions violated

### Dataset Comparison

| Characteristic | Alpaca | ShareGPT |
|---------------|--------|----------|
| Output distribution | Concentrated | Heavy-tailed |
| p (termination prob) | Higher | Lower |
| κ (supremum) | Smaller | Larger |
| Expected accuracy | Better | May have more variance |

## Dependencies

```bash
pip install vllm torch transformers numpy matplotlib tqdm datasets
```

## Notes

1. **Real inference**: This experiment runs actual LLM inference and measures real GPU memory, providing ground-truth validation of the theoretical model.

2. **Memory measurement**: Uses `torch.cuda.memory_allocated()` for current memory and `torch.cuda.max_memory_allocated()` for peak memory tracking.

3. **Geometric assumption**: The theoretical model assumes geometric output lengths. ShareGPT's heavy-tail distribution may cause larger deviations from theory.

4. **Steady-state assumption**: The age distribution formula assumes steady-state operation after initial transient.

5. **Conservative bounds**: The OOM probability bound is designed to be conservative (upper bound), so observed probabilities should be lower.

6. **GPU required**: This experiment requires a CUDA-capable GPU to run vLLM inference and measure GPU memory.
