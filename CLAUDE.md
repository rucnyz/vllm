# CLAUDE.md

This file provides guidance for Claude Code when working with this repository.

## Project Overview

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). Originally developed at UC Berkeley, it is now a PyTorch Foundation project.

Key features:
- PagedAttention for efficient KV cache memory management
- Continuous batching of incoming requests
- CUDA/HIP graph execution
- Quantization support (GPTQ, AWQ, INT4, INT8, FP8)
- OpenAI-compatible API server
- Multi-GPU support (tensor, pipeline, data, expert parallelism)

## Repository Structure

```
vllm/
├── vllm/                    # Main Python package
│   ├── v1/                  # V1 architecture (current)
│   │   ├── core/            # Core scheduling logic
│   │   │   ├── sched/       # Scheduler implementation
│   │   │   ├── kv_cache_*.py # KV cache management
│   │   │   └── block_pool.py # Memory block management
│   │   ├── engine/          # V1 engine implementation
│   │   ├── executor/        # Execution backends
│   │   ├── worker/          # Worker processes
│   │   └── sample/          # Sampling logic
│   ├── entrypoints/         # API entry points
│   │   ├── cli/             # CLI commands
│   │   ├── openai/          # OpenAI-compatible API
│   │   └── llm.py           # LLM class for offline inference
│   ├── model_executor/      # Model execution
│   │   ├── models/          # Model implementations
│   │   ├── layers/          # Custom layers
│   │   └── model_loader/    # Model loading utilities
│   ├── attention/           # Attention implementations
│   ├── distributed/         # Distributed inference
│   ├── lora/                # LoRA support
│   └── multimodal/          # Multi-modal model support
├── csrc/                    # C++/CUDA kernels
├── tests/                   # Test suite
├── benchmarks/              # Performance benchmarks
├── examples/                # Usage examples
├── docs/                    # Documentation source
└── requirements/            # Dependency files
```

## Build and Development

### Prerequisites
- Python 3.10-3.13
- CUDA toolkit (for GPU support)
- CMake >= 3.26.1

### Installation (Development)
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .

# Or with specific requirements
pip install -r requirements/dev.txt
```

### Building from Source
The project uses setuptools with CMake for C++ extensions:
```bash
pip install -e . --no-build-isolation
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_config.py

# Run with markers
pytest -m "not slow_test" tests/

# Run V1-specific tests
pytest tests/v1/
```

Test markers defined in pyproject.toml:
- `slow_test` - Long-running tests
- `core_model` - Core model tests for PRs
- `cpu_model` - CPU-only model tests
- `distributed` - Distributed GPU tests
- `skip_v1` - Skip for V1 architecture

## CLI Usage

```bash
# Start OpenAI-compatible server
vllm serve <model_name>

# Run offline inference
vllm complete --model <model_name> --prompt "Hello"

# Chat mode
vllm chat --model <model_name>

# Benchmarking
vllm bench serve --model <model_name>
vllm bench throughput --model <model_name>
```

## Key Entry Points

- **CLI**: `vllm.entrypoints.cli.main:main`
- **LLM Class**: `vllm.entrypoints.llm:LLM` (offline inference)
- **OpenAI Server**: `vllm.entrypoints.openai.api_server`
- **AsyncLLMEngine**: `vllm.engine.async_llm_engine:AsyncLLMEngine`

## Code Style

- Linting: ruff (configured in pyproject.toml)
- Type checking: mypy with pydantic plugin
- Pre-commit hooks configured in `.pre-commit-config.yaml`

```bash
# Run linting
ruff check .

# Run formatting
ruff format .

# Run type checking
mypy vllm/
```

## Architecture Notes

### V1 Architecture
The V1 architecture (in `vllm/v1/`) is the current production architecture featuring:
- Optimized execution loop with zero-overhead prefix caching
- Cleaner code organization
- Enhanced multimodal support
- Improved scheduler in `vllm/v1/core/sched/scheduler.py`

### Scheduler
The scheduler (`vllm/v1/core/sched/scheduler.py`) manages:
- Request queuing and prioritization
- KV cache block allocation
- Prefill/decode scheduling decisions
- Memory management coordination

### KV Cache Management
- `block_pool.py` - Block-level memory management
- `kv_cache_manager.py` - High-level cache coordination
- `kv_cache_utils.py` - Utility functions for cache operations

## Environment Variables

Key environment variables (see `vllm/envs.py` for full list):
- `VLLM_USE_V1` - Enable V1 architecture
- `VLLM_ATTENTION_BACKEND` - Select attention backend
- `VLLM_PP_LAYER_PARTITION` - Pipeline parallelism configuration
- `VLLM_TRACE_FUNCTION` - Enable function tracing

## Common Development Tasks

### Adding a New Model
1. Create model file in `vllm/model_executor/models/`
2. Register in model registry
3. Add tests in `tests/models/`

### Modifying the Scheduler
Core scheduler logic is in `vllm/v1/core/sched/scheduler.py`. Key methods:
- `schedule()` - Main scheduling loop
- `_schedule_prefills()` - Handle prefill requests
- `_schedule_decodes()` - Handle decode requests

### Running Benchmarks
```bash
# Serving benchmark
python benchmarks/benchmark_serving.py

# Throughput benchmark
python benchmarks/benchmark_throughput.py

# Multi-turn conversation benchmark
python benchmarks/multi_turn/benchmark_serving_multi_turn_threaded.py
```

## Experiments Directory

The `experiments/` directory contains research experiments for PD (Prefill-Decode) scheduler optimization. This is experimental work comparing different scheduling strategies.

### Directory Structure

```
experiments/
├── iteration_time/          # Batch iteration time profiling
├── schedule_stats/          # Scheduler statistics collection
├── serve/                   # Online serving throughput experiments
│   └── online_khat/         # Dynamic k* (k-hat) experiments
└── archive/                 # Archived experiments
```

### iteration_time/
Benchmarks for measuring model iteration time with different batch combinations:
- **Purpose**: Profile prefill/decode iteration times to understand batch throughput characteristics
- **Key script**: `benchmark_batch_combinations.py`
- **Output**: JSON results and visualization plots

```bash
# Run batch combination benchmark
python experiments/iteration_time/benchmark_batch_combinations.py \
    --model Qwen/Qwen3-4B \
    --prefill-sizes 128,256,512,1024 \
    --decode-counts 128,256,512,1024 \
    --output-json results.json

# Generate plots
python experiments/iteration_time/plot_benchmark_results.py
```

### schedule_stats/
Collects and analyzes scheduler statistics during online/offline inference:
- **Purpose**: Understand scheduler behavior (prefill/decode ratio, queue depth, etc.)
- **Key scripts**: `run_benchmark_with_stats.py`, `plot_schedule_stats.py`
- **Subdirs**: `offline/`, `online/`, `online_concurrency*/`

```bash
# Start server with stats collection
VLLM_COLLECT_SCHEDULE_STATS=1 \
VLLM_SCHEDULE_STATS_FILE=schedule_stats.json \
    vllm serve Qwen/Qwen3-8B --port 8000

# Run benchmark and collect stats
python experiments/schedule_stats/run_benchmark_with_stats.py
```

### serve/
Online serving throughput experiments comparing scheduling strategies:
- **Purpose**: Fair comparison between baseline, fixed k, and dynamic k* schedulers
- **Key scripts**: `run_serve_throughput.py`, `sweep_k_star.py`
- **Tools**: Uses `genai-bench` for throughput measurement

#### PD Scheduler Environment Variables
```bash
# Enable PD scheduler
VLLM_USE_PD_SCHEDULER=1

# Enable dynamic k* optimization
VLLM_PD_ENABLE_DYNAMIC_KSTAR=1

# Set fixed k* value (when dynamic is disabled)
VLLM_PD_K_STAR=16

# Model parameters (prefill/decode time coefficients)
VLLM_PD_ALPHA_P=0.006528784356021418
VLLM_PD_BETA_P=6.498792400220424e-06
VLLM_PD_ALPHA_D=0.004303444935141221
VLLM_PD_BETA_D=0.00023557651251992446
```

#### serve/online_khat/
Grid search and analysis for optimal k* values:
- **Key scripts**:
  - `run_grid_search.sh` - Run grid search experiments
  - `run_request_scaling.sh` - Request scaling experiments
  - `analyze_grid_search.py` - Analyze grid search results
  - `analyze_request_scaling.py` - Analyze scaling behavior

```bash
# Run grid search for optimal parameters
bash experiments/serve/online_khat/run_grid_search.sh

# Analyze results
python experiments/serve/online_khat/analyze_grid_search.py
```


