#!/bin/bash
# Sweep k* from 1 to 128 (step=4) and run benchmarks
#
# Usage:
#   chmod +x experiments/serve/sweep_k_star_manual.sh
#   ./experiments/serve/sweep_k_star_manual.sh
#
# Prerequisites:
#   - Export timing parameters (VLLM_PD_ALPHA_P, etc.)
#   - Prepare dataset CSV

set -e

# Configuration
GPU=5
MODEL="Qwen/Qwen3-8B"
PORT=8567
API_KEY="7355608"
MAX_NUM_SEQS=128
DATASET="./experiments/serve/alpaca_prompts.csv"
OUTPUT_BASE="./experiment_results/k_star_sweep"
MAX_REQUESTS=500
MAX_TIME=120
CONCURRENCY=128

# Create output directory
mkdir -p "$OUTPUT_BASE"

# k* values: 1, 4, 8, 12, ..., 128
K_VALUES=(1 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96 100 104 108 112 116 120 124 128)

echo "Will test k* values: ${K_VALUES[*]}"
echo "Total experiments: ${#K_VALUES[@]} + 1 baseline"

# Results file
RESULTS_FILE="$OUTPUT_BASE/results_summary.csv"
echo "k_star,scheduler,throughput,ttft_mean,tpot_mean,latency_mean" > "$RESULTS_FILE"

# ===== First: Run BASELINE (original vLLM scheduler) =====
echo ""
echo "============================================================"
echo "Testing BASELINE (original vLLM scheduler)"
echo "============================================================"

EXP_DIR="$OUTPUT_BASE/baseline"
mkdir -p "$EXP_DIR"

# Start vLLM server with original scheduler
echo "Starting BASELINE vLLM server..."
CUDA_VISIBLE_DEVICES=$GPU \
    VLLM_USE_PD_SCHEDULER=0 \
    VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
    python -m vllm.entrypoints.cli.main serve "$MODEL" \
        --port $PORT \
        --max-num-seqs $MAX_NUM_SEQS \
        --gpu-memory-utilization 0.9 \
        --api-key "$API_KEY" \
        > "$EXP_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
MAX_WAIT=180
WAITED=0
while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Server failed to start within ${MAX_WAIT}s"
        kill $SERVER_PID 2>/dev/null || true
        break
    fi
done

if [ $WAITED -lt $MAX_WAIT ]; then
    echo "Server is ready after ${WAITED}s"
    sleep 5

    # Run benchmark
    echo "Running benchmark..."
    genai-bench benchmark \
        --api-backend vllm \
        --api-key "$API_KEY" \
        --api-base "http://localhost:$PORT" \
        --api-model-name "$MODEL" \
        --model-tokenizer "$MODEL" \
        --task text-to-text \
        --experiment-base-dir "$EXP_DIR" \
        --dataset-path "$DATASET" \
        --dataset-prompt-column prompt \
        --max-time-per-run $MAX_TIME \
        --max-requests-per-run $MAX_REQUESTS \
        --num-concurrency $CONCURRENCY \
        2>&1 | tee "$EXP_DIR/benchmark.log"
fi

# Stop server
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Wait before next experiment
sleep 10

echo "Completed BASELINE"

# ===== Then: Run P/D scheduler with different k* values =====
for K in "${K_VALUES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Testing k* = $K"
    echo "============================================================"

    EXP_DIR="$OUTPUT_BASE/k_star_$(printf '%03d' $K)"
    mkdir -p "$EXP_DIR"

    # Start vLLM server in background
    # Note: VLLM_PD_ALPHA_P/BETA_P/ALPHA_D/BETA_D are NOT needed when
    # VLLM_PD_K_STAR is explicitly set - the scheduler uses the fixed k* directly
    echo "Starting vLLM server with k*=$K..."
    CUDA_VISIBLE_DEVICES=$GPU \
        VLLM_USE_PD_SCHEDULER=1 \
        VLLM_PD_ENABLE_DYNAMIC_KSTAR=0 \
        VLLM_PD_K_STAR=$K \
        python -m vllm.entrypoints.cli.main serve "$MODEL" \
            --port $PORT \
            --max-num-seqs $MAX_NUM_SEQS \
            --gpu-memory-utilization 0.9 \
            --api-key "$API_KEY" \
            > "$EXP_DIR/server.log" 2>&1 &

    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    MAX_WAIT=180
    WAITED=0
    while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "Server failed to start within ${MAX_WAIT}s"
            kill $SERVER_PID 2>/dev/null || true
            continue 2
        fi
    done
    echo "Server is ready after ${WAITED}s"

    # Give it a moment to stabilize
    sleep 5

    # Run benchmark
    echo "Running benchmark..."
    genai-bench benchmark \
        --api-backend vllm \
        --api-key "$API_KEY" \
        --api-base "http://localhost:$PORT" \
        --api-model-name "$MODEL" \
        --model-tokenizer "$MODEL" \
        --task text-to-text \
        --experiment-base-dir "$EXP_DIR" \
        --dataset-path "$DATASET" \
        --dataset-prompt-column prompt \
        --max-time-per-run $MAX_TIME \
        --max-requests-per-run $MAX_REQUESTS \
        --num-concurrency $CONCURRENCY \
        2>&1 | tee "$EXP_DIR/benchmark.log"

    # Stop server
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true

    # Wait before next experiment
    sleep 10

    echo "Completed k* = $K"
done

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "Results are in: $OUTPUT_BASE"
echo "============================================================"

# After all experiments, generate plots
echo "Generating plots..."
python experiments/serve/plot_k_star_sweep.py --input-dir "$OUTPUT_BASE"
