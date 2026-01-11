export CUDA_VISIBLE_DEVICES=3,4

# ============================================
# 1. Original vLLM Scheduler (Baseline)
# ============================================
echo "Running Original vLLM Scheduler tests..."
for dataset in alpaca lmsys; do
    python experiments/run_original_scheduler.py \
        --tensor-parallel-size 2 \
        --dataset $dataset \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/ \
        --disable-thinking \
        --gpu-memory-utilization 0.8
done

sleep 2

# ============================================
# 2. P/D Competition - Fixed k* (offline)
# ============================================
echo "Running Fixed k* tests..."
for dataset in alpaca lmsys; do
    python experiments/run_fixed_k.py \
        --tensor-parallel-size 2 \
        --use-offline-kstar \
        --dataset $dataset \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/fixed/ \
        --disable-thinking \
        --gpu-memory-utilization 0.8
done

sleep 2

# ============================================
# 3. P/D Competition - Dynamic k* (online)
# ============================================
echo "Running Dynamic k* tests..."
for dataset in alpaca lmsys; do
    python experiments/run_dynamic_kstar.py \
        --tensor-parallel-size 2 \
        --ema-alpha 0.3 \
        --dataset $dataset \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/dynamic/ \
        --disable-thinking \
        --gpu-memory-utilization 0.8
done

sleep 2

# ============================================
# 4. K Sweep (find empirical optimal k_hat)
# ============================================
echo "Running K Sweep tests..."
for dataset in alpaca lmsys; do
    python experiments/run_k_sweep.py \
        --tensor-parallel-size 2 \
        --dataset $dataset \
        --model Qwen/Qwen3-8B \
        --output-dir ./experiment_results/sweep/ \
        --disable-thinking \
        --gpu-memory-utilization 0.8
done

echo "All experiments completed!"
