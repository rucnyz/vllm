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