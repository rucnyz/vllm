```shell
cd yuzhou
# test flash attention
python attention_benchmark/benchmark_flash_attn_sweep.py \
      --batch-size 512 \
      --output attention_benchmark/sweep_results.json
```