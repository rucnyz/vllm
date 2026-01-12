```shell
cd yuzhou
# test flash attention
python attention_benchmark/benchmark_flash_attn.py \
        --batch-size 512 \
        --prefill-len 512 \
        --context-len 512 \
        --output results.json
```