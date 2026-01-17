```shell
cd yuzhou
# test flash attention
python attention_benchmark/benchmark_flash_attn_sweep.py \
      --batch-size 512 \
      --output attention_benchmark/sweep_results.json
```


```shell
cd experiments/iteration_time
python benchmark_batch_combinations.py --model Qwen/Qwen3-4B --prefill-sizes 128,256,512,640,768,896,1024,2048 --decode-counts 128,256,512,640,768,896,1024,2048 --decode-context-lens 32 --pure-prefill-sizes 128,256,512,640,768,896,1024,2048,3072,4096 --pure-decode-counts 128,256,512,640,768,896,1024,2048,3072,4096 --max-num-seqs 4096 --max-num-batched-tokens 16384 --num-warmup 5 --num-iterations 20 --output-json results.json

python plot_benchmark_results.py --input results.json --ratio-analysis --ratio-tolerance 0.05 --target-ratios "0.05, 0.10,0.20,0.40,0.60,0.80,0.90"
```