
 python benchmark_batch_combinations.py \
    --model Qwen/Qwen3-4B \
    --prefill-sizes 128,256,512,640,768,896,1024,2048 \
    --decode-counts 128,256,512,640,768,896,1024,2048 \
    --decode-context-lens 32 \
    --pure-prefill-sizes 128,256,512,640,768,896,1024,2048,3072,4096 \
    --pure-decode-counts 128,256,512,640,768,896,1024,2048,3072,4096 \
    --max-num-seqs 4096 \
    --max-num-batched-tokens 16384 \
    --num-warmup 5 \
    --num-iterations 20 \
    --output-json results_separate.json
 
python benchmark_batch_combinations.py \
    --model Qwen/Qwen3-4B \
    --prefill-sizes 128,256,384,512,768,1024,1536,2048,3072,4096 \
    --decode-counts 64,128,192,256,384,512,768,1024,1536,2048,3072 \
    --decode-context-lens 32 \
    --pure-prefill-sizes 128,256,512,1024,2048,3072,4096,6144,8192 \
    --pure-decode-counts 64,128,256,512,1024,2048,3072,4096 \
    --max-num-seqs 4096 \
    --max-num-batched-tokens 16384 \
    --num-warmup 5 \
    --num-iterations 20 \
    --output-json results_full_step.json


CUDA_VISIBLE_DEVICES=2 python benchmark_batch_combinations_full_step.py \
    --model Qwen/Qwen3-4B \
    --prefill-sizes 128,256,384,512,768,1024,1536,2048,3072,4096 \
    --decode-counts 64,128,192,256,384,512,768,1024,1536,2048,3072 \
    --decode-context-lens 32 \
    --pure-prefill-sizes 128,256,512,1024,2048,3072,4096,6144,8192 \
    --pure-decode-counts 64,128,256,512,1024,2048,3072,4096 \
    --max-num-seqs 4096 \
    --max-num-batched-tokens 16384 \
    --num-warmup 5 \
    --num-iterations 20 \
    --output-json results_full_step.json



```shell
python plot_benchmark_results.py results2.json
```
