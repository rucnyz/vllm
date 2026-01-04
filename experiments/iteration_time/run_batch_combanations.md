
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
    --output-json results2.json