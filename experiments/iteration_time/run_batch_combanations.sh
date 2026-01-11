python benchmark_batch_combinations.py \
  --model Qwen/Qwen3-4B \
  --prefill-sizes 256,512,768,1024,1280,1536,2048 \
  --decode-context-lens 256 \
  --decode-counts 64,128,256,512 \
  --num-warmup 5 \
  --num-iterations 20 \
  --output-json results2.json