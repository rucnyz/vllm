## ShareGPT

```shell
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python pd_exp/export_dataset.py \
      --dataset sharegpt \
      --model Qwen/Qwen3-8B \
      --num-samples 4000 \
      --output ./pd_exp/outputs/sharegpt_prompts.jsonl
rm -rf ShareGPT_V3_unfiltered_cleaned_split.json

python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B
export VLLM_PD_CALIBRATION_FILE=/scr/rucnyz/projects/vllm/pd_exp/outputs/pd_calibration.json

# ShareGPT: 关闭 thinking，输出长度 500
ENABLE_THINKING=false CUSTOM_OUTPUT_LEN=500 \
    ./pd_exp/syn/run_grid_search_real.sh ./pd_exp/outputs/sharegpt_prompts.jsonl 4

python pd_exp/syn/analyze_grid_search.py pd_exp/outputs/grid_search_sharegpt_prompts_Con_2048_Prompts_4000
```

## numina_math

```bash
python pd_exp/export_dataset.py \
    --dataset numina_math \
    --model Qwen/Qwen3-8B \
    --num-samples 4000 \
    --min-output-len 800 \
    --output ./pd_exp/outputs/numina_math_prompts.jsonl

# numina_math: 开启 thinking (默认)，输出长度 4000
CUSTOM_OUTPUT_LEN=4000 \
    ./pd_exp/syn/run_grid_search_real.sh ./pd_exp/outputs/numina_math_prompts.jsonl 4

# Grid search 结果分析
python pd_exp/syn/analyze_grid_search.py pd_exp/outputs/grid_search_numina_math_prompts_Con_2048_Prompts_4000

# Input/Output 长度统计 (查看真实的 decode-heavy 程度)
python pd_exp/analyze_benchmark_stats.py pd_exp/outputs/grid_search_numina_math_prompts_Con_2048_Prompts_4000
```
