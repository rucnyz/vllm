# Run PD Experiments in Docker

## Setup (any machine)

```bash
docker pull yuzhounie/vllm-pd-exp:latest
```

## 2-GPU comparison (baseline vs THETA vs THETA+ vs disagg)

```bash
docker run --gpus '"device=0,1"' --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/pd_exp_outputs:/workspace/pd_exp/outputs \
  yuzhounie/vllm-pd-exp:latest bash -c '
    MAX_CONCURRENCY=64 NUM_PROMPTS=4000 INPUT_LEN=512 OUTPUT_LEN=256 \
    VLLM_PD_CP_COST_A=5.14672e-05 VLLM_PD_CP_COST_B=1.73412e-04 \
    VLLM_PD_CP_COST_C=-1.48754e-04 VLLM_PD_MODE_SWITCH_DELTA=0.000001 \
    bash pd_exp/serve/run_2gpu_comparison.sh 0 1'
```

- Change `device=0,1` to select GPUs. Inside container GPUs are renumbered from `0`.
- Results saved to `./pd_exp_outputs/` on the host.

## Build & push (when code changes)

```bash
# Full rebuild (scheduler code changed): ~1.5h first time, cached after
bash pd_exp/docker-build.sh

# Experiment scripts only (pd_exp changed): ~1min
bash pd_exp/docker-build.sh --exp

# Push
docker tag vllm-pd-exp yuzhounie/vllm-pd-exp:latest
docker push yuzhounie/vllm-pd-exp:latest
```
