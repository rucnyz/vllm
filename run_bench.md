# Run PD Experiments in Docker

## Setup (any machine)

```bash
docker pull yuzhounie/vllm-pd-exp:latest
```

Container auto-pulls latest `pd_competition` branch code on every startup.

## pd_auto (1 GPU)

```bash
docker run --gpus '"device=3"' --rm \
  --entrypoint bash \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/pd_exp_outputs:/workspace/pd_exp/outputs \
  yuzhounie/vllm-pd-exp:latest -c '
    SCHEDULERS="pd_auto" \
    CONCURRENCY_PHASES="32:4000" \
    VLLM_PD_CP_COST_A=5.14672e-05 \
    VLLM_PD_CP_COST_B=1.73412e-04 \
    VLLM_PD_CP_COST_C=-1.48754e-04 \
    VLLM_PD_MODE_SWITCH_DELTA=0.000001 \
    bash pd_exp/serve/run_concurrency_shift.sh 0'
```

## disagg baseline (2 GPU)

```bash
docker run --gpus '"device=4,5"' --rm \
  --entrypoint bash \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/pd_exp_outputs:/workspace/pd_exp/outputs \
  yuzhounie/vllm-pd-exp:latest -c '
    CONCURRENCY_PHASES="32:4000" \
    bash pd_exp/serve/run_disagg_baseline.sh 0 1'
```

## Notes

- Change `device=X` to select GPU(s). Inside container GPUs are renumbered from `0`.
- Results are saved to `./pd_exp_outputs/` on the host.
- First run auto-calibrates the hardware (~2min). To reuse a calibration file:
  ```
  -v /path/to/pd_calibration_Qwen3-8B.json:/workspace/pd_exp/outputs/pd_calibration_Qwen3-8B.json:ro
  ```

## Build from source (only needed to update the base vLLM image)

```bash
git clone https://github.com/rucnyz/vllm.git && cd vllm && git checkout pd_competition
bash pd_exp/docker-build.sh
docker tag vllm-pd-exp yuzhounie/vllm-pd-exp:latest
docker push yuzhounie/vllm-pd-exp:latest
```
