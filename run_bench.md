# Run PD Experiments in Docker

## Quick Start (any machine)

```bash
docker pull yuzhounie/vllm-pd-exp:latest

docker run --gpus '"device=3"' --rm \
  --entrypoint bash \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/pd_exp_outputs:/workspace/pd_exp/outputs \
  yuzhounie/vllm-pd-exp:latest -c '
    SCHEDULERS="baseline,pd_ifr,pd_auto" \
    CONCURRENCY_PHASES="32:4000" \
    VLLM_PD_CP_COST_A=5.14672e-05 \
    VLLM_PD_CP_COST_B=1.73412e-04 \
    VLLM_PD_CP_COST_C=-1.48754e-04 \
    VLLM_PD_MODE_SWITCH_DELTA=0.000001 \
    bash pd_exp/serve/run_concurrency_shift.sh 0'
```

- Change `device=3` to select GPU. Inside container GPU is always `0`.
- Results are saved to `./pd_exp_outputs/` on the host.
- First run auto-calibrates the hardware (~2min). To reuse a calibration file, mount it:
  ```
  -v /path/to/pd_calibration_Qwen3-8B.json:/workspace/pd_exp/outputs/pd_calibration_Qwen3-8B.json:ro
  ```

## Build from source (only needed to update the image)

```bash
git clone https://github.com/rucnyz/vllm.git && cd vllm && git checkout pd_competition
bash pd_exp/docker-build.sh
docker tag vllm-pd-exp yuzhounie/vllm-pd-exp:latest
docker push yuzhounie/vllm-pd-exp:latest
```

Rebuild experiment layer only (after editing `pd_exp/`):

```bash
bash pd_exp/docker-build.sh --exp
```
