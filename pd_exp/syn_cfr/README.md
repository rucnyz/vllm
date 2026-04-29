# CFR Synthetic Workload Experiments

Experiments for the CFR (constant-failure-rate / geometric output length)
scenarios that back the three TODO blocks in
`vllm-paper/sections/evaluation.tex`.

## What's here

| File | Backs evaluation.tex line | Purpose |
| --- | --- | --- |
| `common_cfr.sh` | (helper) | GPU auto-detection, calibration resolution, scheduler env setup |
| `run_grid_search_cfr.sh` | line 45 (TODO #1) | $(B,N)$ grid for v1 vs EB($\hat k$) on three CFR workloads |
| `run_validation_cfr.sh` | line 36 (TODO #2) | Online estimator accuracy + OOM rate vs $\varepsilon$ |
| `run_adaptive_selector_cfr.sh` | line 63 (TODO #3) | MB / EB / ADA + diagnostic $\Delta(N)$ |
| `analyze_cfr_e2e.py` | TODO #1 | Tables + heatmaps |
| `analyze_cfr_validation.py` | TODO #2 | Estimator-error / OOM-rate table |
| `analyze_cfr_selector.py` | TODO #3 | Selector-choice / gap table |

All run scripts auto-detect available GPUs (via `nvidia-smi`), resolve
the matching calibration file (`pd_calibration_<MODEL_SHORT>_<GPU_TAG>.json`
or fall back to the legacy `pd_calibration_<MODEL_SHORT>.json`), and write
results under `outputs/<experiment>/<GPU_TAG>_<MODEL_SHORT>/...`.

## Quick start

```bash
cd /scratch/yuzhou/projects/vllm/pd_exp/syn_cfr

# 1. Grid search (~6-10 hours on 4 GPUs)
./run_grid_search_cfr.sh
python analyze_cfr_e2e.py outputs/e2e_grid_search/H200_Qwen3-8B

# 2. Controller validation (~30 minutes)
./run_validation_cfr.sh
python analyze_cfr_validation.py outputs/controller_validation/H200_Qwen3-8B

# 3. Adaptive selector (~30 minutes)
./run_adaptive_selector_cfr.sh
python analyze_cfr_selector.py outputs/adaptive_selector/H200_Qwen3-8B
```

## Common environment overrides

- `MODEL=Qwen/Qwen3-8B` (default).
- `GPUS=0,1,2,3` skip auto-selection.
- `SKIP_EXISTING=0` re-run experiments whose `bench_*.json` already exist.
- `NUM_PROMPTS=4000`, `MAX_CONCURRENCY=2048` (paper defaults).
- `VLLM_PD_OOM_TOLERANCE=0.01` (ε for memory-safe N̂).
- `VLLM_PD_AUTO_COMPUTE_N=1` (default in EB / ADA — turn off with `=0` to
  pin N to `--max-num-seqs`).

For the adaptive selector, an offline kernel sweep gives more accurate
$\Delta(N)$:
```bash
export VLLM_PD_BETA_MB_E=...    # f(\bar r)
export VLLM_PD_ALPHA_MB=...
export VLLM_PD_CP_COST_A=...    # f(r)=a+br+cr^2
export VLLM_PD_CP_COST_B=...
export VLLM_PD_CP_COST_C=...
./run_adaptive_selector_cfr.sh
```

## Smoke test (CFR + EB > MB?)

To check the CFR midpoint scheduler beats v1 before running the full grid:

```bash
SCHEDULERS="v1 eb_khat" \
SCENARIOS="balanced" \
BS_VALUES="1024" \
TB_VALUES="14336" \
NUM_PROMPTS=1000 \
./run_grid_search_cfr.sh 1
python analyze_cfr_e2e.py outputs/e2e_grid_search/H200_Qwen3-8B
```

## Scheduler implementation notes

- Existing IFR mode (`VLLM_PD_K_MODE=ifr`) is untouched — same behaviour
  as before.
- New CFR mode (`VLLM_PD_K_MODE=cfr`) uses the **exact** $\theta_0$ formula
  $\theta/(1-\theta)+\ln(1-\theta) = (-\ln(1-p_0))\alpha_p/\alpha_d$
  and the **midpoint** $\hat k$ from Eq. midpoint_k.
- `VLLM_PD_AUTO_COMPUTE_N=1` enables the memory-safe $\hat N$ from
  Proposition prop:memory; leave at `0` to pin N to `--max-num-seqs`.
- The auto / ADA path (`VLLM_PD_SCHEDULER_MODE=auto`) reuses the existing
  THETA+ online switcher; with `VLLM_PD_K_MODE=cfr` the inner EB uses
  the midpoint construction.
- Schedule stats include the cfr_update_history (per-update snapshot of
  $\hat p_0$, $\hat\mu_L$, $\theta_0$, $\hat k$, $\hat N$, $\Delta(N)$,
  cumulative OOM count) and the existing mode_switch_history.
