# pd_exp 实验目录

## 目录结构

```
pd_exp/
├── syn/                          # P/D Scheduler 实验脚本
│   ├── run_kstar_sweep.sh        # k* / θ* 参数扫描实验
│   ├── run_grid_search.sh        # TB × BS 网格搜索实验
│   ├── common.sh                 # 公共函数库
│   ├── analyze_grid_search.py    # 网格搜索结果分析
│   ├── analyze_request_scaling.py
│   └── plot_kstar_sweep.py
├── attention_benchmark/          # Flash Attention 基准测试
└── outputs/                      # 实验输出目录
    └── pd_calibration.json       # 硬件校准文件 (必须)
```

---

## 0. 硬件校准 (必须先执行)

**PD Scheduler 需要硬件时序参数才能准确调度。** 在运行任何实验之前，必须先进行校准。

### 运行校准

```bash
# 默认输出到 pd_exp/outputs/pd_calibration.json
python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B

# 或指定输出路径
python -m vllm.v1.core.sched.calibration \
    --model Qwen/Qwen3-8B \
    --output /path/to/pd_calibration.json
```

### 校准参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | (必须) | 模型名称 |
| `--prefill-sizes` | 256,512,1024,2048,4096 | Prefill 测试大小 |
| `--decode-counts` | 16,32,64,128,256 | Decode batch 测试大小 |
| `--num-iterations` | 20 | 每个测试点的迭代次数 |
| `--output` | pd_exp/outputs/pd_calibration.json | 输出文件路径 |

### 校准输出

```json
{
  "alpha_p": 0.001296,    // Prefill 固定开销 (秒)
  "beta_p": 2.48e-05,     // Prefill 每 token 开销 (秒/token)
  "alpha_d": 0.009158,    // Decode 固定开销 (秒)
  "beta_d": 3.11e-05,     // Decode 每请求开销 (秒/req)
  "model": "Qwen/Qwen3-8B",
  "device_name": "NVIDIA H200",
  "prefill_r2": 0.9975,   // Prefill 拟合 R²
  "decode_r2": 0.9035     // Decode 拟合 R²
}
```

---

## 1. run_kstar_sweep.sh - 参数扫描实验

对比不同调度策略在不同 scenario 下的性能表现。

### 前置条件

必须先完成硬件校准，校准文件保存在 `../outputs/pd_calibration.json`。

### 支持的调度模式

| 模式 | 环境变量 | 说明 |
|------|----------|------|
| baseline | `RUN_BASELINE=1` | vLLM 默认调度器 |
| kstar | `RUN_KSTAR=1` | 固定 k* 值 |
| kratio | `RUN_KRATIO=1` | 固定 θ* (k* = θ* × N) |
| ratio_auto | `RUN_RATIO_AUTO=1` | 动态 θ* (渐进公式计算) |
| direct | `RUN_DIRECT=1` | 动态 k* (DP 算法) |

### 基本用法

```bash
cd /scratch/pd_exp/aproj/vllm/pd_exp/syn

# 默认运行 (baseline + ratio_auto + direct)
./run_kstar_sweep.sh [MAX_GPUS]

# 指定 GPU 数量
./run_kstar_sweep.sh 4
```

### 运行特定模式组合

```bash
# 只运行 baseline 和 direct
RUN_BASELINE=1 RUN_KSTAR=0 RUN_KRATIO=0 RUN_RATIO_AUTO=0 RUN_DIRECT=1 \
    ./run_kstar_sweep.sh 4

# 只运行 ratio_auto
RUN_BASELINE=0 RUN_KSTAR=0 RUN_KRATIO=0 RUN_RATIO_AUTO=1 RUN_DIRECT=0 \
    ./run_kstar_sweep.sh 4

# 只运行 kratio 模式 (需要指定 K_RATIO_VALUES)
RUN_BASELINE=0 RUN_KSTAR=0 RUN_KRATIO=1 RUN_RATIO_AUTO=0 RUN_DIRECT=0 \
    K_RATIO_VALUES="0.2 0.4 0.6 0.8" \
    ./run_kstar_sweep.sh 4

# 运行所有模式
RUN_BASELINE=1 RUN_KSTAR=1 RUN_KRATIO=1 RUN_RATIO_AUTO=1 RUN_DIRECT=1 \
    K_STAR_VALUES="8 16 32 64" \
    K_RATIO_VALUES="0.2 0.4 0.6 0.8" \
    ./run_kstar_sweep.sh 4
```

### 自定义实验参数

```bash
# 自定义模型和请求数
MODEL="Qwen/Qwen3-4B" \
NUM_PROMPTS=2000 \
MAX_CONCURRENCY=1024 \
    ./run_kstar_sweep.sh 2

# 自定义 scenario
SCENARIOS="128 1024|1024 128|512 512|256 2048" \
    ./run_kstar_sweep.sh 4

# 自定义 TB 和 BS
MAX_NUM_BATCHED_TOKENS=16384 \
MAX_NUM_SEQS=1024 \
    ./run_kstar_sweep.sh 4
```

### 指定 GPU

```bash
# 使用特定 GPU
GPUS="0,1,2,3" ./run_kstar_sweep.sh

# 使用 2 张指定 GPU
GPUS="4,5" ./run_kstar_sweep.sh 2
```

### 环境变量一览

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | Qwen/Qwen3-8B | 模型名称 |
| `NUM_PROMPTS` | 4000 | 请求数量 |
| `MAX_CONCURRENCY` | 2048 | 最大并发数 |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | TB 参数 |
| `MAX_NUM_SEQS` | 1024 | BS 参数 |
| `SCENARIOS` | "128 1024\|1024 128\|512 512" | 测试场景 |
| `RUN_BASELINE` | 1 | 是否运行 baseline |
| `RUN_KSTAR` | 0 | 是否运行固定 k* |
| `RUN_KRATIO` | 0 | 是否运行固定 θ* |
| `RUN_RATIO_AUTO` | 1 | 是否运行动态 θ* |
| `RUN_DIRECT` | 1 | 是否运行 direct |
| `K_STAR_VALUES` | 8 16 32 64 128 | k* 扫描值 |
| `K_RATIO_VALUES` | 0.1 0.2 ... 0.9 | θ* 扫描值 |
| `GPUS` | 自动检测 | 指定 GPU 列表 |

### 输出目录结构

```
outputs/kstar_sweep_<timestamp>/
├── experiment_config.json      # 实验配置
├── experiment_queue.txt        # 实验队列
├── progress.txt                # 进度记录
└── in128_out1024/              # 各 scenario 目录
    ├── logs/
    │   ├── baseline.log
    │   ├── ratio_auto.log
    │   └── direct.log
    ├── bench_baseline.json
    ├── bench_ratio_auto.json
    ├── bench_direct.json
    ├── baseline_stats.json
    ├── ratio_auto_stats.json
    └── direct_stats.json
```

---

## 2. run_grid_search.sh - TB × BS 网格搜索

在所有 (TB, BS) 组合下对比 4 种调度策略的性能。

### 前置条件

必须先完成硬件校准，校准文件保存在 `../outputs/pd_calibration.json`。

### 支持的调度策略

| 调度器 | 说明 |
|--------|------|
| baseline | vLLM 默认调度器 |
| pd_ratio | PD 调度器，固定 θ* |
| pd_ratio_auto | PD 调度器，动态 θ* |
| pd_direct | PD 调度器，动态 k* (DP) |

### 基本用法

```bash
cd /scratch/pd_exp/aproj/vllm/pd_exp/syn

# 默认运行
./run_grid_search.sh [MAX_GPUS]

# 指定 GPU 数量
./run_grid_search.sh 4
```

### 自定义参数

```bash
# 自定义模型和请求数
MODEL="Qwen/Qwen3-4B" \
NUM_PROMPTS=2000 \
MAX_CONCURRENCY=1024 \
    ./run_grid_search.sh 4

# 自定义 k_ratio (用于 pd_ratio)
K_RATIO=0.6 ./run_grid_search.sh 4

# 指定 GPU
GPUS="0,1,2,3" ./run_grid_search.sh
```

### 网格搜索参数

默认搜索范围 (在脚本中定义):
- `BS_VALUES`: 256, 512, 1024, 1536, 2048
- `TB_VALUES`: 8192, 10240, 14336, 16384, 18432
- `SCENARIOS`: (128, 1024), (1024, 128), (512, 512)

总实验数 = 5 × 5 × 3 × 4 = 300 个实验

### 输出目录结构

```
outputs/grid_search_Con_2048_Prompts_4000/
├── experiment_config.json
├── experiment_queue.txt
├── progress.txt
└── tb16384/
    └── bs1024/
        └── in128_out1024/
            ├── logs/
            │   ├── baseline.log
            │   ├── pd_ratio.log
            │   ├── pd_ratio_auto.log
            │   └── pd_direct.log
            ├── bench_baseline.json
            ├── bench_pd_ratio.json
            ├── bench_pd_ratio_auto.json
            ├── bench_pd_direct.json
            └── *_stats.json
```

### 分析结果

```bash
python /scratch/pd_exp/aproj/vllm/pd_exp/syn/analyze_grid_search.py \
    outputs/grid_search_Con_2048_Prompts_4000/
```

---

## 3. Flash Attention Benchmark

```bash
cd pd_exp

# 运行 flash attention sweep
python attention_benchmark/benchmark_flash_attn_sweep.py \
    --batch-size 512 \
    --output attention_benchmark/sweep_results.json
```

---

## 常见问题

### Q: 运行脚本时报错 "未找到硬件校准文件"?

必须先运行校准:
```bash
python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B
```

校准文件会自动保存到 `pd_exp/outputs/pd_calibration.json`。

### Q: 如何使用其他位置的校准文件?

```bash
VLLM_PD_CALIBRATION_FILE=/path/to/calibration.json ./run_grid_search.sh 4
```

### Q: 如何只运行部分实验?

对于 `run_kstar_sweep.sh`，使用环境变量控制:
```bash
RUN_BASELINE=1 RUN_DIRECT=1 RUN_RATIO_AUTO=0 ./run_kstar_sweep.sh
```

### Q: 实验中断后如何继续?

目前不支持断点续传，需要重新运行。可以通过修改 `experiment_queue.txt` 删除已完成的实验来手动实现。

### Q: 如何查看实验进度?

```bash
watch -n 5 'wc -l outputs/*/progress.txt'
```

### Q: GPU 内存不足怎么办?

1. 减小 BS 和 TB 参数
2. 使用更小的模型
3. 增加 GPU 数量分散负载
