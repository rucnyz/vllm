# pd_exp 实验目录

## 前置条件: 硬件校准

所有 PD Scheduler 实验都需要硬件校准文件。serve/ 下的脚本会**自动运行校准**（如果文件不存在）。
也可手动运行或通过 `VLLM_PD_CALIBRATION_FILE` 环境变量指定：

```bash
python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B
```

## 目录结构

```
pd_exp/
├── common.sh                     # 公共函数库 (所有实验脚本共用)
├── dataset_utils.py              # 数据集加载工具
├── serve/                        # Online serving 实验 (IFR controller 验证)
│   ├── run_distribution_shift.sh # Input/output 分布突变
│   ├── run_concurrency_shift.sh  # 并发水平突变
│   ├── generate_distribution_shift_dataset.py
│   └── plot_distribution_shift.py
├── syn/                          # 合成 workload 参数搜索
│   ├── run_kstar_sweep.sh        # k* / θ* 参数扫描
│   ├── run_grid_search.sh        # TB × BS 网格搜索
│   └── analyze_*.py / plot_*.py
├── real/                         # 真实 workload 实验
├── multiturn/                    # 多轮对话实验
├── attention_benchmark/          # Flash Attention 基准测试
└── outputs/                      # 实验输出目录
```

## serve/ - Online Serving 实验

验证 IFR controller 在 workload 突变时的适应能力。对比 `baseline`（vLLM v1 默认）vs `pd_ifr`（自适应 θ\*）vs `pd_ratio`（固定 θ\*）。

### Distribution Shift (input/output 分布突变)

默认 3 phase: prefill-heavy (1024:128) → balanced (512:512) → decode-heavy (128:1024)。

```bash
bash pd_exp/serve/run_distribution_shift.sh [GPU_ID]

# 自定义
TB=8192 BS=1024 PHASES="1024:128,128:1024" \
    bash pd_exp/serve/run_distribution_shift.sh 0

# 画图
python pd_exp/serve/plot_distribution_shift.py <output_dir>
```

### Concurrency Shift (并发突变)

Server 全程不重启，IFR 状态连续。每个 phase 相同 prompt 分布，只改变并发度。

```bash
bash pd_exp/serve/run_concurrency_shift.sh [GPU_ID]

# 自定义并发阶段 (每个 phase 独立控制数量: concurrency:num_prompts)
CONCURRENCY_PHASES="32:500,2048:4000,500:2000" \
    bash pd_exp/serve/run_concurrency_shift.sh 0

# 不指定数量则使用默认值 NUM_PROMPTS_PER_PHASE=2000
CONCURRENCY_PHASES="64,1024,256" \
    bash pd_exp/serve/run_concurrency_shift.sh 0
```

> 所有环境变量及默认值见各脚本头部注释。

#### 标准 test case (H200, Qwen3-8B)

```bash
CONCURRENCY_PHASES="32:1000,2048:3000,256:2000" \
    bash pd_exp/serve/run_concurrency_shift.sh 0
```

参考结果 (TB=18432, BS=2048, input~512, output~256):

| Phase | Conc. | Scheduler | Throughput (tok/s) | TTFT (ms) | TPOT (ms) |
|-------|-------|-----------|--------------------|-----------|-----------|
| 1 (低) | 32 | baseline | **4166** | **41** | 7.45 |
| | | pd_ifr | 3039 | 963 | **6.59** |
| | | pd_ratio | 2960 | 964 | 6.77 |
| 2 (高) | 2048 | baseline | 10104 | **16823** | 104.71 |
| | | pd_ifr | **10875** | 23172 | **49.53** |
| | | pd_ratio | 10757 | 23050 | 51.70 |
| 3 (中) | 256 | baseline | **9144** | **337** | 25.99 |
| | | pd_ifr | 8770 | 2498 | **18.30** |
| | | pd_ratio | 8623 | 2732 | 17.68 |

要点:
- 高并发 (2048) 时 PD 吞吐反超 baseline (+7.6%)，**TPOT 仅为 baseline 的一半** (49.5 vs 104.7ms)
- PD 的 TPOT 在所有 phase 均优于 baseline，prefill/decode 分离有效减少了 decode 被 prefill 打断的干扰
- 代价是 TTFT 更高，prefill 需要等待 decode slot

### 2-GPU 公平对比 (baseline vs pd_ifr vs pd_ratio vs disagg)

在相同 GPU 数量 (2) 下对比：baseline/pd_ifr/pd_ratio 用 TP=2 单实例，disagg 用 vLLM 官方 P/D 分离（1 prefill GPU + 1 decode GPU）。

```bash
bash pd_exp/serve/run_2gpu_comparison.sh [GPU1] [GPU2]

# 自定义
MAX_CONCURRENCY=64 NUM_PROMPTS=1000 \
    bash pd_exp/serve/run_2gpu_comparison.sh 0 1

# 跳过 disagg (高并发下可能 hang)
SKIP_DISAGG=1 MAX_CONCURRENCY=512 \
    bash pd_exp/serve/run_2gpu_comparison.sh 0 1
```

参考结果 (H200 x2, Qwen3-8B, concurrency=64, input~512, output~256):

| Scheduler | 方案 | Throughput (tok/s) | TTFT (ms) | TPOT (ms) |
|-----------|------|--------------------|-----------|-----------|
| baseline | TP=2 | **8538** | **56** | 7.12 |
| disagg | P/D 分离 | 7147 | 138 | 8.19 |
| pd_ratio | TP=2 | 6723 | 831 | **5.94** |
| pd_ifr | TP=2 | 6569 | 799 | 6.20 |

要点:
- baseline 在低并发下吞吐最高，chunked prefill 开销足够小
- disagg TTFT 优于 PD scheduler (138 vs ~800ms)，因为有专用 prefill GPU，但高并发 (512+) 下 KV transfer 会 hang
- PD scheduler 的 TPOT 最优 (5.94ms)，decode 质量最好

## syn/ - 参数搜索实验

### kstar_sweep - 参数扫描

对比 baseline / kstar / kratio / ratio_auto / direct 在不同 scenario 下的性能。

```bash
cd pd_exp/syn
./run_kstar_sweep.sh [MAX_GPUS]

# 只运行部分模式
RUN_BASELINE=1 RUN_DIRECT=1 RUN_RATIO_AUTO=0 ./run_kstar_sweep.sh 4
```

### grid_search - TB x BS 网格搜索

在所有 (TB, BS) 组合下对比 4 种调度策略。

```bash
cd pd_exp/syn
./run_grid_search.sh [MAX_GPUS]

# 分析
python pd_exp/syn/analyze_grid_search.py <output_dir>
```

## attention_benchmark/ - Flash Attention 基准测试

```bash
python pd_exp/attention_benchmark/benchmark_flash_attn_sweep.py \
    --batch-size 512 --output results.json
```
