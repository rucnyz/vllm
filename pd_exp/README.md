# pd_exp 实验目录

## 前置条件: 硬件校准

所有 PD Scheduler 实验都需要硬件校准文件，运行前必须先执行：

```bash
python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B
# 输出: pd_exp/outputs/pd_calibration_Qwen3-8B.json
# 也可通过 VLLM_PD_CALIBRATION_FILE 环境变量指定其他路径
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

验证 IFR controller 在 workload 突变时的适应能力。对比 `pd_ifr`（自适应 θ\*）vs `pd_ratio`（固定 θ\*）。

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

# 自定义
TB=8192 BS=1024 CONCURRENCY_PHASES="64,1024,256" \
    bash pd_exp/serve/run_concurrency_shift.sh 0
```

> 所有环境变量及默认值见各脚本头部注释。

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
