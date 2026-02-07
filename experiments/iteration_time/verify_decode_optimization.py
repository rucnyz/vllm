# experiments/verify_decode_optimization.py
"""
验证假设：Prefill的存在打断了Decode的批处理优化

实验设计：
1. 纯decode N请求 → 测量单步时间 T1
2. 混合批次：N decode + 1个极小prefill → 测量单步时间 T2
3. 如果 T2 >> T1，说明即使1个token的prefill也会打断优化

关键指标：比较 max_query_len=1 vs max_query_len>1 的影响

使用方法：
    python verify_decode_optimization.py --model Qwen/Qwen3-4B
"""

import argparse
import time
import uuid

import numpy as np
import torch

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor import Executor


class GPUTimer:
    """Accurate GPU timing using CUDA events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        torch.cuda.synchronize()
        self.start_event.record()

    def stop(self) -> float:
        """Stop timing and return elapsed time in milliseconds."""
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class DecodeOptimizationVerifier:
    """验证prefill是否打断decode优化"""

    def __init__(self, model: str, decode_context_len: int = 512):
        self.model = model
        self.decode_context_len = decode_context_len
        self.engine_core: EngineCore | None = None

    def setup(self):
        """Initialize EngineCore."""
        print(f"Initializing model: {self.model}")

        engine_args = EngineArgs(
            model=self.model,
            dtype="float16",
            max_num_batched_tokens=16384,
            max_num_seqs=2048,  # 增加到 2048 以支持更多请求
            gpu_memory_utilization=0.9,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            block_size=16,
        )

        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        with set_default_torch_num_threads(1):
            self.engine_core = EngineCore(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
            )

        print("Setup complete!")

    def _add_request(self, prompt_len: int, max_tokens: int = 100000) -> str:
        """Add a new request to the engine."""
        assert self.engine_core is not None

        req_id = f"req_{uuid.uuid4().hex[:8]}"
        prompt_token_ids = list(np.random.randint(100, 10000, size=prompt_len))

        request = EngineCoreRequest(
            request_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=max_tokens),
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        self.engine_core.add_request(
            *self.engine_core.preprocess_add_request(request)
        )
        return req_id

    def _cleanup_all_requests(self):
        """Clean up all requests."""
        if self.engine_core is None:
            return

        running_ids = [req.request_id for req in self.engine_core.scheduler.running]
        waiting_ids = [req.request_id for req in self.engine_core.scheduler.waiting]
        all_ids = running_ids + waiting_ids

        if all_ids:
            self.engine_core.abort_requests(all_ids)

        while self.engine_core.scheduler.has_requests():
            try:
                self.engine_core.step_fn()
            except Exception:
                break

    def _prepare_decode_requests(self, num_decode: int) -> bool:
        """Prepare decode requests by completing their prefill phase."""
        if num_decode == 0:
            return True

        assert self.engine_core is not None

        # Add requests
        for _ in range(num_decode):
            self._add_request(self.decode_context_len, max_tokens=100000)

        # Run until all requests have completed prefill
        tokens_to_process = self.decode_context_len * num_decode
        max_steps = (tokens_to_process // 16384) + 100

        for step in range(max_steps):
            # First run a step to process tokens
            try:
                self.engine_core.step_fn()
            except Exception as e:
                print(f"  Warning: Error during prefill preparation: {e}")
                return False

            # Then check if all requests are in decode phase
            num_running = len(self.engine_core.scheduler.running)
            if num_running >= num_decode:
                all_in_decode = all(
                    req.num_computed_tokens >= req.num_prompt_tokens
                    for req in self.engine_core.scheduler.running
                )
                if all_in_decode:
                    break

        num_in_decode = sum(
            1 for req in self.engine_core.scheduler.running
            if req.num_computed_tokens >= req.num_prompt_tokens
        )
        return num_in_decode >= num_decode

    def _run_single_step(self) -> tuple[int, int, int, float]:
        """Run a single step and measure time."""
        assert self.engine_core is not None

        # Check if there are requests to schedule
        if not self.engine_core.scheduler.has_requests():
            return 0, 0, 0, 0.0

        try:
            scheduler_output = self.engine_core.scheduler.schedule()
        except AssertionError:
            # Scheduler assertion failed (no tokens to schedule)
            return 0, 0, 0, 0.0

        if scheduler_output.total_num_scheduled_tokens == 0:
            return 0, 0, 0, 0.0

        # Count decode vs prefill and get max_query_len
        num_decode = 0
        num_prefill = 0
        max_query_len = 0
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            max_query_len = max(max_query_len, num_tokens)
            if num_tokens == 1:
                num_decode += 1
            else:
                num_prefill += 1

        total_tokens = scheduler_output.total_num_scheduled_tokens

        # Measure execute_model time
        timer = GPUTimer()
        timer.start()
        model_output = self.engine_core.model_executor.execute_model(
            scheduler_output, non_block=False
        )
        if model_output is None:
            model_output = self.engine_core.model_executor.sample_tokens(None)
        elapsed = timer.stop()

        # Update scheduler state
        self.engine_core.scheduler.update_from_output(scheduler_output, model_output)

        return num_decode, num_prefill, max_query_len, elapsed

    def run_pure_decode(self, num_decode: int, num_iterations: int = 10) -> dict:
        """测量纯decode的单步执行时间 (复用decode状态，连续测量多步)"""
        self._cleanup_all_requests()

        success = self._prepare_decode_requests(num_decode)
        if not success:
            return {"error": "Failed to prepare decode requests"}

        # Warmup + Measure: 连续运行多步，每步都是纯decode
        times = []
        max_query_lens = []
        total_steps = 3 + num_iterations  # 3 warmup + num_iterations

        for i in range(total_steps):
            num_d, num_p, max_qlen, elapsed = self._run_single_step()
            # 只有纯 decode (num_p == 0) 才记录
            if i >= 3 and num_d > 0 and num_p == 0:
                times.append(elapsed)
                max_query_lens.append(max_qlen)

        if not times:
            return {"error": "No valid pure decode measurements"}

        return {
            "type": "pure_decode",
            "num_decode": num_decode,
            "prefill_size": 0,
            "mean_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "max_query_len": max_query_lens[0] if max_query_lens else 0,
            "times": times,
        }

    def run_pure_prefill(self, prefill_size: int, num_iterations: int = 10) -> dict:
        """测量纯prefill的单步执行时间"""
        self._cleanup_all_requests()

        times = []

        for i in range(3 + num_iterations):  # 3 warmup + num_iterations
            self._cleanup_all_requests()

            # Add a prefill request
            self._add_request(prefill_size, max_tokens=1)

            # Run one step
            num_d, num_p, max_qlen, elapsed = self._run_single_step()

            if i >= 3 and max_qlen == prefill_size:
                times.append(elapsed)

        if not times:
            return {"error": f"No valid prefill measurements"}

        return {
            "type": "pure_prefill",
            "num_decode": 0,
            "prefill_size": prefill_size,
            "mean_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "max_query_len": prefill_size,
            "times": times,
        }

    def run_mixed_batch(self, num_decode: int, prefill_size: int, num_iterations: int = 10) -> dict:
        """测量混合批次的单步执行时间"""
        times = []
        max_query_lens = []
        debug_printed = False

        for i in range(3 + num_iterations):  # 3 warmup + num_iterations
            self._cleanup_all_requests()

            # Prepare decode requests
            success = self._prepare_decode_requests(num_decode)
            if not success:
                return {"error": "Failed to prepare decode requests"}

            # Add a new prefill request
            self._add_request(prefill_size, max_tokens=1)

            # Run one step
            num_d, num_p, max_qlen, elapsed = self._run_single_step()

            # Debug: 打印第一次的调度情况
            if not debug_printed and i == 0:
                print(f"\n    [Debug] num_d={num_d}, num_p={num_p}, max_qlen={max_qlen}")
                debug_printed = True

            # 验证条件：decode 数量足够，且 max_query_len == prefill_size (表示 prefill 被调度了)
            is_valid = (num_d >= num_decode) and (max_qlen == prefill_size)

            if i >= 3 and is_valid:
                times.append(elapsed)
                max_query_lens.append(max_qlen)

        if not times:
            return {
                "type": "mixed",
                "num_decode": num_decode,
                "prefill_size": prefill_size,
                "mean_time_ms": float('nan'),
                "std_time_ms": float('nan'),
                "max_query_len": 0,
                "times": [],
                "error": f"No valid measurements (last: num_d={num_d}, num_p={num_p}, max_qlen={max_qlen})"
            }

        return {
            "type": "mixed",
            "num_decode": num_decode,
            "prefill_size": prefill_size,
            "mean_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "max_query_len": max_query_lens[0] if max_query_lens else 0,
            "times": times,
        }

    def verify(self):
        """运行验证实验"""
        print("\n" + "=" * 70)
        print("验证假设: Prefill的存在是否打断Decode的批处理优化")
        print("=" * 70)

        results = []

        # 测试配置：分别测量 decode 和 prefill 的边际成本
        # 优化：减少测试点数量 (5个点足够拟合斜率)
        configs = []

        # 1. 纯 decode 测试：测量纯 decode 的边际成本
        decode_counts = [256, 512, 768, 1024, 1280]
        for num_d in decode_counts:
            configs.append((f"pure_decode_{num_d}", num_d, 0))

        # 2. 混合 batch - 固定 prefill，变化 decode：测量混合时 decode 的边际成本
        fixed_prefill = 256
        for num_d in decode_counts:
            configs.append((f"mixed_p{fixed_prefill}_d{num_d}", num_d, fixed_prefill))

        # 3. 混合 batch - 固定 decode，变化 prefill：测量混合时 prefill 的边际成本
        fixed_decode = 512
        prefill_sizes = [64, 192, 320, 448, 512]
        for p_size in prefill_sizes:
            configs.append((f"mixed_d{fixed_decode}_p{p_size}", fixed_decode, p_size))

        # 4. 纯 prefill 测试：测量纯 prefill 的边际成本
        for p_size in prefill_sizes:
            configs.append((f"pure_prefill_{p_size}", 0, p_size))

        print(f"\nDecode context length: {self.decode_context_len}")
        print("-" * 70)

        for name, num_decode, prefill_size in configs:
            print(f"Testing {name}...", end=" ", flush=True)

            if prefill_size == 0 and num_decode > 0:
                result = self.run_pure_decode(num_decode)
            elif num_decode == 0 and prefill_size > 0:
                result = self.run_pure_prefill(prefill_size)
            else:
                result = self.run_mixed_batch(num_decode, prefill_size)

            if "error" in result:
                print(f"FAILED: {result['error']}")
                continue

            result["name"] = name
            results.append(result)

            print(f"{result['mean_time_ms']:.3f}ms ± {result['std_time_ms']:.3f}ms "
                  f"(max_query_len={result['max_query_len']})")

        # 分析结果：分别计算三种斜率
        print("\n" + "=" * 70)
        print("分析结果: 分别测量 Decode 和 Prefill 的边际成本")
        print("=" * 70)

        # 计算斜率（线性回归）
        def calc_slope(data):
            if len(data) < 2:
                return 0, 0
            x = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            slope, intercept = np.polyfit(x, y, 1)
            return slope, intercept

        # 1. 纯 decode 数据：(num_decode, time)
        pure_decode_data = [(r["num_decode"], r["mean_time_ms"])
                           for r in results if r["type"] == "pure_decode"]
        pure_decode_data.sort()

        # 2. 混合 batch 固定 prefill，变化 decode：(num_decode, time)
        mixed_vary_decode = [(r["num_decode"], r["mean_time_ms"])
                            for r in results
                            if r["type"] == "mixed" and r["prefill_size"] == fixed_prefill]
        mixed_vary_decode.sort()

        # 3. 混合 batch 固定 decode，变化 prefill：(prefill_size, time)
        mixed_vary_prefill = [(r["prefill_size"], r["mean_time_ms"])
                             for r in results
                             if r["type"] == "mixed" and r["num_decode"] == fixed_decode]
        mixed_vary_prefill.sort()

        # 4. 纯 prefill 数据：(prefill_size, time)
        pure_prefill_data = [(r["prefill_size"], r["mean_time_ms"])
                            for r in results if r["type"] == "pure_prefill"]
        pure_prefill_data.sort()

        # 打印数据表 1: 纯 decode vs 混合 decode
        print(f"\n=== 表1: Decode 边际成本比较 (固定 prefill={fixed_prefill}) ===")
        print(f"{'Decode数量':<12} {'纯Decode(ms)':<15} {'混合(ms)':<15} {'差值(ms)':<12}")
        print("-" * 55)

        pure_dict = dict(pure_decode_data)
        mixed_d_dict = dict(mixed_vary_decode)
        for num_d in sorted(set(pure_dict.keys()) & set(mixed_d_dict.keys())):
            diff = mixed_d_dict[num_d] - pure_dict[num_d]
            print(f"{num_d:<12} {pure_dict[num_d]:<15.3f} {mixed_d_dict[num_d]:<15.3f} {diff:<+12.3f}")

        # 打印数据表 2: 纯 prefill vs 混合 prefill
        print(f"\n=== 表2: Prefill 边际成本比较 (混合时固定 decode={fixed_decode}) ===")
        print(f"{'Prefill大小':<12} {'纯Prefill(ms)':<15} {'混合(ms)':<15} {'差值(ms)':<12}")
        print("-" * 55)

        pure_p_dict = dict(pure_prefill_data)
        mixed_p_dict = dict(mixed_vary_prefill)
        for p_size in sorted(set(pure_p_dict.keys()) & set(mixed_p_dict.keys())):
            diff = mixed_p_dict[p_size] - pure_p_dict[p_size]
            print(f"{p_size:<12} {pure_p_dict[p_size]:<15.3f} {mixed_p_dict[p_size]:<15.3f} {diff:<+12.3f}")

        # 计算四种斜率
        pure_decode_slope, _ = calc_slope(pure_decode_data)
        mixed_decode_slope, _ = calc_slope(mixed_vary_decode)
        pure_prefill_slope, _ = calc_slope(pure_prefill_data)
        mixed_prefill_slope, _ = calc_slope(mixed_vary_prefill)

        # 斜率分析
        print("\n" + "=" * 70)
        print("斜率分析 (边际成本)")
        print("=" * 70)
        print(f"  1. 纯 Decode 边际成本:           {pure_decode_slope * 1000:.4f} μs/token")
        print(f"  2. 混合时 Decode 边际成本:       {mixed_decode_slope * 1000:.4f} μs/token")
        print(f"  3. 纯 Prefill 边际成本:          {pure_prefill_slope * 1000:.4f} μs/token")
        print(f"  4. 混合时 Prefill 边际成本:      {mixed_prefill_slope * 1000:.4f} μs/token")

        # 比较
        print("\n比较:")
        if pure_decode_slope > 0:
            decode_ratio = mixed_decode_slope / pure_decode_slope
            print(f"  混合时Decode成本 / 纯Decode成本 = {decode_ratio:.2f}x")

        if pure_prefill_slope > 0:
            prefill_ratio = mixed_prefill_slope / pure_prefill_slope
            print(f"  混合时Prefill成本 / 纯Prefill成本 = {prefill_ratio:.2f}x")

        if pure_decode_slope > 0 and pure_prefill_slope > 0:
            pure_ratio = pure_prefill_slope / pure_decode_slope
            print(f"  纯Prefill成本 / 纯Decode成本 = {pure_ratio:.2f}x")

        # 可视化
        print("\n斜率可视化 (█ = 1 μs/token):")
        bar1 = "█" * max(1, int(pure_decode_slope * 1000))
        bar2 = "█" * max(1, int(mixed_decode_slope * 1000))
        bar3 = "█" * max(1, int(pure_prefill_slope * 1000))
        bar4 = "█" * max(1, int(mixed_prefill_slope * 1000))
        print(f"  纯Decode:    {bar1} ({pure_decode_slope * 1000:.2f} μs)")
        print(f"  混合Decode:  {bar2} ({mixed_decode_slope * 1000:.2f} μs)")
        print(f"  纯Prefill:   {bar3} ({pure_prefill_slope * 1000:.2f} μs)")
        print(f"  混合Prefill: {bar4} ({mixed_prefill_slope * 1000:.2f} μs)")

        # 结论
        print("\n" + "=" * 70)
        print("结论:")

        # Decode 分析
        if mixed_decode_slope > pure_decode_slope * 1.1:
            overhead = (mixed_decode_slope / pure_decode_slope - 1) * 100
            print(f"  ✓ 混合batch中每个decode token成本增加了 {overhead:.1f}%")
            print("    原因: prefill的存在可能打断了decode的批处理优化")
        else:
            print("  ~ 混合batch中decode成本与纯decode接近")

        # Prefill 分析
        if pure_prefill_slope > 0:
            if mixed_prefill_slope > pure_prefill_slope * 1.1:
                overhead = (mixed_prefill_slope / pure_prefill_slope - 1) * 100
                print(f"  ✓ 混合batch中每个prefill token成本增加了 {overhead:.1f}%")
                print("    原因: decode的存在增加了prefill的处理开销")
            elif mixed_prefill_slope < pure_prefill_slope * 0.9:
                saving = (1 - mixed_prefill_slope / pure_prefill_slope) * 100
                print(f"  ✓ 混合batch中每个prefill token成本降低了 {saving:.1f}%")
                print("    原因: decode请求的KV cache可能提高了内存访问效率")
            else:
                print("  ~ 混合batch中prefill成本与纯prefill接近")

        # 综合分析
        if pure_prefill_slope > 0 and pure_decode_slope > 0:
            print("\n  综合分析:")
            ratio = pure_prefill_slope / pure_decode_slope
            print(f"    纯Prefill边际成本 / 纯Decode边际成本 = {ratio:.2f}x")
            if ratio < 1:
                print("    → Prefill是compute-bound, Decode是memory-bound")
            else:
                print("    → Prefill和Decode都可能是memory-bound")

        print("=" * 70)

        return results


def main():
    parser = argparse.ArgumentParser(description="验证Prefill对Decode优化的影响")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="Model to test")
    parser.add_argument("--decode-context-len", type=int, default=512,
                        help="Context length for decode requests")
    args = parser.parse_args()

    verifier = DecodeOptimizationVerifier(
        model=args.model,
        decode_context_len=args.decode_context_len,
    )
    verifier.setup()
    verifier.verify()


if __name__ == "__main__":
    main()
