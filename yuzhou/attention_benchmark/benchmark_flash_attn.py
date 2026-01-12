"""
FlashAttention Microbenchmark: Pure Prefill vs Pure Decode vs Mixed

测试 FlashAttention kernel 在不同 batch 组成下的性能差异，
用于分析为什么 A6000 上 PD scheduler 效果比 H200 更显著。

Usage:
    python yuzhou/attention_benchmark/benchmark_flash_attn.py \
        --batch-size 512 \
        --prefill-len 512 \
        --context-len 512 \
        --output results.json
"""
import torch
import argparse
from typing import List, Tuple
import json
from datetime import datetime


def create_attention_inputs(
    batch_size: int,
    query_lens: List[int],
    context_lens: List[int],
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, ...]:
    """创建 FlashAttention 的输入张量"""

    total_tokens = sum(query_lens)
    max_context = max(context_lens)
    max_blocks = (max_context + block_size - 1) // block_size

    # Query tensor: [total_tokens, num_heads, head_size]
    query = torch.randn(total_tokens, num_heads, head_size,
                       device=device, dtype=dtype)

    # KV cache: [2, num_blocks, block_size, num_kv_heads, head_size]
    num_blocks = batch_size * max_blocks
    kv_cache = torch.randn(2, num_blocks, block_size, num_kv_heads, head_size,
                          device=device, dtype=dtype)

    # cu_seqlens_q: cumulative query lengths
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(torch.tensor(query_lens, device=device), dim=0)

    # seqused_k: context length per request
    seqused_k = torch.tensor(context_lens, dtype=torch.int32, device=device)

    # block_table: [batch_size, max_blocks]
    block_table = torch.arange(batch_size * max_blocks, device=device, dtype=torch.int32)
    block_table = block_table.view(batch_size, max_blocks)

    # Output tensor
    output = torch.empty(total_tokens, num_heads, head_size,
                        device=device, dtype=dtype)

    return (query, kv_cache, cu_seqlens_q, seqused_k, block_table, output,
            max(query_lens), max_context)


def benchmark_scenario(
    scenario_name: str,
    query_lens: List[int],
    context_lens: List[int],
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    block_size: int = 16,
    warmup: int = 10,
    repeat: int = 100,
) -> dict:
    """Benchmark a single scenario"""

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    batch_size = len(query_lens)
    query, kv_cache, cu_seqlens_q, seqused_k, block_table, output, max_q, max_k = \
        create_attention_inputs(
            batch_size, query_lens, context_lens,
            num_heads, num_kv_heads, head_size, block_size
        )

    key_cache = kv_cache[0]
    value_cache = kv_cache[1]

    # Warmup
    for _ in range(warmup):
        flash_attn_varlen_func(
            q=query, k=key_cache, v=value_cache, out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_k,
            softmax_scale=1.0 / (head_size ** 0.5),
            causal=True,
            block_table=block_table,
        )

    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        start_event.record()
        flash_attn_varlen_func(
            q=query, k=key_cache, v=value_cache, out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_k,
            softmax_scale=1.0 / (head_size ** 0.5),
            causal=True,
            block_table=block_table,
        )
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    total_tokens = sum(query_lens)
    avg_time = sum(times) / len(times)

    return {
        "scenario": scenario_name,
        "batch_size": batch_size,
        "total_query_tokens": total_tokens,
        "total_context_tokens": sum(context_lens),
        "max_query_len": max_q,
        "max_context_len": max_k,
        "avg_time_ms": avg_time,
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "tokens_per_sec": total_tokens / (avg_time / 1000),
        "query_lens_sample": query_lens[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="FlashAttention Microbenchmark")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size (number of sequences)")
    parser.add_argument("--prefill-len", type=int, default=512,
                       help="Prefill sequence length")
    parser.add_argument("--context-len", type=int, default=512,
                       help="Context length (KV cache size per request)")
    parser.add_argument("--num-heads", type=int, default=32,
                       help="Number of attention heads (Qwen3-8B: 32)")
    parser.add_argument("--num-kv-heads", type=int, default=8,
                       help="Number of KV heads (Qwen3-8B: 8)")
    parser.add_argument("--head-size", type=int, default=128,
                       help="Head dimension (Qwen3-8B: 128)")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=100,
                       help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: batch_size={args.batch_size}, prefill_len={args.prefill_len}, "
          f"context_len={args.context_len}")
    print(f"Model: num_heads={args.num_heads}, num_kv_heads={args.num_kv_heads}, "
          f"head_size={args.head_size}")

    results = []

    # Scenario 1: Pure Prefill
    num_prefill_seqs = args.batch_size // args.prefill_len
    if num_prefill_seqs < 1:
        num_prefill_seqs = 1
    print(f"\n=== Scenario 1: Pure Prefill ({num_prefill_seqs} seqs x {args.prefill_len} tokens) ===")
    result = benchmark_scenario(
        "pure_prefill",
        query_lens=[args.prefill_len] * num_prefill_seqs,
        context_lens=[args.context_len] * num_prefill_seqs,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results.append(result)
    print(f"  Time: {result['avg_time_ms']:.3f} ms, Tokens/s: {result['tokens_per_sec']:.0f}")

    # Scenario 2: Pure Decode
    print(f"\n=== Scenario 2: Pure Decode ({args.batch_size} seqs x 1 token) ===")
    result = benchmark_scenario(
        "pure_decode",
        query_lens=[1] * args.batch_size,
        context_lens=[args.context_len] * args.batch_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results.append(result)
    print(f"  Time: {result['avg_time_ms']:.3f} ms, Tokens/s: {result['tokens_per_sec']:.0f}")

    # Scenario 3: Mixed (1 prefill + rest decode)
    num_decode = args.batch_size - 1
    print(f"\n=== Scenario 3: Mixed (1 prefill x {args.prefill_len} + {num_decode} decode x 1) ===")
    result = benchmark_scenario(
        "mixed_1prefill",
        query_lens=[args.prefill_len] + [1] * num_decode,
        context_lens=[args.context_len] * args.batch_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results.append(result)
    print(f"  Time: {result['avg_time_ms']:.3f} ms, Tokens/s: {result['tokens_per_sec']:.0f}")

    # Scenario 4: Mixed (10% prefill + 90% decode)
    num_prefill = max(1, args.batch_size // 10)
    num_decode = args.batch_size - num_prefill
    print(f"\n=== Scenario 4: Mixed ({num_prefill} prefill + {num_decode} decode) ===")
    result = benchmark_scenario(
        "mixed_10pct_prefill",
        query_lens=[args.prefill_len] * num_prefill + [1] * num_decode,
        context_lens=[args.context_len] * args.batch_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results.append(result)
    print(f"  Time: {result['avg_time_ms']:.3f} ms, Tokens/s: {result['tokens_per_sec']:.0f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Scenario':<25} {'Time(ms)':<12} {'Tokens':<10} {'Tok/s':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['scenario']:<25} {r['avg_time_ms']:<12.3f} {r['total_query_tokens']:<10} {r['tokens_per_sec']:<15.0f}")

    # Analysis: Compare mixed vs pure_decode
    pure_decode = next(r for r in results if r['scenario'] == 'pure_decode')
    mixed_1 = next(r for r in results if r['scenario'] == 'mixed_1prefill')
    mixed_10 = next(r for r in results if r['scenario'] == 'mixed_10pct_prefill')

    print("\n" + "="*70)
    print("ANALYSIS: Mixed vs Pure Decode")
    print("="*70)
    print(f"Pure Decode time:        {pure_decode['avg_time_ms']:.3f} ms")
    print(f"Mixed (1 prefill) time:  {mixed_1['avg_time_ms']:.3f} ms "
          f"({mixed_1['avg_time_ms']/pure_decode['avg_time_ms']:.2f}x slower)")
    print(f"Mixed (10% prefill) time: {mixed_10['avg_time_ms']:.3f} ms "
          f"({mixed_10['avg_time_ms']/pure_decode['avg_time_ms']:.2f}x slower)")

    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "gpu": torch.cuda.get_device_name(),
            "results": results,
            "analysis": {
                "pure_decode_time_ms": pure_decode['avg_time_ms'],
                "mixed_1prefill_slowdown": mixed_1['avg_time_ms'] / pure_decode['avg_time_ms'],
                "mixed_10pct_slowdown": mixed_10['avg_time_ms'] / pure_decode['avg_time_ms'],
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
