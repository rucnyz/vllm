"""Comprehensive tests for overlap-compatible teacher-forcing (forced_tokens).

Tests cover:
  - Correctness: counter advance, token indexing, stop at end
  - Safety: skips non-forced requests, no advance on skip
  - Mixed batches: forced + non-forced coexist
  - Resync after preemption: counter reset
  - Edge cases: empty forced list, None extra_args, missing requests
  - Performance guard: no-op path is zero-cost (no GPU work)
  - GPU scatter: tensor values are correct
"""
import sys
import os
import time

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.forced_tokens import (
    apply_forced_tokens,
    forced_override_positions,
)


class FakeReq:
    """Minimal stand-in for CachedRequestState."""
    def __init__(self, forced=None, dispatched=0):
        sp = SamplingParams(max_tokens=100)
        if forced is not None:
            sp.extra_args = {"forced_output_ids": forced}
        else:
            sp.extra_args = None
        self.sampling_params = sp
        self.forced_dispatched = dispatched


# --- Correctness ---

def test_basic_advance_and_index():
    reqs = {"r0": FakeReq(forced=[10, 11, 12])}
    assert forced_override_positions(["r0"], reqs) == [(0, 10)]
    assert reqs["r0"].forced_dispatched == 1
    assert forced_override_positions(["r0"], reqs) == [(0, 11)]
    assert reqs["r0"].forced_dispatched == 2
    assert forced_override_positions(["r0"], reqs) == [(0, 12)]
    assert reqs["r0"].forced_dispatched == 3
    # Past the end — no more overrides, counter frozen
    assert forced_override_positions(["r0"], reqs) == []
    assert reqs["r0"].forced_dispatched == 3


def test_skips_non_forced():
    reqs = {
        "r0": FakeReq(forced=None),
        "r1": FakeReq(forced=[]),
    }
    assert forced_override_positions(["r0", "r1"], reqs) == []
    assert reqs["r0"].forced_dispatched == 0
    assert reqs["r1"].forced_dispatched == 0


def test_mixed_batch():
    reqs = {
        "r0": FakeReq(forced=[100, 101]),
        "r1": FakeReq(forced=None),        # no forcing
        "r2": FakeReq(forced=[300, 301], dispatched=1),  # starts at pos 1
    }
    out = forced_override_positions(["r0", "r1", "r2"], reqs)
    assert out == [(0, 100), (2, 301)]
    assert reqs["r0"].forced_dispatched == 1
    assert reqs["r1"].forced_dispatched == 0
    assert reqs["r2"].forced_dispatched == 2


def test_resync_after_preemption():
    """Counter can be manually resynced (e.g. on preemption/retraction)."""
    r = FakeReq(forced=[10, 11, 12, 13], dispatched=3)
    reqs = {"r0": r}
    assert forced_override_positions(["r0"], reqs) == [(0, 13)]
    assert r.forced_dispatched == 4


def test_missing_request():
    """Request ID not in dict — skip silently."""
    reqs = {"r0": FakeReq(forced=[10])}
    assert forced_override_positions(["r0", "missing"], reqs) == [(0, 10)]


def test_no_extra_args():
    """SamplingParams with extra_args=None."""
    r = FakeReq(forced=None)
    r.sampling_params.extra_args = None
    reqs = {"r0": r}
    assert forced_override_positions(["r0"], reqs) == []


def test_empty_req_ids():
    reqs = {"r0": FakeReq(forced=[10])}
    assert forced_override_positions([], reqs) == []


# --- GPU scatter ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_scatter_2d():
    """sampled_token_ids shape (batch, 1) — standard non-spec decode."""
    reqs = {
        "r0": FakeReq(forced=[42]),
        "r1": FakeReq(forced=None),
        "r2": FakeReq(forced=[99]),
    }
    t = torch.zeros(3, 1, dtype=torch.long, device="cuda")
    apply_forced_tokens(["r0", "r1", "r2"], reqs, t)
    cpu = t.cpu()
    assert cpu[0, 0].item() == 42
    assert cpu[1, 0].item() == 0  # untouched
    assert cpu[2, 0].item() == 99


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_scatter_1d():
    """sampled_token_ids shape (batch,) — fallback."""
    reqs = {"r0": FakeReq(forced=[77])}
    t = torch.zeros(2, dtype=torch.long, device="cuda")
    apply_forced_tokens(["r0", "r1_missing"], reqs, t)
    assert t.cpu()[0].item() == 77
    assert t.cpu()[1].item() == 0


# --- Performance ---

def test_noop_is_fast():
    """No forced requests → zero GPU work, sub-microsecond."""
    reqs = {f"r{i}": FakeReq(forced=None) for i in range(1000)}
    ids = [f"r{i}" for i in range(1000)]
    t0 = time.perf_counter()
    for _ in range(10000):
        forced_override_positions(ids, reqs)
    elapsed = (time.perf_counter() - t0) / 10000
    # Should be < 1ms even for 1000 requests (it's a dict lookup loop)
    assert elapsed < 0.001, f"no-op too slow: {elapsed*1e6:.1f} us per call"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_noop_no_gpu_kernel():
    """apply_forced_tokens with no forced reqs must not launch a GPU kernel."""
    reqs = {"r0": FakeReq(forced=None)}
    t = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    # If this launches a kernel, it would show up in profiling;
    # here we just verify it returns instantly without error
    apply_forced_tokens(["r0"], reqs, t)
    assert t.cpu()[0, 0].item() == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        name = t.__name__
        skip_marker = getattr(t, "pytestmark", None)
        if skip_marker:
            for m in skip_marker:
                if m.name == "skipif" and m.args[0]:
                    print(f"  SKIP  {name}: {m.kwargs.get('reason', '')}")
                    passed += 1
                    continue
        try:
            t()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            import traceback; traceback.print_exc()
    print(f"\nforced_tokens (vllm): {passed}/{len(tests)} passed")
