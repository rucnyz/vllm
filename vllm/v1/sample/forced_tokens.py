"""Overlap-compatible teacher-forcing token override for agentreplay.

Overrides sampled tokens with forced values on the GPU, enabling
token-exact replay of captured agent traces. The scatter runs AFTER
sampling but BEFORE the GPU relay (prev_sampled_token_ids), so the
next step's input sees the forced token. Overlap/async-scheduling safe.

The forced sequence is passed via ``SamplingParams.extra_args["forced_output_ids"]``
and reaches the worker through ``CachedRequestState.sampling_params``.
A per-request dispatch counter is maintained on ``CachedRequestState``
(worker-side, not scheduler-side) to track which forced token to emit
next — analogous to sglang's ``Req.forced_dispatched``.

A complete no-op (zero GPU work) when no request carries forced_output_ids.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


def _get_forced(req: "CachedRequestState") -> list | None:
    sp = req.sampling_params
    if sp is None:
        return None
    ea = sp.extra_args
    if ea is None:
        return None
    return ea.get("forced_output_ids")


def forced_override_positions(
    req_ids: Sequence[str],
    requests: dict[str, "CachedRequestState"],
) -> List[Tuple[int, int]]:
    """Return (batch_index, forced_token) for each request that should
    have its sampled token overridden this step.

    Pure bookkeeping — no tensor ops. The ``forced_dispatched`` counter
    advances at dispatch time (not commit time), so the 1-step lag
    under async scheduling is handled correctly.
    """
    out: List[Tuple[int, int]] = []
    for i, rid in enumerate(req_ids):
        req = requests.get(rid)
        if req is None:
            continue
        forced = _get_forced(req)
        if not forced:
            continue
        pos = getattr(req, "forced_dispatched", 0)
        if pos < len(forced):
            out.append((i, int(forced[pos])))
            req.forced_dispatched = pos + 1
    return out


def apply_forced_tokens(
    req_ids: Sequence[str],
    requests: dict[str, "CachedRequestState"],
    sampled_token_ids: torch.Tensor,
) -> None:
    """Scatter forced tokens onto the GPU sampled_token_ids tensor.

    Called between sampling and the GPU relay (prev_sampled_token_ids)
    so the override propagates to the next step's input. A complete
    no-op (zero GPU work) when no request carries forced_output_ids.

    Args:
        req_ids: batch-ordered request IDs.
        requests: the model_runner's CachedRequestState dict.
        sampled_token_ids: GPU tensor, shape (batch, num_samples).
    """
    overrides = forced_override_positions(req_ids, requests)
    if not overrides:
        return
    idx = torch.tensor(
        [o[0] for o in overrides], device=sampled_token_ids.device, dtype=torch.long
    )
    val = torch.tensor(
        [o[1] for o in overrides],
        device=sampled_token_ids.device,
        dtype=sampled_token_ids.dtype,
    )
    if sampled_token_ids.dim() == 2:
        sampled_token_ids[idx, 0] = val
    else:
        sampled_token_ids[idx] = val
