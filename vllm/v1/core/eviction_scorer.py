"""Pluggable eviction scorer for aginfer value-aware KV scheduling.

Updates ``KVCacheBlock.eviction_score`` based on reuse history.
Higher score = more valuable = evicted later.

The default scorer uses a hit-count + recency proxy:
  score = hit_count * REUSE_WEIGHT + recency * RECENCY_WEIGHT

This is the same signal as sglang's ``ours_greedy_score`` (reuse-based
p_hat = 1 - exp(-alpha * (hits - 1))), simplified to a linear proxy
for the block-level granularity (vllm blocks are fixed-size, not
variable-length radix nodes).

Integration point: ``BlockPool.touch()`` calls ``update_scores()``
after incrementing ref_cnt. ``BlockPool.free_blocks()`` sorts by
eviction_score before inserting into the free queue.

When the scorer is None (default), eviction_score stays 0.0 and
free_blocks preserves LRU order (do-no-harm).
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


class EvictionScorer(Protocol):
    def update_scores(self, blocks: Sequence["KVCacheBlock"]) -> None:
        """Update eviction_score on the given blocks (just touched)."""
        ...


class HitCountScorer:
    """Score = cumulative hit count. Simple, effective, no tuning.

    Every touch (= a new request matched this block's prefix)
    increments the block's score by 1. Blocks that are shared by
    many requests accumulate high scores and survive eviction.
    """

    def __init__(self):
        self._boot_time = time.monotonic()

    def update_scores(self, blocks: Sequence["KVCacheBlock"]) -> None:
        for block in blocks:
            block.eviction_score += 1.0
