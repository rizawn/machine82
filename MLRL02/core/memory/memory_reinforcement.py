"""
MEMORY REINFORCEMENT — Adaptive Long-Term Semantic Memory

Features:
    - Track frequently used memories
    - Increase relevance priority over time
    - Simulate reinforcement behavior
    - Store memory usage statistics
    - Improve semantic retrieval quality

Architecture:
    This module wraps around MemoryContext to add adaptive behavior:

        Query → Retrieve → Reinforce Used → Decay Unused
            → Re-rank by reinforced score → Return

    Memories that are frequently accessed become more prominent.
    Memories that go unused gradually decay in priority.
    This simulates how biological memory strengthens with use.

    Reinforcement model:
        reinforced_score = base_score * (1 + usage_weight - decay_factor)

    Where:
        usage_weight  = how often this chunk has been accessed
        decay_factor  = how long since last access

Usage:
    reinforcement = MemoryReinforcement()
    contexts = reinforcement.retrieve("apa itu embeddings?")
    reinforcement.record_usage(contexts[0].chunk_id, quality=0.8)
    print(reinforcement.stats())
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

# Project root sys.path hack removed from module level (M-1)

from core.memory.memory_context import MemoryContext, ContextItem


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REINFORCEMENT_DIR = os.path.join(BASE_DIR, "workspace")
REINFORCEMENT_FILE = os.path.join(REINFORCEMENT_DIR, "memory_stats.json")

# Reinforcement parameters
MAX_USAGE_BOOST = 1.0       # Maximum boost from usage (caps at +100%)
DECAY_RATE = 0.01           # Per-day decay rate
DECAY_HALF_LIFE = 30        # Days until priority halves from disuse
MIN_SCORE = 0.05            # Minimum reinforced score (never fully forgotten)
REINFORCEMENT_WEIGHT = 0.4  # How much reinforcement affects ranking (0–1)


# ──────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """
    Tracking data for a single memory chunk.

    Attributes:
        chunk_id:       Unique identifier in vector store
        source:         Origin file
        access_count:   Total times retrieved
        last_access:    Unix timestamp of last access
        first_access:   Unix timestamp of first access
        avg_quality:    Average quality score of uses
        total_boost:    Current accumulated reinforcement boost
    """
    chunk_id: str
    source: str = "unknown"
    access_count: int = 0
    last_access: float = 0.0
    first_access: float = 0.0
    avg_quality: float = 0.0
    total_boost: float = 0.0

    @property
    def days_since_access(self) -> float:
        """Days since this memory was last used."""
        if self.last_access == 0:
            return float('inf')
        return (time.time() - self.last_access) / 86400

    @property
    def decay_factor(self) -> float:
        """How much this memory has decayed from disuse."""
        if self.last_access == 0:
            return 1.0  # Fully decayed (never used)
        days = self.days_since_access
        # Exponential decay: halve every DECAY_HALF_LIFE days
        return 2 ** (-days / DECAY_HALF_LIFE)

    def record_use(self, quality: float = 0.5):
        """Record a usage of this memory chunk."""
        now = time.time()
        if self.first_access == 0:
            self.first_access = now

        self.access_count += 1
        self.last_access = now

        # Running average of quality
        self.avg_quality = (
            (self.avg_quality * (self.access_count - 1) + quality)
            / self.access_count
        )

        # Calculate new boost
        self._recalculate_boost()

    def _recalculate_boost(self):
        """Recalculate the reinforcement boost."""
        # Usage component: logarithmic growth (diminishing returns)
        import math
        usage_component = math.log1p(self.access_count) / math.log1p(100)
        usage_component = min(MAX_USAGE_BOOST, usage_component)

        # Quality component: high-quality uses boost more
        quality_component = self.avg_quality * 0.3

        # Decay: reduces the boost over time
        decay = self.decay_factor

        self.total_boost = (usage_component + quality_component) * decay
        self.total_boost = min(MAX_USAGE_BOOST, self.total_boost)


@dataclass
class ReinforcementStats:
    """
    Aggregate statistics about memory reinforcement.

    Attributes:
        total_tracked:      Number of memory chunks being tracked
        total_accesses:     Total retrievals across all chunks
        avg_boost:          Average reinforcement boost
        top_memories:       Most accessed chunk IDs
        forgotten_memories: Chunks that haven't been used recently
        active_memories:    Chunks used in last 7 days
    """
    total_tracked: int = 0
    total_accesses: int = 0
    avg_boost: float = 0.0
    top_memories: list[str] = field(default_factory=list)
    forgotten_memories: list[str] = field(default_factory=list)
    active_memories: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
#  MEMORY REINFORCEMENT ENGINE
# ──────────────────────────────────────────────

class MemoryReinforcement:
    """
    Adaptive memory layer that reinforces frequently used memories
    and decays unused ones.

    Wraps MemoryContext and adds:
    - Usage tracking per chunk
    - Score reinforcement based on access patterns
    - Automatic decay for stale memories
    - Persistent stats storage

    Usage:
        engine = MemoryReinforcement()
        contexts = engine.retrieve("apa itu embeddings?")
        engine.record_usage(contexts[0].chunk_id, quality=0.85)
    """

    def __init__(
        self,
        memory: Optional[MemoryContext] = None,
        stats_file: Optional[str] = None,
    ):
        self.memory = memory or MemoryContext()
        self.stats_file = stats_file or REINFORCEMENT_FILE

        # Memory tracking: chunk_id → MemoryRecord
        self._records: dict[str, MemoryRecord] = {}

        # Load persisted stats
        self._load_stats()

    # ──────────────────────────────────────────
    #  RETRIEVE (with reinforcement)
    # ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        use_reinforcement: bool = True,
    ) -> list[ContextItem]:
        """
        Retrieve contexts with reinforced scoring.

        Args:
            query:             The search query
            top_k:             Number of results
            use_reinforcement: Whether to apply reinforcement scoring

        Returns:
            List of ContextItem ranked by reinforced score
        """
        # Base retrieval from vector store
        contexts = self.memory.retrieve(query, top_k=top_k * 2)

        if not contexts:
            return []

        if use_reinforcement:
            # Apply reinforcement to scores
            for ctx in contexts:
                ctx.score = self._reinforced_score(ctx)

            # Re-sort by reinforced score
            contexts.sort(key=lambda c: c.score, reverse=True)

        # Track that these were retrieved (without quality yet)
        for ctx in contexts:
            if ctx.chunk_id not in self._records:
                self._records[ctx.chunk_id] = MemoryRecord(
                    chunk_id=ctx.chunk_id,
                    source=ctx.source,
                )

        return contexts[:top_k]

    def retrieve_raw(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Raw retrieval without reinforcement — for debugging.
        """
        return self.memory.retrieve_raw(query, n_results=n_results)

    # ──────────────────────────────────────────
    #  REINFORCEMENT
    # ──────────────────────────────────────────

    def _reinforced_score(self, context: ContextItem) -> float:
        """
        Calculate reinforced score for a context.

        Formula:
            reinforced = base * (1 - w) + reinforced_component * w

        Where w = REINFORCEMENT_WEIGHT
        """
        base_score = context.score
        record = self._records.get(context.chunk_id)

        if not record or record.access_count == 0:
            return base_score

        # Reinforcement component
        boost = record.total_boost
        recency = record.decay_factor
        reinforced = base_score * (1 + boost * recency)

        # Weighted blend: don't let reinforcement completely override base score
        blended = (
            base_score * (1 - REINFORCEMENT_WEIGHT) +
            reinforced * REINFORCEMENT_WEIGHT
        )

        return max(MIN_SCORE, min(1.0, blended))

    def record_usage(self, chunk_id: str, quality: float = 0.5):
        """
        Record that a memory chunk was used and how good it was.

        Call this after the LLM uses a context to generate an answer.
        Higher quality = the context was more relevant to the answer.

        Args:
            chunk_id: The memory chunk ID
            quality:  How relevant it was (0.0 – 1.0)
        """
        if chunk_id not in self._records:
            self._records[chunk_id] = MemoryRecord(
                chunk_id=chunk_id,
                source="unknown",
            )

        self._records[chunk_id].record_use(quality)
        self._save_stats()

    def record_batch_usage(
        self,
        chunk_ids: list[str],
        qualities: Optional[list[float]] = None,
    ):
        """
        Record usage for multiple chunks at once.

        Args:
            chunk_ids:  List of chunk IDs that were used
            qualities:  Per-chunk quality scores (defaults to 0.5)
        """
        if not qualities:
            qualities = [0.5] * len(chunk_ids)

        for chunk_id, quality in zip(chunk_ids, qualities):
            self.record_usage(chunk_id, quality)

    # ──────────────────────────────────────────
    #  DECAY & MAINTENANCE
    # ──────────────────────────────────────────

    def apply_decay(self):
        """
        Apply decay to all tracked memories.

        Call this periodically (e.g., daily) to simulate
        natural forgetting of unused memories.
        """
        for record in self._records.values():
            if record.access_count > 0:
                record._recalculate_boost()

        self._save_stats()

    def forget_old(
        self,
        days_threshold: int = 90,
        min_access: int = 1,
    ) -> list[str]:
        """
        Remove tracking for memories that haven't been used
        and have low importance.

        Args:
            days_threshold: Forget if unused for this many days
            min_access:     Don't forget if accessed at least this many times

        Returns:
            List of forgotten chunk IDs
        """
        forgotten = []

        for chunk_id, record in list(self._records.items()):
            if (record.access_count < min_access
                    and record.days_since_access > days_threshold):
                forgotten.append(chunk_id)
                del self._records[chunk_id]

        if forgotten:
            self._save_stats()

        return forgotten

    # ──────────────────────────────────────────
    #  STATS & ANALYTICS
    # ──────────────────────────────────────────

    def get_record(self, chunk_id: str) -> Optional[MemoryRecord]:
        """Get the tracking record for a chunk."""
        return self._records.get(chunk_id)

    def get_most_used(self, n: int = 10) -> list[MemoryRecord]:
        """Get the most frequently accessed memory chunks."""
        records = sorted(
            self._records.values(),
            key=lambda r: r.access_count,
            reverse=True,
        )
        return records[:n]

    def get_highest_boosted(self, n: int = 10) -> list[MemoryRecord]:
        """Get memories with the highest reinforcement boost."""
        records = sorted(
            self._records.values(),
            key=lambda r: r.total_boost,
            reverse=True,
        )
        return records[:n]

    def get_active(self, days: int = 7) -> list[MemoryRecord]:
        """Get memories accessed within the last N days."""
        now = time.time()
        cutoff = now - (days * 86400)

        return [
            r for r in self._records.values()
            if r.last_access >= cutoff
        ]

    def get_forgotten(self, days: int = 30) -> list[MemoryRecord]:
        """Get memories not accessed for N+ days."""
        now = time.time()
        cutoff = now - (days * 86400)

        return [
            r for r in self._records.values()
            if r.last_access > 0 and r.last_access < cutoff
        ]

    def stats(self) -> ReinforcementStats:
        """Generate aggregate statistics."""
        if not self._records:
            return ReinforcementStats()

        total_accesses = sum(r.access_count for r in self._records.values())
        avg_boost = (
            sum(r.total_boost for r in self._records.values() if r.access_count > 0)
            / max(1, sum(1 for r in self._records.values() if r.access_count > 0))
        )

        # Top memories by access count
        top = sorted(
            self._records.values(),
            key=lambda r: r.access_count,
            reverse=True,
        )[:5]
        top_ids = [r.chunk_id for r in top]

        # Active memories (last 7 days)
        active = self.get_active(days=7)
        active_ids = [r.chunk_id for r in active]

        # Forgotten memories (30+ days)
        forgotten = self.get_forgotten(days=30)
        forgotten_ids = [r.chunk_id for r in forgotten]

        return ReinforcementStats(
            total_tracked=len(self._records),
            total_accesses=total_accesses,
            avg_boost=round(avg_boost, 3),
            top_memories=top_ids,
            active_memories=active_ids,
            forgotten_memories=forgotten_ids,
        )

    # ──────────────────────────────────────────
    #  PERSISTENCE
    # ──────────────────────────────────────────

    def _save_stats(self):
        """Persist tracking data to disk."""
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)

        data = {
            "records": {
                chunk_id: {
                    "source": r.source,
                    "access_count": r.access_count,
                    "last_access": r.last_access,
                    "first_access": r.first_access,
                    "avg_quality": r.avg_quality,
                    "total_boost": r.total_boost,
                }
                for chunk_id, r in self._records.items()
            },
            "timestamp": time.time(),
        }

        try:
            with open(self.stats_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[MemoryReinforcement] ⚠️  Failed to save stats: {e}")

    def _load_stats(self):
        """Load tracking data from disk."""
        if not os.path.exists(self.stats_file):
            return

        try:
            with open(self.stats_file, "r") as f:
                data = json.load(f)

            for chunk_id, record_data in data.get("records", {}).items():
                self._records[chunk_id] = MemoryRecord(
                    chunk_id=chunk_id,
                    source=record_data.get("source", "unknown"),
                    access_count=record_data.get("access_count", 0),
                    last_access=record_data.get("last_access", 0),
                    first_access=record_data.get("first_access", 0),
                    avg_quality=record_data.get("avg_quality", 0),
                    total_boost=record_data.get("total_boost", 0),
                )

            if self._records:
                print(f"[MemoryReinforcement] Loaded {len(self._records)} tracked memories.")
        except Exception as e:
            print(f"[MemoryReinforcement] ⚠️  Failed to load stats: {e}")

    # ──────────────────────────────────────────
    #  DEBUG
    # ──────────────────────────────────────────

    def print_stats(self):
        """Pretty-print reinforcement statistics."""
        s = self.stats()

        print("\n" + "=" * 60)
        print("  🧠 MEMORY REINFORCEMENT STATS")
        print("=" * 60)

        print(f"\n📊 Overview:")
        print(f"   Tracked memories: {s.total_tracked}")
        print(f"   Total accesses:   {s.total_accesses}")
        print(f"   Avg boost:        {s.avg_boost:.0%}")

        if s.top_memories:
            print(f"\n🔥 Most Used Memories:")
            for i, cid in enumerate(s.top_memories, 1):
                record = self._records.get(cid)
                if record:
                    print(f"   [{i}] {cid[:40]} "
                          f"({record.access_count}x, "
                          f"boost: {record.total_boost:.0%}, "
                          f"quality: {record.avg_quality:.0%})")

        if s.active_memories:
            print(f"\n⚡ Active (last 7 days): {len(s.active_memories)} memories")

        if s.forgotten_memories:
            print(f"\n💤 Forgotten (30+ days): {len(s.forgotten_memories)} memories")

        print()


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure project root is on path for standalone execution
    import os
    import sys
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    print("=" * 60)
    print("  MEMORY REINFORCEMENT — Quick Test")
    print("=" * 60 + "\n")

    engine = MemoryReinforcement()
    print(f"Memory documents: {engine.memory.count()}\n")

    # ── Simulate usage ──
    queries = [
        "apa itu embeddings?",
        "bagaimana semantic search?",
        "apa itu embeddings?",  # Repeated query — should reinforce
        "apa itu vector database?",
    ]

    for i, q in enumerate(queries):
        print(f"── Query {i + 1}: {q} ──")
        contexts = engine.retrieve(q, top_k=3)

        for ctx in contexts:
            # Simulate quality assessment
            quality = ctx.score * 0.8 + 0.1  # Roughly 0.1–0.9
            engine.record_usage(ctx.chunk_id, quality=quality)

        if contexts:
            print(f"   Top result: {contexts[0].source} "
                  f"(reinforced score: {contexts[0].score:.0%})")
        print()

    # ── Show reinforced vs base ──
    print("=" * 60)
    print("  REINFORCED vs BASE SCORES")
    print("=" * 60 + "\n")

    test_query = "apa itu embeddings?"

    # Without reinforcement
    raw = engine.retrieve(test_query, top_k=3, use_reinforcement=False)
    print("  Without reinforcement:")
    for ctx in raw:
        record = engine.get_record(ctx.chunk_id)
        boost_str = f"boost: {record.total_boost:.0%}" if record and record.access_count > 0 else "no usage"
        print(f"    {ctx.source[:30]:30s} score: {ctx.score:.0%} | {boost_str}")

    # With reinforcement
    reinforced = engine.retrieve(test_query, top_k=3, use_reinforcement=True)
    print("\n  With reinforcement:")
    for ctx in reinforced:
        record = engine.get_record(ctx.chunk_id)
        boost_str = f"boost: {record.total_boost:.0%}" if record and record.access_count > 0 else "no usage"
        print(f"    {ctx.source[:30]:30s} score: {ctx.score:.0%} | {boost_str}")

    # ── Final stats ──
    engine.print_stats()
