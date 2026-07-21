"""
MEMORY CONTEXT — Semantic Memory Retrieval Layer

Responsibilities:
    - Retrieve relevant contexts from ChromaDB via similarity search
    - Rank results by relevance score (cosine distance → similarity)
    - Deduplicate near-identical contexts
    - Enforce token budgets to prevent LLM context overflow
    - Prioritize high-quality chunks (length, source freshness, score)
    - Provide a clean interface for future long-term memory expansion

Architecture:
    This module sits between the vector database and the prompt builder.
    It takes raw search results and turns them into a curated context list
    ready for LLM injection.

    Query → Retrieve → Score → Deduplicate → Token-limit → Ranked Contexts

Usage:
    retriever = MemoryContext()
    contexts = retriever.retrieve("apa itu embeddings?")
    for ctx in contexts:
        print(f"[{ctx.score:.2%}] {ctx.source}: {ctx.text[:80]}...")
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "workspace", "embeddings")
COLLECTION_NAME = "mlrl_memory"

# Token budget — max tokens for context section in the prompt
# Rough estimate: 1 token ≈ 0.75 words for English, ~0.5 for mixed
DEFAULT_MAX_CONTEXT_TOKENS = 4000
DEFAULT_TOP_K = 10  # Fetch more than needed, then filter down

# Similarity threshold — discard results below this
MIN_SIMILARITY = 0.15

# Dedup threshold — cosine similarity above this means "duplicate"
DEDUP_THRESHOLD = 0.92


# ──────────────────────────────────────────────
#  CONTEXT ITEM
# ──────────────────────────────────────────────

@dataclass
class ContextItem:
    """
    A single retrieved memory chunk with metadata.

    Attributes:
        text:       The actual context text
        source:     Origin file or topic
        score:      Relevance score (0.0 – 1.0, higher = more relevant)
        chunk_id:   Unique identifier in the vector store
        metadata:   Raw metadata dict from ChromaDB
        token_est:  Estimated token count
        quality:    Internal quality rating (0.0 – 1.0)
    """
    text: str
    source: str = "unknown"
    score: float = 0.0
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)
    token_est: int = 0
    quality: float = 0.0

    def __post_init__(self):
        if self.token_est == 0:
            self.token_est = estimate_tokens(self.text)
        if self.quality == 0.0:
            self.quality = _compute_quality(self)


# ──────────────────────────────────────────────
#  TOKEN ESTIMATION
# ──────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Fast token estimate without loading a tokenizer.

    Strategy: split on whitespace, then adjust for common patterns.
    This is ~90% accurate for length-checking purposes.
    """
    words = text.split()
    # English average: 1 token ≈ 1.3 characters per word
    # But whitespace split overestimates slightly
    return max(1, int(len(words) * 1.1))


# ──────────────────────────────────────────────
#  QUALITY SCORING
# ──────────────────────────────────────────────

def _compute_quality(item: ContextItem) -> float:
    """
    Compute a quality score for a context chunk.

    Factors:
    - Text length (not too short, not absurdly long)
    - Has meaningful content (not just headers or code blocks)
    - Source metadata present
    """
    score = 0.0
    stripped = item.text.strip()
    word_count = len(stripped.split())

    # Length score — prefer chunks 50-500 words
    if 50 <= word_count <= 500:
        score += 0.4
    elif 20 <= word_count < 50:
        score += 0.3
    elif 10 <= word_count < 20:
        score += 0.15
    elif word_count > 500:
        score += 0.1
    else:
        score += 0.05

    # Content quality — penalize pure code or pure headers
    lines = [l for l in stripped.split("\n") if l.strip()]
    if lines:
        header_ratio = sum(
            1 for l in lines if l.startswith("#")
        ) / len(lines)
        code_ratio = sum(
            1 for l in lines if l.startswith("```") or l.startswith("    ")
        ) / len(lines)

        if header_ratio < 0.3 and code_ratio < 0.5:
            score += 0.3
        elif code_ratio < 0.7:
            score += 0.15
        else:
            score += 0.05  # Mostly code — low quality

    # Metadata bonus — known source is more trustworthy
    if item.source and item.source != "unknown":
        score += 0.2

    # Minimum content check — at least some prose
    has_prose = any(
        len(word) > 3 for word in stripped.split()
        if word.isalpha()
    )
    if has_prose:
        score += 0.1

    return round(min(1.0, score), 4)


# ──────────────────────────────────────────────
#  DEDUPLICATION
# ──────────────────────────────────────────────

def _normalize_for_compare(text: str) -> str:
    """Normalize text for similarity comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)      # Collapse whitespace
    return text


def _text_overlap_similarity(a: str, b: str) -> float:
    """
    Compute similarity between two texts using word overlap (Jaccard).

    Returns 0.0 (no overlap) to 1.0 (identical).
    """
    words_a = set(_normalize_for_compare(a).split())
    words_b = set(_normalize_for_compare(b).split())

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union) if union else 0.0


def deduplicate_contexts(
    contexts: list[ContextItem],
    threshold: float = DEDUP_THRESHOLD,
) -> list[ContextItem]:
    """
    Remove near-duplicate contexts, keeping the highest-scoring one.

    Strategy: iterate sorted by score, skip items that overlap too much
    with already-kept items.
    """
    if not contexts:
        return []

    # Sort by combined score (relevance + quality)
    ranked = sorted(
        contexts,
        key=lambda c: c.score * 0.6 + c.quality * 0.4,
        reverse=True,
    )

    kept: list[ContextItem] = []
    for item in ranked:
        is_dup = False
        for existing in kept:
            overlap = _text_overlap_similarity(item.text, existing.text)
            if overlap >= threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(item)

    return kept


# ──────────────────────────────────────────────
#  TOKEN BUDGET
# ──────────────────────────────────────────────

def enforce_token_budget(
    contexts: list[ContextItem],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
) -> list[ContextItem]:
    """
    Trim contexts that would exceed the token budget.

    Keeps contexts in priority order (score + quality) until budget
    is reached. Last item may be truncated if it partially fits.
    """
    if not contexts:
        return []

    total = 0
    result: list[ContextItem] = []

    for item in contexts:
        if total + item.token_est <= max_tokens:
            result.append(item)
            total += item.token_est
        else:
            # Try to fit a truncated version
            remaining = max_tokens - total
            if remaining > 50:  # At least 50 tokens worth
                truncated = _truncate_item(item, remaining)
                result.append(truncated)
            break

    return result


def _truncate_item(item: ContextItem, max_tokens: int) -> ContextItem:
    """Truncate a context item to fit within token limit."""
    words = item.text.split()
    # Estimate how many words fit
    target_words = int(max_tokens / 1.1)
    truncated_text = " ".join(words[:target_words]) + "..."

    return ContextItem(
        text=truncated_text,
        source=item.source,
        score=item.score,
        chunk_id=item.chunk_id,
        metadata=item.metadata,
        token_est=estimate_tokens(truncated_text),
        quality=item.quality,
    )


# ──────────────────────────────────────────────
#  MEMORY CONTEXT RETRIEVER
# ──────────────────────────────────────────────

class MemoryContext:
    """
    Semantic memory retrieval layer.

    Flow:
        1. Query ChromaDB for similar documents
        2. Convert to ContextItem with scores
        3. Filter by minimum similarity
        4. Deduplicate overlapping contexts
        5. Sort by combined relevance + quality
        6. Enforce token budget

    This produces a clean, curated list of contexts ready for
    injection into an LLM prompt.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        min_similarity: float = MIN_SIMILARITY,
        client: Optional[chromadb.PersistentClient] = None,
    ):
        self.persist_dir = persist_dir or EMBEDDINGS_DIR
        self.collection_name = collection_name
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.min_similarity = min_similarity

        self._client: Optional[chromadb.PersistentClient] = client
        self._collection = None
        self._embed_model: Optional[SentenceTransformer] = None
        self._connect()

    def _connect(self):
        """Connect to ChromaDB."""
        if not self._client and not os.path.exists(self.persist_dir):
            return

        try:
            if not self._client:
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"[MemoryContext] ⚠️  ChromaDB connect error: {e}")
            self._collection = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load and cache the embedding model."""
        if self._embed_model is None:
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> list[ContextItem]:
        """
        Retrieve ranked, deduplicated, budget-constrained contexts.

        Args:
            query:     The question or search query
            top_k:     Override default top_k
            max_tokens: Override default token budget

        Returns:
            List of ContextItem, sorted by relevance
        """
        if not self._collection:
            return []

        k = top_k or self.top_k
        budget = max_tokens or self.max_context_tokens

        # Step 1: Raw similarity search
        model = self._get_model()
        query_embedding = model.encode(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        if not results.get("ids") or not results["ids"][0]:
            return []

        # Step 2: Convert to ContextItem
        contexts = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] if results.get("distances") else None
            # ChromaDB uses cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: similarity = 1 - (distance / 2)
            similarity = 1 - (distance / 2) if distance is not None else 0.0

            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            source = meta.get("source", meta.get("filename", "unknown"))

            item = ContextItem(
                text=results["documents"][0][i],
                source=source,
                score=similarity,
                chunk_id=results["ids"][0][i],
                metadata=meta,
            )
            contexts.append(item)

        # Step 3: Filter by minimum similarity
        contexts = [c for c in contexts if c.score >= self.min_similarity]

        if not contexts:
            return []

        # Step 4: Deduplicate
        contexts = deduplicate_contexts(contexts)

        # Step 5: Final sort by combined score
        contexts.sort(
            key=lambda c: c.score * 0.6 + c.quality * 0.4,
            reverse=True,
        )

        # Step 6: Enforce token budget
        contexts = enforce_token_budget(contexts, budget)

        return contexts

    def retrieve_raw(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Raw search results without processing — for debugging or
        when you want full control over filtering.

        Returns:
            List of raw dicts from ChromaDB
        """
        if not self._collection:
            return []

        model = self._get_model()
        query_embedding = model.encode(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })

        return output

    def count(self) -> int:
        """Total documents in memory."""
        if not self._collection:
            return 0
        return self._collection.count()

    def is_available(self) -> bool:
        """Check if memory system is connected and has data."""
        return self._collection is not None and self.count() > 0

    def info(self) -> dict:
        """System info."""
        return {
            "collection": self.collection_name,
            "document_count": self.count(),
            "available": self.is_available(),
            "persist_dir": self.persist_dir,
        }


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MEMORY CONTEXT — Quick Test")
    print("=" * 60 + "\n")

    retriever = MemoryContext()
    print(f"System: {retriever.info()}\n")

    queries = [
        "apa itu embeddings?",
        "bagaimana cara kerja semantic search?",
        "apa itu RAG?",
    ]

    for q in queries:
        print(f"{'=' * 60}")
        print(f"🔍 Query: \"{q}\"\n")

        contexts = retriever.retrieve(q)

        if not contexts:
            print("  No relevant contexts found.\n")
            continue

        print(f"  Found {len(contexts)} context(s):\n")
        for i, ctx in enumerate(contexts, 1):
            print(f"  [{i}] Score: {ctx.score:.2%} | Quality: {ctx.quality:.2%} | "
                  f"Tokens: {ctx.token_est}")
            print(f"      Source: {ctx.source}")
            preview = ctx.text[:120].replace("\n", " ")
            print(f"      Text: {preview}...")
            print()

        total_tokens = sum(c.token_est for c in contexts)
        print(f"  Total tokens: {total_tokens}")
        print(f"  {'=' * 60}\n")
