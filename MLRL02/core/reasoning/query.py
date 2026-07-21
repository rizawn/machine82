"""
QUERY / RETRIEVAL SYSTEM — Semantic Search Engine

Flow:
    Question → Embed → Search Vector DB → Return Relevant Context

Module ini menghubungkan user question ke vector memory,
sehingga AI bisa jawab berdasarkan knowledge yang sudah disimpan.
"""

import os
import sys
from typing import Optional

# Project root sys.path hack removed from module level (M-1)

from core.memory.vector_store import VectorStore


class RetrievalEngine:
    """
    Retrieval Engine — cari knowledge berdasarkan MAKNA.

    Bukan keyword matching biasa.
    Ini semantic search — "bagaimana AI menyimpan knowledge?"
    bisa nemu "Vector databases store embeddings."

    Usage:
        engine = RetrievalEngine()
        results = engine.query("apa itu embeddings?")
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Args:
            vector_store: Instance VectorStore yang sudah ada.
                          Kalau None, bikin baru dengan default config.
        """
        self.store = vector_store or VectorStore()
        print(f"[RetrievalEngine] Ready — {self.store.count()} documents in memory.")

    # ──────────────────────────────────────────
    #  QUERY
    # ──────────────────────────────────────────

    def query(self, question: str, n_results: int = 5) -> list[dict]:
        """
        Cari jawaban dari vector memory.

        Args:
            question:  Pertanyaan user
            n_results: Jumlah dokumen yang dikembalikan

        Returns:
            List of relevant documents dengan score
        """
        print(f"\n🔍 [Query] \"{question}\"")
        results = self.store.similarity_search(question, n_results=n_results)

        if not results:
            print("  ❌ No relevant documents found.")
            return []

        print(f"  ✅ Found {len(results)} relevant document(s):\n")
        for i, r in enumerate(results, 1):
            distance = r.get("distance", 0)
            # Cosine distance → similarity (ChromaDB cosine distance is [0, 2])
            similarity = 1 - (distance / 2) if distance is not None else 0
            source = r.get("metadata", {}).get("filename", "unknown")

            print(f"  [{i}] (similarity: {similarity:.2%}) [{source}]")
            print(f"      {r['text'][:150]}...")
            print()

        return results

    def query_with_context(self, question: str, n_results: int = 3) -> str:
        """
        Sama kayak query(), tapi return sebagai formatted context string.
        Berguna buat dikasih ke LLM sebagai context.

        Args:
            question:  Pertanyaan user
            n_results: Jumlah dokumen context

        Returns:
            Formatted context string
        """
        results = self.query(question, n_results=n_results)

        if not results:
            return "No relevant context found in memory."

        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("filename", "unknown")
            context_parts.append(
                f"--- Context {i} (from: {source}) ---\n{r['text']}"
            )

        context = "\n\n".join(context_parts)
        return context

    # ──────────────────────────────────────────
    #  INGEST (shortcut)
    # ──────────────────────────────────────────

    def ingest_markdown(self, markdown_dir: Optional[str] = None):
        """Shortcut untuk ingest markdown ke memory."""
        self.store.ingest_markdown(markdown_dir)
        print(f"[RetrievalEngine] Memory updated — {self.store.count()} total documents.")

    # ──────────────────────────────────────────
    #  STATS
    # ──────────────────────────────────────────

    def memory_info(self) -> dict:
        """Info tentang memory yang tersimpan."""
        return self.store.info()


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
    print("  RETRIEVAL ENGINE — Quick Test")
    print("=" * 60)

    engine = RetrievalEngine()

    # Tambah sample data kalau kosong
    if engine.store.count() == 0:
        print("\n Adding sample knowledge...")
        engine.store.add_document(
            "knowledge_1",
            "Vector databases store embeddings for fast semantic retrieval. "
            "They convert text into mathematical vectors using models like BERT or MiniLM."
        )
        engine.store.add_document(
            "knowledge_2",
            "AI agents use persistent memory to remember past conversations "
            "and learned knowledge across sessions."
        )
        engine.store.add_document(
            "knowledge_3",
            "Reinforcement learning trains AI by giving rewards for good actions "
            "and penalties for bad ones. The agent learns optimal behavior over time."
        )
        engine.store.add_document(
            "knowledge_4",
            "RAG (Retrieval Augmented Generation) combines search with language models. "
            "First retrieve relevant documents, then generate answers based on context."
        )
        engine.store.add_document(
            "knowledge_5",
            "Embeddings are numerical representations of text. Similar texts have "
            "similar embeddings, allowing machines to understand semantic relationships."
        )

    # ── Test queries ──
    print("\n" + "=" * 60)
    print("  RUNNING QUERIES")
    print("=" * 60)

    queries = [
        "apa itu embeddings?",
        "bagaimana AI menyimpan knowledge?",
        "apa itu RAG?",
        "how does reinforcement learning work?",
    ]

    for q in queries:
        engine.query(q, n_results=3)
        print("-" * 60)

    # ── Context format ──
    print("\n" + "=" * 60)
    print("  CONTEXT FORMAT (for LLM)")
    print("=" * 60)
    ctx = engine.query_with_context("apa itu embeddings?", n_results=2)
    print(ctx)
