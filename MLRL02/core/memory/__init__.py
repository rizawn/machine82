"""
Memory subsystem — vector storage, context retrieval, and adaptive reinforcement.

Usage:
    from core.memory import VectorStore, MemoryContext, MemoryReinforcement
"""

__all__ = [
    "VectorStore",
    "MemoryContext",
    "ContextItem",
    "MemoryReinforcement",
    "MemoryRecord",
    "Document",
    "load_markdown",
    "extract_frontmatter",
    "chunk_by_headers",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in ("Document", "load_markdown", "extract_frontmatter", "chunk_by_headers"):
        from core.memory.loader import (
            Document, load_markdown, extract_frontmatter, chunk_by_headers
        )
        return {
            "Document": Document,
            "load_markdown": load_markdown,
            "extract_frontmatter": extract_frontmatter,
            "chunk_by_headers": chunk_by_headers
        }[name]
    if name == "VectorStore":
        from core.memory.vector_store import VectorStore as _VS
        return _VS
    if name == "MemoryContext":
        from core.memory.memory_context import MemoryContext as _MC
        return _MC
    if name == "ContextItem":
        from core.memory.memory_context import ContextItem as _CI
        return _CI
    if name == "MemoryReinforcement":
        from core.memory.memory_reinforcement import MemoryReinforcement as _MR
        return _MR
    if name == "MemoryRecord":
        from core.memory.memory_reinforcement import MemoryRecord as _MR2
        return _MR2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
