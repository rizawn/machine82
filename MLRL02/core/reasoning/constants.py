"""
REASONING CONSTANTS — Shared constants for the reasoning subsystem.

This module centralizes constants used across multiple reasoning modules
to avoid cross-module imports for simple values.
"""

# Stop words for concept/term extraction (English + Indonesian)
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "need", "must", "of", "in", "to", "for", "with", "on", "at",
    "from", "by", "about", "as", "into", "like", "through",
    "after", "over", "between", "out", "against", "during",
    "without", "before", "under", "around", "among", "and",
    "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own",
    "same", "than", "too", "very", "just", "also", "now",
    "that", "this", "these", "those", "it", "its", "they",
    "them", "we", "us", "our", "you", "your", "he", "him",
    "his", "she", "her", "what", "which", "who", "whom",
    "when", "where", "why", "how", "if", "then", "else",
    "there", "here", "one", "two", "first", "second",
    # Indonesian
    "apa", "itu", "yang", "dan", "atau", "dari", "untuk",
    "dengan", "pada", "dalam", "adalah", "bagaimana", "cara",
    "bisa", "tidak", "ini", "jika", "karena", "oleh",
    "tentang", "juga", "sudah", "akan", "secara", "merupakan",
    "memiliki", "yaitu", "sehingga", "agar",
})
