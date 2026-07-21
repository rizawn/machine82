"""
QUERY CLASSIFIER — Intelligent Routing Layer

This module determines whether a user query requires semantic memory retrieval
(internal project knowledge) or should be handled by the LLM's general knowledge.

Classification Categories:
    - "memory":  Queries about internal docs, project structure, or user-specific notes.
    - "general": Broad questions, general definitions, or creative tasks.

Usage:
    classifier = QueryClassifier()
    category = classifier.classify("What is inside roadmap.md?")
    # returns "memory"
"""

import re

class QueryClassifier:
    """
    Lightweight semantic classifier for routing user queries.
    Uses pattern matching and keyword analysis to detect intent.
    """

    # Keywords that strongly indicate project/internal knowledge
    MEMORY_KEYWORDS = [
        r"\bmy notes\b",
        r"\bcatatan saya\b",
        r"\bproject\b",
        r"\bproyek\b",
        r"\bworkspace\b",
        r"\bmemory\b",
        r"\bmemori\b",
        r"\bthis repo\b",
        r"\brepo ini\b",
        r"\binternal\b",
        r"\bstored\b",
        r"\broadmap\b",
        r"\binside\b",
        r"\bdidalam\b",
        r"\bfrom my\b",
        r"\bdari my\b",
        r"\bin my\b",
        r"\bdi my\b",
        r"\bdocument\b",
        r"\bdokumen\b",
        r"\bdoc\b",
        r"\bfile\b",
        r"\blocal\b",
        r"\blokas\b",
        r"\bcodebase\b",
    ]

    # File extensions that imply memory retrieval
    FILE_EXTENSIONS = [
        r"\.md\b",
        r"\.py\b",
        r"\.txt\b",
        r"\.json\b",
        r"\.pdf\b",
        r"\.csv\b",
        r"\.log\b",
    ]

    # Patterns that suggest general knowledge
    GENERAL_PATTERNS = [
        r"^(what is|apa itu)\b(?!.*(inside|in my|my notes|didalam|catatan saya))",
        r"^(explain|jelaskan)\b(?!.*(inside|in my|my notes|didalam|catatan saya))",
        r"^(how (do|does)|bagaimana)\b",
        r"\bideas for\b",
        r"\bide untuk\b",
        r"\bgive me\b",
        r"\bberikan saya\b",
        r"\btell me about\b",
        r"\bceritakan tentang\b",
    ]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def classify(self, query: str) -> str:
        """
        Classifies the query into 'memory' or 'general'.
        
        Args:
            query: The user's question.
            
        Returns:
            "memory" or "general"
        """
        query_lower = query.lower().strip()

        # 1. Check for specific file extensions
        if any(re.search(ext, query_lower) for ext in self.FILE_EXTENSIONS):
            if self.verbose:
                print(f"[Classifier] Detected file extension in: '{query}'")
            return "memory"

        # 2. Check for memory keywords
        if any(re.search(kw, query_lower) for kw in self.MEMORY_KEYWORDS):
            if self.verbose:
                print(f"[Classifier] Detected memory keyword in: '{query}'")
            return "memory"

        # 3. Check for general knowledge patterns
        # We check this AFTER memory because "Explain X from my notes" 
        # should be memory, but "Explain X" should be general.
        if any(re.search(pat, query_lower) for pat in self.GENERAL_PATTERNS):
            if self.verbose:
                print(f"[Classifier] Detected general pattern in: '{query}'")
            return "general"

        # 4. Default to general if no strong indicators
        # But if the question is very specific about "What is [X]?" it's usually general
        # unless it mentions the internal context.
        return "general"

# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    classifier = QueryClassifier(verbose=True)

    test_cases = [
        "What is inside roadmap.md?",
        "Give me ideas for login pages",
        "Explain embeddings from my notes",
        "What is machine learning?",
        "How do I use this project?",
        "Tell me a joke",
        "Read the main.py file",
        "What are embeddings?",
    ]

    print("=" * 60)
    print("  QUERY CLASSIFIER — Test Results")
    print("=" * 60)

    for q in test_cases:
        category = classifier.classify(q)
        print(f"Query:    '{q}'")
        print(f"Category:  {category}")
        print("-" * 30)
