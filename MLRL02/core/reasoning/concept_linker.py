"""
CONCEPT LINKER — Internal Conceptual Understanding Layer

Responsibilities:
    - Detect relationships between concepts across documents
    - Build semantic links between markdown knowledge chunks
    - Group related ideas automatically
    - Identify recurring concepts across the knowledge base
    - Provide data structures ready for knowledge graph integration

Architecture:
    This module scans all loaded documents and builds a concept map:

        Documents → Extract Concepts → Compute Similarity → Build Links
            → Group into clusters → Output ConceptGraph

    The output can feed into:
    - Reasoning engine (for concept-aware answers)
    - Prompt builder (for contextually relevant memory injection)
    - Future knowledge graph (Neo4j, NetworkX, etc.)

Usage:
    linker = ConceptLinker()
    linker.scan_documents(docs)
    graph = linker.build_graph()
    print(graph.get_related("embeddings"))
"""

import os
import sys
import re
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

# Project root sys.path hack removed from module level (M-1)

from core.memory.loader import Document, load_markdown


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARKDOWN_DIR = os.path.join(BASE_DIR, "workspace", "markdown")

# Stop words for concept extraction (imported from shared constants)
from core.reasoning.constants import STOP_WORDS

# Minimum concept frequency to be considered "significant"
MIN_CONCEPT_FREQ = 1

# Minimum link strength to include in graph (0.0 – 1.0)
MIN_LINK_STRENGTH = 0.1


# ──────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class Concept:
    """
    A single concept extracted from the knowledge base.

    Attributes:
        name:        Normalized concept name (lowercase, no spaces)
        display:     Human-readable form
        frequency:   How many documents mention this concept
        documents:   List of source document names
        contexts:    Snippets where the concept appears
        category:    Auto-detected category (technology, process, etc.)
    """
    name: str
    display: str = ""
    frequency: int = 0
    documents: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    category: str = "unknown"

    def __post_init__(self):
        if not self.display:
            self.display = self.name.replace("_", " ").title()


@dataclass
class ConceptLink:
    """
    A relationship between two concepts.

    Attributes:
        source:     Source concept name
        target:     Target concept name
        strength:   Link strength (0.0 – 1.0)
        link_type:  Type of relationship
        evidence:   Text evidence for this link
        co_occurrence: How often they appear together
    """
    source: str
    target: str
    strength: float
    link_type: str = "association"  # association, hierarchy, dependency, contrast
    evidence: str = ""
    co_occurrence: int = 0


@dataclass
class ConceptCluster:
    """
    A group of related concepts.

    Attributes:
        name:       Cluster label
        concepts:   Member concept names
        cohesion:   How tightly related the cluster is (0.0 – 1.0)
    """
    name: str
    concepts: list[str] = field(default_factory=list)
    cohesion: float = 0.0


@dataclass
class ConceptGraph:
    """
    Full concept map — the output of ConceptLinker.

    Attributes:
        concepts:   All discovered concepts
        links:      Relationships between concepts
        clusters:   Auto-grouped concept clusters
        stats:      Summary statistics
    """
    concepts: dict[str, Concept] = field(default_factory=dict)
    links: list[ConceptLink] = field(default_factory=list)
    clusters: list[ConceptCluster] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def get_related(self, concept_name: str, min_strength: float = 0.0) -> list[ConceptLink]:
        """Find all concepts related to a given concept."""
        related = []
        for link in self.links:
            if link.source == concept_name or link.target == concept_name:
                if link.strength >= min_strength:
                    related.append(link)
        return sorted(related, key=lambda l: l.strength, reverse=True)

    def get_cluster_for(self, concept_name: str) -> Optional[ConceptCluster]:
        """Find which cluster a concept belongs to."""
        for cluster in self.clusters:
            if concept_name in cluster.concepts:
                return cluster
        return None

    def to_dict(self) -> dict:
        """Serialize to dict for export or knowledge graph import."""
        return {
            "concepts": {
                name: {
                    "display": c.display,
                    "frequency": c.frequency,
                    "documents": c.documents,
                    "category": c.category,
                }
                for name, c in self.concepts.items()
            },
            "links": [
                {
                    "source": l.source,
                    "target": l.target,
                    "strength": l.strength,
                    "type": l.link_type,
                    "co_occurrence": l.co_occurrence,
                }
                for l in self.links
            ],
            "clusters": [
                {
                    "name": c.name,
                    "concepts": c.concepts,
                    "cohesion": c.cohesion,
                }
                for c in self.clusters
            ],
            "stats": self.stats,
        }


# ──────────────────────────────────────────────
#  CONCEPT EXTRACTOR
# ──────────────────────────────────────────────

class ConceptExtractor:
    """
    Extract concepts from document text.

    Strategy:
    1. Extract n-grams (1-3 words) that appear meaningful
    2. Filter stop words and low-signal terms
    3. Normalize and count occurrences
    4. Tag with source document
    """

    # Patterns that indicate meaningful technical concepts
    TECH_PATTERNS = [
        re.compile(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z]?[a-zA-Z]*){0,2}\b'),  # TitleCase terms
        re.compile(r'\b\w+(?:[_-]\w+)+\b'),  # snake_case or kebab-case
        re.compile(r'\b[A-Z]{2,}\b'),  # Acronyms
    ]

    # Known category keywords
    CATEGORY_KEYWORDS = {
        "technology": {"database", "api", "model", "vector", "embedding",
                       "algorithm", "system", "engine", "server", "client",
                       "chroma", "ollama", "langchain", "python"},
        "process": {"ingest", "search", "retrieve", "load", "build",
                    "extract", "chunk", "embed", "train", "deploy"},
        "concept": {"memory", "knowledge", "context", "reasoning",
                    "semantic", "agent", "learning", "intelligence"},
        "structure": {"header", "file", "document", "chunk", "section",
                      "module", "class", "function", "method"},
    }

    def extract_from_document(self, doc: Document) -> list[Concept]:
        """Extract concepts from a single document."""
        concepts = self._extract_ngrams(doc.page_content)
        source = doc.metadata.get("source", "unknown")

        result = []
        for term, count in concepts.items():
            normalized = self._normalize(term)
            if not normalized or len(normalized) < 2:
                continue

            concept = Concept(
                name=normalized,
                frequency=count,
                documents=[source],
                contexts=self._get_context_snippets(doc.page_content, term),
                category=self._categorize(term),
            )
            result.append(concept)

        return result

    def _extract_ngrams(self, text: str) -> Counter:
        """
        Extract meaningful n-grams (1-3 words) from text.

        Filters stop words, short tokens, and non-informative terms.
        """
        # Clean text: remove markdown, normalize
        cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove images
        cleaned = re.sub(r'#{1,6}\s+', '', cleaned)      # Remove header marks
        cleaned = re.sub(r'```[\s\S]*?```', '', cleaned) # Remove code blocks
        cleaned = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', cleaned)  # Keep link text

        words = cleaned.lower().split()
        ngrams = Counter()

        # Unigrams (single meaningful words)
        for w in words:
            w_clean = re.sub(r'[^\w]', '', w)
            if (len(w_clean) >= 4
                    and w_clean not in STOP_WORDS
                    and not w_clean.isdigit()):
                ngrams[w_clean] += 1

        # Bigrams (two-word phrases)
        for i in range(len(words) - 1):
            w1 = re.sub(r'[^\w]', '', words[i])
            w2 = re.sub(r'[^\w]', '', words[i + 1])
            if (w1 and w2
                    and w1 not in STOP_WORDS
                    and w2 not in STOP_WORDS
                    and len(w1) >= 3
                    and len(w2) >= 3):
                bigram = f"{w1}_{w2}"
                ngrams[bigram] += 1

        # Trigrams (three-word phrases, limited)
        for i in range(len(words) - 2):
            w1 = re.sub(r'[^\w]', '', words[i])
            w2 = re.sub(r'[^\w]', '', words[i + 1])
            w3 = re.sub(r'[^\w]', '', words[i + 2])
            if (w1 and w2 and w3
                    and w1 not in STOP_WORDS
                    and w3 not in STOP_WORDS
                    and len(w1) >= 3
                    and len(w3) >= 3):
                trigram = f"{w1}_{w2}_{w3}"
                # Only count if the middle word is short (preposition-like phrase)
                if len(w2) <= 4:
                    ngrams[trigram] += 1

        return ngrams

    def _normalize(self, term: str) -> str:
        """Normalize a concept term."""
        term = term.lower().strip()
        term = re.sub(r'[^\w]', '', term)
        return term

    def _get_context_snippets(self, text: str, term: str, max_snippets: int = 2) -> list[str]:
        """Extract short context snippets where the term appears."""
        snippets = []
        lower_text = text.lower()
        lower_term = term.lower()

        start = 0
        while len(snippets) < max_snippets:
            pos = lower_text.find(lower_term, start)
            if pos == -1:
                break

            # Extract surrounding context
            snippet_start = max(0, pos - 40)
            snippet_end = min(len(text), pos + len(term) + 40)
            snippet = text[snippet_start:snippet_end].strip()

            # Add ellipsis if truncated
            if snippet_start > 0:
                snippet = "..." + snippet
            if snippet_end < len(text):
                snippet = snippet + "..."

            snippets.append(snippet)
            start = pos + 1

        return snippets

    def _categorize(self, term: str) -> str:
        """Auto-categorize a concept based on keywords."""
        term_lower = term.lower()

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in term_lower or term_lower in kw:
                    return category

        return "unknown"


# ──────────────────────────────────────────────
#  LINK BUILDER
# ──────────────────────────────────────────────

class LinkBuilder:
    """
    Build relationships between concepts.

    Strategies:
    1. Co-occurrence: concepts appearing in the same document
    2. Textual similarity: concepts with similar names/contexts
    3. Contextual overlap: concepts sharing context snippets
    """

    def build_links(
        self,
        concepts: dict[str, Concept],
    ) -> list[ConceptLink]:
        """
        Build all links between concepts.

        Args:
            concepts: Dict of concept_name → Concept

        Returns:
            List of ConceptLink objects
        """
        links: list[ConceptLink] = []
        concept_names = list(concepts.keys())

        for i, name_a in enumerate(concept_names):
            for name_b in concept_names[i + 1:]:
                link = self._compute_link(name_a, name_b, concepts)
                if link and link.strength >= MIN_LINK_STRENGTH:
                    links.append(link)

        return sorted(links, key=lambda l: l.strength, reverse=True)

    def _compute_link(
        self,
        name_a: str,
        name_b: str,
        concepts: dict[str, Concept],
    ) -> Optional[ConceptLink]:
        """Compute a single link between two concepts."""
        concept_a = concepts[name_a]
        concept_b = concepts[name_b]

        # Strategy 1: Co-occurrence (shared documents)
        docs_a = set(concept_a.documents)
        docs_b = set(concept_b.documents)
        shared_docs = docs_a & docs_b

        if not shared_docs and not self._has_textual_similarity(name_a, name_b):
            return None

        co_occurrence = len(shared_docs)
        total_docs = len(docs_a | docs_b)
        co_occurrence_score = co_occurrence / total_docs if total_docs > 0 else 0

        # Strategy 2: Textual similarity
        text_similarity = self._text_similarity(name_a, name_b)

        # Strategy 3: Context overlap
        context_overlap = self._context_overlap(concept_a, concept_b)

        # Combined strength
        strength = (
            co_occurrence_score * 0.4 +
            text_similarity * 0.35 +
            context_overlap * 0.25
        )

        if strength < MIN_LINK_STRENGTH:
            return None

        # Determine link type
        link_type = self._classify_link_type(name_a, name_b, strength)

        # Build evidence from shared contexts
        evidence = self._build_evidence(concept_a, concept_b)

        return ConceptLink(
            source=name_a,
            target=name_b,
            strength=round(strength, 3),
            link_type=link_type,
            evidence=evidence,
            co_occurrence=co_occurrence,
        )

    def _has_textual_similarity(self, a: str, b: str) -> bool:
        """Check if two concept names share significant text."""
        words_a = set(a.split("_"))
        words_b = set(b.split("_"))
        return bool(words_a & words_b)

    def _text_similarity(self, a: str, b: str) -> float:
        """Compute text-based similarity between concept names."""
        words_a = set(a.split("_"))
        words_b = set(b.split("_"))

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        jaccard = len(intersection) / len(union)

        # Bonus for shared prefix (hierarchical relationship)
        if a.startswith(b) or b.startswith(a):
            jaccard = min(1.0, jaccard + 0.2)

        return jaccard

    def _context_overlap(self, a: Concept, b: Concept) -> float:
        """Compute context snippet overlap between two concepts."""
        if not a.contexts or not b.contexts:
            return 0.0

        # Compare context snippets via word overlap
        total_overlap = 0.0
        comparisons = 0

        for ctx_a in a.contexts[:2]:  # Limit comparisons
            for ctx_b in b.contexts[:2]:
                words_a = set(ctx_a.lower().split())
                words_b = set(ctx_b.lower().split())
                words_a = {w for w in words_a if len(w) > 3}
                words_b = {w for w in words_b if len(w) > 3}

                if words_a and words_b:
                    overlap = len(words_a & words_b) / len(words_a | words_b)
                    total_overlap += overlap
                    comparisons += 1

        return total_overlap / comparisons if comparisons > 0 else 0.0

    def _classify_link_type(
        self, a: str, b: str, strength: float
    ) -> str:
        """Classify the type of relationship."""
        # Hierarchical: one contains the other
        if a.startswith(b) or b.startswith(a):
            return "hierarchy"

        # Contrast: opposing concepts
        contrast_pairs = {
            ("client", "server"), ("input", "output"),
            ("train", "infer"), ("encode", "decode"),
        }
        pair = tuple(sorted([a, b]))
        for cp in contrast_pairs:
            if tuple(sorted(cp)) == pair:
                return "contrast"

        # Dependency: one concept depends on another
        dependency_words = {"use", "build", "create", "require", "need"}
        if any(w in a for w in dependency_words) or any(w in b for w in dependency_words):
            return "dependency"

        return "association"

    def _build_evidence(self, a: Concept, b: Concept) -> str:
        """Build evidence string from shared contexts."""
        shared = set(a.documents) & set(b.documents)
        if shared:
            return f"Co-occurs in: {', '.join(sorted(shared))}"
        return ""


# ──────────────────────────────────────────────
#  CLUSTER BUILDER
# ──────────────────────────────────────────────

class ClusterBuilder:
    """
    Group related concepts into clusters.

    Strategy: Simple agglomerative clustering based on link strength.
    Concepts connected by strong links form a cluster.
    """

    def build_clusters(
        self,
        concepts: dict[str, Concept],
        links: list[ConceptLink],
        min_cluster_size: int = 2,
    ) -> list[ConceptCluster]:
        """
        Build concept clusters from links.

        Args:
            concepts: All concepts
            links: All links
            min_cluster_size: Minimum concepts per cluster

        Returns:
            List of ConceptCluster objects
        """
        if not links:
            # Each concept is its own cluster
            return [
                ConceptCluster(
                    name=c.display,
                    concepts=[c.name],
                    cohesion=0.0,
                )
                for c in concepts.values()
            ]

        # Union-Find for grouping
        parent = {name: name for name in concepts}
        link_strength = {}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str, strength: float):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
            key = tuple(sorted([rx, ry]))
            link_strength[key] = max(
                link_strength.get(key, 0), strength
            )

        # Group by strong links
        for link in links:
            if link.strength >= 0.2:
                union(link.source, link.target, link.strength)

        # Collect clusters
        groups: dict[str, list[str]] = {}
        for name in concepts:
            root = find(name)
            if root not in groups:
                groups[root] = []
            groups[root].append(name)

        # Build cluster objects
        clusters = []
        for root, members in groups.items():
            if len(members) < min_cluster_size:
                continue

            # Compute cohesion (avg link strength within cluster)
            member_set = set(members)
            internal_links = [
                l for l in links
                if l.source in member_set and l.target in member_set
            ]
            cohesion = (
                sum(l.strength for l in internal_links) / len(internal_links)
                if internal_links else 0.0
            )

            # Name cluster after most frequent concept
            cluster_name = self._name_cluster(members, concepts)

            clusters.append(ConceptCluster(
                name=cluster_name,
                concepts=sorted(members),
                cohesion=round(cohesion, 3),
            ))

        return sorted(clusters, key=lambda c: c.cohesion, reverse=True)

    def _name_cluster(
        self, members: list[str], concepts: dict[str, Concept]
    ) -> str:
        """Generate a descriptive name for a cluster."""
        if not members:
            return "Unknown"

        # Use the concept with highest frequency as the cluster name
        best = max(
            members,
            key=lambda m: concepts.get(m, Concept(name=m)).frequency,
        )
        return concepts[best].display


# ──────────────────────────────────────────────
#  CONCEPT LINKER (Main Class)
# ──────────────────────────────────────────────

class ConceptLinker:
    """
    Main concept linking engine.

    Orchestrates concept extraction, link building,
    and clustering to produce a ConceptGraph.

    Usage:
        linker = ConceptLinker()
        linker.scan_directory("workspace/markdown")
        graph = linker.build_graph()
        print(graph.get_related("embeddings"))
    """

    def __init__(self):
        self._extractor = ConceptExtractor()
        self._link_builder = LinkBuilder()
        self._cluster_builder = ClusterBuilder()

        self._concepts: dict[str, Concept] = {}
        self._links: list[ConceptLink] = []
        self._clusters: list[ConceptCluster] = []
        self._scanned = False

    def scan_documents(self, documents: list[Document]):
        """
        Scan a list of Document objects for concepts.

        Args:
            documents: List of Document from loader
        """
        self._concepts.clear()
        self._links.clear()
        self._clusters.clear()

        # Extract concepts from each document
        for doc in documents:
            doc_concepts = self._extractor.extract_from_document(doc)

            for concept in doc_concepts:
                if concept.name in self._concepts:
                    # Merge with existing concept
                    existing = self._concepts[concept.name]
                    existing.frequency += concept.frequency
                    existing.documents.extend(concept.documents)
                    existing.contexts.extend(concept.contexts[:1])
                else:
                    self._concepts[concept.name] = concept

        self._scanned = True

    def scan_directory(self, dir_path: Optional[str] = None):
        """
        Scan all markdown files in a directory.

        Args:
            dir_path: Path to markdown directory (default: MARKDOWN_DIR)
        """
        path = dir_path or MARKDOWN_DIR
        documents = load_markdown(path)
        self.scan_documents(documents)

    def build_graph(self) -> ConceptGraph:
        """
        Build the full concept graph.

        Must call scan_documents() or scan_directory() first.

        Returns:
            ConceptGraph with concepts, links, and clusters
        """
        if not self._scanned:
            raise RuntimeError(
                "No documents scanned. Call scan_documents() or "
                "scan_directory() first."
            )

        # Build links
        self._links = self._link_builder.build_links(self._concepts)

        # Build clusters
        self._clusters = self._cluster_builder.build_clusters(
            self._concepts, self._links
        )

        # Compute stats
        stats = {
            "total_concepts": len(self._concepts),
            "total_links": len(self._links),
            "total_clusters": len(self._clusters),
            "avg_links_per_concept": (
                round(len(self._links) * 2 / len(self._concepts), 1)
                if self._concepts else 0
            ),
            "most_connected": self._most_connected(),
            "strongest_link": self._strongest_link(),
        }

        return ConceptGraph(
            concepts=self._concepts,
            links=self._links,
            clusters=self._clusters,
            stats=stats,
        )

    def _most_connected(self) -> str:
        """Find the concept with the most connections."""
        if not self._links:
            return "none"

        degree: Counter = Counter()
        for link in self._links:
            degree[link.source] += 1
            degree[link.target] += 1

        return degree.most_common(1)[0][0] if degree else "none"

    def _strongest_link(self) -> str:
        """Find the strongest link between concepts."""
        if not self._links:
            return "none"

        strongest = max(self._links, key=lambda l: l.strength)
        return f"{strongest.source} ↔ {strongest.target} ({strongest.strength:.0%})"

    def print_summary(self, graph: ConceptGraph):
        """Pretty-print concept graph summary."""
        print("\n" + "=" * 60)
        print("  🧠 CONCEPT GRAPH SUMMARY")
        print("=" * 60)

        print(f"\n📊 Stats:")
        for k, v in graph.stats.items():
            print(f"   {k}: {v}")

        print(f"\n🏷️  Top Concepts (by frequency):")
        top_concepts = sorted(
            graph.concepts.values(),
            key=lambda c: c.frequency,
            reverse=True,
        )[:10]
        for c in top_concepts:
            print(f"   • {c.display} ({c.frequency}x) [{c.category}] "
                  f"in {', '.join(c.documents[:3])}")

        if graph.links:
            print(f"\n🔗 Top Links:")
            for link in graph.links[:10]:
                print(f"   {link.source} ↔ {link.target} "
                      f"({link.strength:.0%}, {link.link_type})")

        if graph.clusters:
            print(f"\n📦 Clusters ({len(graph.clusters)}):")
            for cluster in graph.clusters:
                members = ", ".join(cluster.concepts[:5])
                print(f"   📁 {cluster.name} (cohesion: {cluster.cohesion:.0%})")
                print(f"      [{members}]")


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
    print("  CONCEPT LINKER — Quick Test")
    print("=" * 60 + "\n")

    linker = ConceptLinker()
    linker.scan_directory()
    graph = linker.build_graph()
    linker.print_summary(graph)

    # Test related concepts lookup
    print("\n" + "=" * 60)
    print("  RELATED CONCEPTS LOOKUP")
    print("=" * 60)

    for term in ["embedding", "semantic", "memory", "vector"]:
        related = graph.get_related(term, min_strength=0.15)
        if related:
            print(f"\n🔍 Related to '{term}':")
            for r in related[:5]:
                other = r.target if r.source == term else r.source
                print(f"   → {other} ({r.strength:.0%}, {r.link_type})")
        else:
            print(f"\n🔍 No links found for '{term}'")

    print("\n" + "=" * 60)
    print("  EXPORT-READY DICT (first 3 concepts)")
    print("=" * 60)
    export = graph.to_dict()
    for name, data in list(export["concepts"].items())[:3]:
        print(f"\n{name}: {data}")
