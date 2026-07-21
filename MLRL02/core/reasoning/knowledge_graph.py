"""
KNOWLEDGE GRAPH — Internal Cognitive Graph System

Requirements:
    - Build relationships between concepts
    - Create graph structures from markdown knowledge
    - Support semantic linking
    - Detect important nodes
    - Enable future visualization support

Architecture:
    This module builds a proper graph data structure from the
    concept links discovered by ConceptLinker:

        Documents → Concepts → Links → Graph Analysis
            → Centrality → Clustering → Visualization Export

    The graph supports:
    - Node importance (degree, betweenness, PageRank)
    - Path finding between concepts
    - Community detection
    - Export to formats ready for visualization (JSON, GEXF, DOT)

Usage:
    graph = KnowledgeGraph()
    graph.build_from_markdown()
    print(graph.get_important_nodes())
    print(graph.find_path("embeddings", "semantic_search"))
    graph.export_json("workspace/knowledge_graph.json")
"""

import os
import sys
import json
import math
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, deque

# Project root sys.path hack removed from module level (M-1)

from core.reasoning.concept_linker import (
    ConceptLinker,
    ConceptGraph,
    Concept,
    ConceptLink,
)


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DIR = os.path.join(BASE_DIR, "workspace")

# Minimum link strength to include in graph
MIN_EDGE_WEIGHT = 0.15

# PageRank parameters
PAGE_RANK_DAMPING = 0.85
PAGE_RANK_ITERATIONS = 50


# ──────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class GraphNode:
    """
    A node in the knowledge graph.

    Attributes:
        id:           Unique node identifier (concept name)
        label:        Human-readable label
        category:     Concept category
        frequency:    How often this concept appears
        documents:    Source documents
        degree:       Number of connections
        centrality:   Betweenness centrality score
        pagerank:     PageRank score
        community:    Detected community/cluster ID
    """
    id: str
    label: str = ""
    category: str = "unknown"
    frequency: int = 0
    documents: list[str] = field(default_factory=list)
    degree: int = 0
    centrality: float = 0.0
    pagerank: float = 0.0
    community: int = -1

    def __post_init__(self):
        if not self.label:
            self.label = self.id.replace("_", " ").title()

    @property
    def importance(self) -> float:
        """Combined importance score."""
        return (
            self.centrality * 0.4 +
            self.pagerank * 0.3 +
            min(1.0, self.degree / 10) * 0.2 +
            min(1.0, self.frequency / 5) * 0.1
        )


@dataclass
class GraphEdge:
    """
    An edge between two nodes in the knowledge graph.

    Attributes:
        source:     Source node ID
        target:     Target node ID
        weight:     Edge weight (0.0 – 1.0)
        link_type:  Type of relationship
        evidence:   Why this edge exists
    """
    source: str
    target: str
    weight: float
    link_type: str = "association"
    evidence: str = ""


@dataclass
class GraphPath:
    """
    A path between two nodes.

    Attributes:
        nodes:       Sequence of node IDs
        edges:       Edges traversed
        total_weight: Product of edge weights along path
        length:      Number of hops
    """
    nodes: list[str]
    edges: list[GraphEdge]
    total_weight: float
    length: int = 0

    def __post_init__(self):
        self.length = len(self.nodes) - 1


# ──────────────────────────────────────────────
#  KNOWLEDGE GRAPH
# ──────────────────────────────────────────────

class KnowledgeGraph:
    """
    Cognitive knowledge graph built from markdown concepts.

    Provides:
    - Graph construction from ConceptLinker output
    - Node importance analysis (degree, centrality, PageRank)
    - Path finding between concepts
    - Community detection
    - Visualization export (JSON, DOT, GEXF)
    """

    def __init__(self):
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._concept_graph: Optional[ConceptGraph] = None
        self._built = False

    # ──────────────────────────────────────────
    #  BUILD
    # ──────────────────────────────────────────

    def build_from_markdown(self, markdown_dir: Optional[str] = None):
        """
        Build the full knowledge graph from markdown files.

        Args:
            markdown_dir: Path to markdown directory
        """
        linker = ConceptLinker()
        linker.scan_directory(markdown_dir)
        self._concept_graph = linker.build_graph()
        self._build_from_concept_graph(self._concept_graph)

    def build_from_concept_graph(self, concept_graph: ConceptGraph):
        """
        Build from an existing ConceptGraph.

        Args:
            concept_graph: Output from ConceptLinker.build_graph()
        """
        self._concept_graph = concept_graph
        self._build_from_concept_graph(concept_graph)

    def _build_from_concept_graph(self, concept_graph: ConceptGraph):
        """Internal: build nodes and edges from concept graph data."""
        self._nodes.clear()
        self._edges.clear()
        self._adjacency.clear()

        # Create nodes
        for name, concept in concept_graph.concepts.items():
            node = GraphNode(
                id=name,
                label=concept.display,
                category=concept.category,
                frequency=concept.frequency,
                documents=concept.documents,
            )
            self._nodes[name] = node

        # Create edges
        for link in concept_graph.links:
            if link.strength >= MIN_EDGE_WEIGHT:
                edge = GraphEdge(
                    source=link.source,
                    target=link.target,
                    weight=link.strength,
                    link_type=link.link_type,
                    evidence=link.evidence,
                )
                self._edges.append(edge)

                # Build adjacency list
                self._adjacency[link.source].append((link.target, link.strength))
                self._adjacency[link.target].append((link.source, link.strength))

        # Assign communities from clusters
        if concept_graph.clusters:
            for i, cluster in enumerate(concept_graph.clusters):
                for member in cluster.concepts:
                    if member in self._nodes:
                        self._nodes[member].community = i

        # Compute graph metrics
        self._compute_degree()
        self._compute_betweenness()
        self._compute_pagerank()

        self._built = True

    # ──────────────────────────────────────────
    #  NODE QUERIES
    # ──────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_neighbors(self, node_id: str, min_weight: float = 0.0) -> list[tuple[str, float]]:
        """Get neighbors of a node with edge weights."""
        neighbors = self._adjacency.get(node_id, [])
        return [(nid, w) for nid, w in neighbors if w >= min_weight]

    def get_important_nodes(self, n: int = 10, by: str = "importance") -> list[GraphNode]:
        """
        Get the most important nodes.

        Args:
            n:  Number of nodes
            by: Sorting criterion ('importance', 'degree', 'pagerank', 'centrality')

        Returns:
            Top N nodes sorted by importance
        """
        if not self._nodes:
            return []

        sort_key = {
            "importance": lambda n: n.importance,
            "degree": lambda n: n.degree,
            "pagerank": lambda n: n.pagerank,
            "centrality": lambda n: n.centrality,
        }.get(by, lambda n: n.importance)

        return sorted(self._nodes.values(), key=sort_key, reverse=True)[:n]

    def get_nodes_by_category(self, category: str) -> list[GraphNode]:
        """Get all nodes in a category."""
        return [
            n for n in self._nodes.values()
            if n.category == category
        ]

    # ──────────────────────────────────────────
    #  PATH FINDING
    # ──────────────────────────────────────────

    def find_path(
        self,
        source: str,
        target: str,
        max_depth: int = 5,
    ) -> Optional[GraphPath]:
        """
        Find a path between two concepts.

        Uses BFS for unweighted shortest path.

        Args:
            source:    Starting concept
            target:    Ending concept
            max_depth: Maximum search depth

        Returns:
            GraphPath or None if no path exists
        """
        if source not in self._nodes or target not in self._nodes:
            return None

        if source == target:
            return GraphPath(
                nodes=[source],
                edges=[],
                total_weight=1.0,
            )

        # BFS
        queue = deque([(source, [source], [], 1.0)])
        visited = {source}

        while queue:
            current, path, edge_path, weight = queue.popleft()

            if len(path) > max_depth:
                continue

            for neighbor, edge_weight in self._adjacency.get(current, []):
                if neighbor == target:
                    # Find the edge
                    edge = self._find_edge(current, neighbor)
                    final_edges = edge_path + ([edge] if edge else [])
                    return GraphPath(
                        nodes=path + [neighbor],
                        edges=final_edges,
                        total_weight=weight * (edge.weight if edge else 1.0),
                    )

                if neighbor not in visited:
                    visited.add(neighbor)
                    edge = self._find_edge(current, neighbor)
                    new_edges = edge_path + ([edge] if edge else [])
                    queue.append((
                        neighbor,
                        path + [neighbor],
                        new_edges,
                        weight * (edge.weight if edge else 1.0),
                    ))

        return None

    def find_all_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 4,
        max_paths: int = 5,
    ) -> list[GraphPath]:
        """
        Find multiple paths between two concepts.

        Uses DFS with depth limiting.

        Args:
            source:    Starting concept
            target:    Ending concept
            max_depth: Maximum path length
            max_paths: Maximum number of paths to find

        Returns:
            List of GraphPath objects
        """
        if source not in self._nodes or target not in self._nodes:
            return []

        paths: list[GraphPath] = []
        visited = {source}

        def dfs(current: str, path: list[str], edge_path: list[GraphEdge], weight: float):
            if len(paths) >= max_paths:
                return
            if len(path) > max_depth:
                return
            if current == target:
                paths.append(GraphPath(
                    nodes=list(path),
                    edges=list(edge_path),
                    total_weight=weight,
                ))
                return

            for neighbor, edge_weight in self._adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    edge = self._find_edge(current, neighbor)
                    if edge:
                        edge_path.append(edge)
                    path.append(neighbor)

                    dfs(neighbor, path, edge_path, weight * (edge.weight if edge else 1.0))

                    path.pop()
                    if edge:
                        edge_path.pop()
                    visited.remove(neighbor)

        dfs(source, [source], [], 1.0)

        # Sort by weight (highest first)
        return sorted(paths, key=lambda p: p.total_weight, reverse=True)

    def _find_edge(self, source: str, target: str) -> Optional[GraphEdge]:
        """Find an edge between two nodes."""
        for edge in self._edges:
            if (edge.source == source and edge.target == target) or \
               (edge.source == target and edge.target == source):
                return edge
        return None

    # ──────────────────────────────────────────
    #  GRAPH METRICS
    # ──────────────────────────────────────────

    def _compute_degree(self):
        """Compute degree for all nodes."""
        for node_id in self._nodes:
            self._nodes[node_id].degree = len(self._adjacency.get(node_id, []))

    def _compute_betweenness(self):
        """
        Compute betweenness centrality for all nodes.

        Brandes' algorithm simplified for unweighted graphs.
        """
        centrality = {node: 0.0 for node in self._nodes}

        for source in self._nodes:
            # BFS from source
            stack = []
            pred = defaultdict(list)
            sigma = defaultdict(int)
            sigma[source] = 1
            dist = {source: 0}
            queue = deque([source])

            while queue:
                v = queue.popleft()
                stack.append(v)

                for w, _ in self._adjacency.get(v, []):
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        queue.append(w)

                    if dist.get(w) == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            # Accumulate
            delta = defaultdict(float)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != source:
                    centrality[w] += delta[w]

        # Normalize
        n = len(self._nodes)
        if n > 2:
            norm = 1.0 / ((n - 1) * (n - 2))
            for node_id in centrality:
                centrality[node_id] *= norm
                self._nodes[node_id].centrality = centrality[node_id]

    def _compute_pagerank(self):
        """
        Compute PageRank for all nodes.

        Simple iterative PageRank on the concept graph.
        """
        n = len(self._nodes)
        if n == 0:
            return

        # Initialize
        pr = {node: 1.0 / n for node in self._nodes}

        for _ in range(PAGE_RANK_ITERATIONS):
            new_pr = {}
            for node in self._nodes:
                # Sum of incoming PageRank
                rank_sum = 0.0
                for neighbor, weight in self._adjacency.get(node, []):
                    out_degree = len(self._adjacency.get(neighbor, []))
                    if out_degree > 0:
                        rank_sum += pr[neighbor] * weight / out_degree

                new_pr[node] = (
                    (1 - PAGE_RANK_DAMPING) / n +
                    PAGE_RANK_DAMPING * rank_sum
                )

            # Normalize
            total = sum(new_pr.values())
            if total > 0:
                for node in new_pr:
                    new_pr[node] /= total

            pr = new_pr

        for node_id, score in pr.items():
            if node_id in self._nodes:
                self._nodes[node_id].pagerank = score

    # ──────────────────────────────────────────
    #  COMMUNITY DETECTION
    # ──────────────────────────────────────────

    def get_communities(self) -> dict[int, list[str]]:
        """Get nodes grouped by community."""
        communities: dict[int, list[str]] = defaultdict(list)

        for node in self._nodes.values():
            if node.community >= 0:
                communities[node.community].append(node.id)

        return dict(communities)

    def get_community_info(self) -> list[dict]:
        """Get summary info for each community."""
        communities = self.get_communities()
        result = []

        for comm_id, members in communities.items():
            nodes = [self._nodes[m] for m in members if m in self._nodes]
            avg_importance = (
                sum(n.importance for n in nodes) / len(nodes)
                if nodes else 0
            )

            # Find dominant category
            categories = defaultdict(int)
            for n in nodes:
                categories[n.category] += 1
            dominant = max(categories, key=categories.get) if categories else "unknown"

            result.append({
                "id": comm_id,
                "size": len(members),
                "members": members[:10],  # First 10
                "avg_importance": round(avg_importance, 3),
                "dominant_category": dominant,
            })

        return sorted(result, key=lambda c: c["size"], reverse=True)

    # ──────────────────────────────────────────
    #  EXPORT
    # ──────────────────────────────────────────

    def export_json(self, filepath: Optional[str] = None) -> dict:
        """
        Export graph as JSON — ready for visualization tools.

        Format compatible with D3.js force-directed graphs.

        Args:
            filepath: Where to save (optional)

        Returns:
            Dict with 'nodes' and 'links' arrays
        """
        data = {
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "category": n.category,
                    "frequency": n.frequency,
                    "degree": n.degree,
                    "importance": round(n.importance, 3),
                    "community": n.community,
                }
                for n in self._nodes.values()
            ],
            "links": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": round(e.weight, 3),
                    "type": e.link_type,
                }
                for e in self._edges
            ],
            "meta": {
                "total_nodes": len(self._nodes),
                "total_edges": len(self._edges),
                "communities": len(self.get_communities()),
            },
        }

        if filepath:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        return data

    def export_dot(self) -> str:
        """
        Export as DOT format — for Graphviz visualization.

        Returns:
            DOT language string
        """
        lines = ["digraph KnowledgeGraph {", "  rankdir=LR;", "  node [shape=box, style=filled];", ""]

        # Color map for categories
        colors = {
            "technology": "#4A90D9",
            "process": "#50B960",
            "concept": "#D94A4A",
            "structure": "#D9B44A",
            "unknown": "#888888",
        }

        # Nodes
        for node in self._nodes.values():
            color = colors.get(node.category, "#888888")
            label = node.label.replace('"', '\\"')
            size = max(0.5, min(2.0, node.importance * 2))
            lines.append(
                f'  "{node.id}" [label="{label}", fillcolor="{color}", '
                f'fontsize={8 + node.degree}, width={size}];'
            )

        lines.append("")

        # Edges
        for edge in self._edges:
            width = max(0.5, edge.weight * 3)
            lines.append(
                f'  "{edge.source}" -> "{edge.target}" '
                f'[penwidth={width:.1f}, label="{edge.weight:.2f}"];'
            )

        lines.append("}")
        return "\n".join(lines)

    def export_gexf(self, filepath: Optional[str] = None) -> str:
        """
        Export as GEXF format — for Gephi visualization.

        Args:
            filepath: Where to save (optional)

        Returns:
            GEXF XML string
        """
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gexf xmlns="http://www.gexf.net/1.3" version="1.3">',
            '  <graph defaultedgetype="undirected">',
            '    <attributes class="node" mode="static">',
            '      <attribute id="0" title="label" type="string"/>',
            '      <attribute id="1" title="category" type="string"/>',
            '      <attribute id="2" title="frequency" type="integer"/>',
            '      <attribute id="3" title="importance" type="float"/>',
            '      <attribute id="4" title="community" type="integer"/>',
            '    </attributes>',
            '    <nodes>',
        ]

        for i, node in enumerate(self._nodes.values()):
            lines.append(
                f'      <node id="{i}" label="{node.label}">'
                f'<attvalues>'
                f'<attvalue for="0" value="{node.label}"/>'
                f'<attvalue for="1" value="{node.category}"/>'
                f'<attvalue for="2" value="{node.frequency}"/>'
                f'<attvalue for="3" value="{node.importance:.3f}"/>'
                f'<attvalue for="4" value="{node.community}"/>'
                f'</attvalues></node>'
            )

        lines.append('    </nodes>')
        lines.append('    <edges>')

        for i, edge in enumerate(self._edges):
            source_idx = list(self._nodes.keys()).index(edge.source)
            target_idx = list(self._nodes.keys()).index(edge.target)
            lines.append(
                f'      <edge id="{i}" source="{source_idx}" '
                f'target="{target_idx}" weight="{edge.weight:.3f}"/>'
            )

        lines.extend(['    </edges>', '  </graph>', '</gexf>'])

        gexf = "\n".join(lines)

        if filepath:
            with open(filepath, "w") as f:
                f.write(gexf)

        return gexf

    # ──────────────────────────────────────────
    #  STATS
    # ──────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def density(self) -> float:
        """Graph density (0 = no edges, 1 = fully connected)."""
        n = len(self._nodes)
        if n < 2:
            return 0.0
        max_edges = n * (n - 1) / 2
        return len(self._edges) / max_edges if max_edges > 0 else 0.0

    def summary(self) -> dict:
        """Graph summary statistics."""
        degrees = [n.degree for n in self._nodes.values()]
        importances = [n.importance for n in self._nodes.values()]

        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "density": round(self.density, 4),
            "avg_degree": round(sum(degrees) / len(degrees), 1) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "avg_importance": round(sum(importances) / len(importances), 3) if importances else 0,
            "communities": len(self.get_communities()),
            "categories": dict(defaultdict(
                int,
                {n.category: sum(1 for nn in self._nodes.values() if nn.category == n.category)
                 for n in self._nodes.values()}
            )),
        }

    def print_summary(self):
        """Pretty-print graph summary."""
        s = self.summary()

        print("\n" + "=" * 60)
        print("  🕸️  KNOWLEDGE GRAPH SUMMARY")
        print("=" * 60)

        print(f"\n📊 Structure:")
        print(f"   Nodes:       {s['nodes']}")
        print(f"   Edges:       {s['edges']}")
        print(f"   Density:     {s['density']:.4f}")
        print(f"   Avg degree:  {s['avg_degree']}")
        print(f"   Max degree:  {s['max_degree']}")
        print(f"   Communities: {s['communities']}")

        print(f"\n🏆 Top Nodes (by importance):")
        for node in self.get_important_nodes(5):
            print(f"   • {node.label} "
                  f"(importance: {node.importance:.3f}, "
                  f"degree: {node.degree}, "
                  f"pagerank: {node.pagerank:.4f})")

        communities = self.get_community_info()
        if communities:
            print(f"\n📦 Communities:")
            for comm in communities[:5]:
                members = ", ".join(comm["members"][:5])
                print(f"   [{comm['id']}] {comm['size']} nodes "
                      f"(avg importance: {comm['avg_importance']:.3f}, "
                      f"dominant: {comm['dominant_category']})")
                print(f"       {members}")

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
    print("  KNOWLEDGE GRAPH — Quick Test")
    print("=" * 60 + "\n")

    graph = KnowledgeGraph()
    graph.build_from_markdown()
    graph.print_summary()

    # ── Path finding ──
    print("=" * 60)
    print("  PATH FINDING")
    print("=" * 60 + "\n")

    paths_to_test = [
        ("embedding", "search"),
        ("memory", "system"),
        ("vector", "database"),
    ]

    for source, target in paths_to_test:
        path = graph.find_path(source, target)
        if path:
            print(f"  {source} → {target}:")
            print(f"    Path: {' → '.join(path.nodes)}")
            print(f"    Hops: {path.length}, Weight: {path.total_weight:.3f}")
            if path.edges:
                for e in path.edges:
                    print(f"    [{e.source} -({e.weight:.0%})-> {e.target}]")
        else:
            print(f"  {source} → {target}: No path found")
        print()

    # ── Export test ──
    print("=" * 60)
    print("  EXPORT TEST")
    print("=" * 60 + "\n")

    import tempfile
    json_path = os.path.join(GRAPH_DIR, "knowledge_graph.json")
    data = graph.export_json(json_path)
    print(f"  JSON exported: {json_path}")
    print(f"  Nodes: {len(data['nodes'])}, Links: {len(data['links'])}")

    dot = graph.export_dot()
    dot_preview = dot[:200] + "..." if len(dot) > 200 else dot
    print(f"\n  DOT preview:\n  {dot_preview}")
