"""
WORKSPACE AGENT — Autonomous Project Analysis

Features:
    - Read project folders and files
    - Analyze markdown notes for knowledge extraction
    - Summarize project knowledge
    - Detect project structure and architecture
    - Generate documentation from analysis
    - Suggest improvements based on patterns
    - Track project evolution over time

Architecture:
    This is the top-level autonomous agent for MLRL02.
    It orchestrates all other modules to provide a complete
    project understanding:

        Scan → Analyze → Summarize → Document → Suggest

    Pipeline:
    1. SCAN:       Read project directory tree
    2. PARSE:      Extract content from files
    3. CONCEPTS:   Build concept map (ConceptLinker)
    4. GRAPH:      Create knowledge graph (KnowledgeGraph)
    5. SUMMARIZE:  Generate project summary
    6. SUGGEST:    Recommend improvements

Usage:
    agent = WorkspaceAgent("path/to/project")
    agent.analyze()
    agent.print_report()
    suggestions = agent.get_suggestions()
"""

import os
import sys
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from collections import Counter, defaultdict

# Calculate project root for configuration constants (M-1 cleanup)
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from core.memory.loader import Document, load_markdown
from core.reasoning.concept_linker import ConceptLinker, ConceptGraph
from core.reasoning.knowledge_graph import KnowledgeGraph


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

# File types the agent analyzes
ANALYZABLE_EXTENSIONS = {
    ".md", ".txt", ".py", ".json", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".rst",
}

# Directories to skip
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".cache", ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".eggs",
}

# Documentation output
DOCS_DIR = os.path.join(_project_root, "workspace", "docs")


# ──────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class FileInfo:
    """
    Metadata about a single file in the project.

    Attributes:
        path:        Relative path from project root
        extension:   File extension
        size_bytes:  File size
        line_count:  Number of lines
        content_hash: SHA256 hash for change detection
    """
    path: str
    extension: str
    size_bytes: int = 0
    line_count: int = 0
    content_hash: str = ""
    last_modified: float = 0.0


@dataclass
class ProjectStructure:
    """
    Detected project structure.

    Attributes:
        root:         Project root directory
        total_files:  Total files found
        total_dirs:   Total directories
        file_types:   Count by extension
        dirs:         Directory names found
        top_level:    Items at root level
        has_tests:    Whether test directory exists
        has_docs:     Whether docs directory exists
        has_config:   Whether config files exist
    """
    root: str
    total_files: int = 0
    total_dirs: int = 0
    file_types: dict[str, int] = field(default_factory=dict)
    dirs: list[str] = field(default_factory=list)
    top_level: list[str] = field(default_factory=list)
    has_tests: bool = False
    has_docs: bool = False
    has_config: bool = False

    @property
    def is_python_project(self) -> bool:
        return self.file_types.get(".py", 0) > 0

    @property
    def is_documentation_project(self) -> bool:
        return self.file_types.get(".md", 0) > 0


@dataclass
class KnowledgeSummary:
    """
    Summarized knowledge from project analysis.

    Attributes:
        total_documents:  Number of knowledge documents
        key_concepts:     Top concepts found
        topics:           Detected topic areas
        concept_density:  How rich the knowledge base is
        coverage_score:   How comprehensive the docs are
    """
    total_documents: int = 0
    key_concepts: list[tuple[str, int]] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    concept_density: float = 0.0
    coverage_score: float = 0.0


@dataclass
class Improvement:
    """
    A suggested improvement for the project.

    Attributes:
        category:     Type of improvement
        priority:     How important
        title:        Short description
        description:  Detailed explanation
        action:       Specific action to take
    """
    category: str
    priority: str  # high, medium, low
    title: str
    description: str
    action: str


@dataclass
class AnalysisReport:
    """
    Complete analysis report.

    Attributes:
        project:       ProjectStructure
        knowledge:     KnowledgeSummary
        concepts:      ConceptGraph
        graph:         KnowledgeGraph
        improvements:  Suggested improvements
        timestamp:     When analysis was run
    """
    project: ProjectStructure
    knowledge: KnowledgeSummary
    concepts: ConceptGraph
    graph: KnowledgeGraph
    improvements: list[Improvement] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ──────────────────────────────────────────────
#  PROJECT SCANNER
# ──────────────────────────────────────────────

class ProjectScanner:
    """
    Scan and analyze project directory structure.
    """

    def scan(self, root_path: str) -> ProjectStructure:
        """
        Scan a project directory.

        Args:
            root_path: Path to project root

        Returns:
            ProjectStructure with analysis
        """
        structure = ProjectStructure(root=root_path)
        file_types: Counter = Counter()

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Skip ignored directories
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            rel_dir = os.path.relpath(dirpath, root_path)
            if rel_dir != ".":
                structure.dirs.append(rel_dir)

            # Count files
            for filename in filenames:
                if filename.startswith("."):
                    continue

                ext = os.path.splitext(filename)[1].lower()
                file_types[ext] += 1
                structure.total_files += 1

                # Track top-level items
                if dirpath == root_path:
                    structure.top_level.append(filename)

        structure.total_dirs = len(structure.dirs)
        structure.file_types = dict(file_types.most_common())

        # Detect features
        structure.has_tests = any(
            "test" in d.lower() for d in structure.dirs
        )
        structure.has_docs = any(
            "doc" in d.lower() for d in structure.dirs
        )
        structure.has_config = any(
            ext in (".toml", ".cfg", ".ini", ".yaml", ".yml")
            for ext in file_types
        )

        return structure


# ──────────────────────────────────────────────
#  KNOWLEDGE ANALYZER
# ──────────────────────────────────────────────

class KnowledgeAnalyzer:
    """
    Analyze markdown documents for knowledge content.
    """

    def analyze(self, documents: list[Document]) -> KnowledgeSummary:
        """
        Analyze loaded documents.

        Args:
            documents: List of Document from loader

        Returns:
            KnowledgeSummary
        """
        if not documents:
            return KnowledgeSummary()

        # Extract concepts
        linker = ConceptLinker()
        linker.scan_documents(documents)
        concept_graph = linker.build_graph()

        # Top concepts by frequency
        concept_freq = Counter()
        for name, concept in concept_graph.concepts.items():
            concept_freq[name] = concept.frequency

        key_concepts = concept_freq.most_common(15)

        # Detect topics from categories
        categories = Counter()
        for concept in concept_graph.concepts.values():
            if concept.category != "unknown":
                categories[concept.category] += 1

        topics = [cat for cat, _ in categories.most_common(5)]

        # Concept density: concepts per document
        concept_density = (
            len(concept_graph.concepts) / len(documents)
            if documents else 0
        )

        # Coverage score: how well topics are covered
        total_docs = len(documents)
        docs_with_content = sum(
            1 for d in documents
            if len(d.page_content.split()) > 50
        )
        coverage_score = docs_with_content / total_docs if total_docs > 0 else 0

        return KnowledgeSummary(
            total_documents=total_docs,
            key_concepts=key_concepts,
            topics=topics,
            concept_density=round(concept_density, 1),
            coverage_score=round(coverage_score, 2),
        )


# ──────────────────────────────────────────────
#  IMPROVEMENT SUGGESTER
# ──────────────────────────────────────────────

class ImprovementSuggester:
    """
    Generate improvement suggestions based on project analysis.
    """

    def suggest(
        self,
        structure: ProjectStructure,
        knowledge: KnowledgeSummary,
        graph: KnowledgeGraph,
    ) -> list[Improvement]:
        """
        Generate suggestions based on analysis.

        Args:
            structure: Project structure
            knowledge: Knowledge summary
            graph:     Knowledge graph

        Returns:
            List of Improvement objects
        """
        improvements = []

        # ── Structure-based suggestions ──
        if not structure.has_tests and structure.is_python_project:
            improvements.append(Improvement(
                category="structure",
                priority="high",
                title="Add test directory",
                description="Python projects benefit from automated tests.",
                action="Create a 'tests/' directory with pytest test files.",
            ))

        if not structure.has_docs:
            improvements.append(Improvement(
                category="documentation",
                priority="medium",
                title="Create docs directory",
                description="No documentation directory found.",
                action="Create a 'docs/' directory for project documentation.",
            ))

        if structure.total_files > 20 and not structure.has_config:
            improvements.append(Improvement(
                category="configuration",
                priority="medium",
                title="Add project configuration",
                description="Larger projects need centralized configuration.",
                action="Add pyproject.toml or setup.cfg for project metadata.",
            ))

        # ── Knowledge-based suggestions ──
        if knowledge.total_documents < 3:
            improvements.append(Improvement(
                category="knowledge",
                priority="high",
                title="Expand knowledge base",
                description="Only {} knowledge document(s). More docs improve AI reasoning.".format(
                    knowledge.total_documents
                ),
                action="Add more markdown files with structured documentation.",
            ))

        if knowledge.concept_density < 5:
            improvements.append(Improvement(
                category="knowledge",
                priority="medium",
                title="Increase concept density",
                description="Low concept density ({:.1f} concepts/doc). "
                           "Documents may be too brief.".format(
                    knowledge.concept_density
                ),
                action="Enrich documents with more detailed explanations and examples.",
            ))

        if knowledge.coverage_score < 0.5:
            improvements.append(Improvement(
                category="knowledge",
                priority="medium",
                title="Improve document coverage",
                description="{:.0%} of documents have substantial content.".format(
                    knowledge.coverage_score
                ),
                action="Expand thin documents or merge empty ones.",
            ))

        # ── Graph-based suggestions ──
        if graph.node_count > 0:
            important = graph.get_important_nodes(1)
            if important and important[0].degree < 3:
                improvements.append(Improvement(
                    category="knowledge-graph",
                    priority="low",
                    title="Strengthen concept connections",
                    description="Top concept '{}' has only {} connections.".format(
                        important[0].label, important[0].degree
                    ),
                    action="Add more cross-references between concepts in documentation.",
                ))

            disconnected = sum(
                1 for n in graph._nodes.values() if n.degree == 0
            )
            if disconnected > 0:
                improvements.append(Improvement(
                    category="knowledge-graph",
                    priority="low",
                    title="Connect orphan concepts",
                    description="{} concept(s) have no connections in the knowledge graph.".format(
                        disconnected
                    ),
                    action="Link orphan concepts to related topics in your markdown files.",
                ))

        # ── General suggestions ──
        if structure.file_types.get(".md", 0) > 0:
            improvements.append(Improvement(
                category="knowledge",
                priority="low",
                title="Add frontmatter to markdown files",
                description="Frontmatter helps the AI categorize and prioritize knowledge.",
                action="Add YAML frontmatter (--- ... ---) to markdown files.",
            ))

        return improvements


# ──────────────────────────────────────────────
#  DOCUMENTATION GENERATOR
# ──────────────────────────────────────────────

class DocGenerator:
    """
    Generate documentation from analysis results.
    """

    def generate_report(
        self,
        report: AnalysisReport,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Generate a markdown analysis report.

        Args:
            report:     Analysis report
            output_dir: Where to save (optional)

        Returns:
            Markdown report content
        """
        lines = [
            "# MLRL02 — Project Analysis Report",
            "",
            f"Generated: {report.timestamp}",
            f"Project: {report.project.root}",
            "",
            "---",
            "",
            "## Project Structure",
            "",
            f"- **Total files:** {report.project.total_files}",
            f"- **Total directories:** {report.project.total_dirs}",
            f"- **Python project:** {'Yes' if report.project.is_python_project else 'No'}",
            f"- **Has tests:** {'Yes' if report.project.has_tests else 'No'}",
            f"- **Has docs:** {'Yes' if report.project.has_docs else 'No'}",
            "",
            "### File Types",
            "",
        ]

        for ext, count in report.project.file_types.items():
            lines.append(f"- `{ext or '(no ext)'}`: {count}")

        lines.extend([
            "",
            "## Knowledge Base",
            "",
            f"- **Documents:** {report.knowledge.total_documents}",
            f"- **Concepts:** {len(report.concepts.concepts)}",
            f"- **Concept links:** {len(report.concepts.links)}",
            f"- **Concept density:** {report.knowledge.concept_density:.1f} concepts/doc",
            f"- **Coverage score:** {report.knowledge.coverage_score:.0%}",
            "",
            "### Key Concepts",
            "",
        ])

        for concept, freq in report.knowledge.key_concepts[:10]:
            display = concept.replace("_", " ").title()
            lines.append(f"- **{display}** ({freq} occurrences)")

        if report.knowledge.topics:
            lines.extend([
                "",
                "### Topics",
                "",
            ])
            for topic in report.knowledge.topics:
                lines.append(f"- {topic.replace('_', ' ').title()}")

        lines.extend([
            "",
            "## Knowledge Graph",
            "",
            f"- **Nodes:** {report.graph.node_count}",
            f"- **Edges:** {report.graph.edge_count}",
            f"- **Density:** {report.graph.density:.4f}",
            f"- **Communities:** {len(report.graph.get_communities())}",
            "",
        ])

        important = report.graph.get_important_nodes(5)
        if important:
            lines.extend(["### Most Important Concepts", ""])
            for node in important:
                lines.append(
                    f"- **{node.label}** "
                    f"(importance: {node.importance:.3f}, "
                    f"degree: {node.degree})"
                )

        if report.improvements:
            lines.extend([
                "",
                "## Suggested Improvements",
                "",
            ])
            for i, imp in enumerate(report.improvements, 1):
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(imp.priority, "•")
                lines.extend([
                    f"### {i}. {priority_icon} {imp.title}",
                    "",
                    f"**Category:** {imp.category}",
                    f"**Priority:** {imp.priority}",
                    "",
                    imp.description,
                    "",
                    f"**Action:** {imp.action}",
                    "",
                ])

        lines.extend([
            "",
            "---",
            "",
            "*Report generated by MLRL02 Workspace Agent*",
        ])

        content = "\n".join(lines)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, "analysis_report.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def generate_evolution_log(
        self,
        report: AnalysisReport,
        log_dir: Optional[str] = None,
    ) -> str:
        """
        Generate an evolution log entry for tracking changes.

        Args:
            report:  Analysis report
            log_dir: Where to save evolution log (optional)

        Returns:
            Log entry content
        """
        s = report.graph.summary()

        entry = {
            "timestamp": report.timestamp,
            "project_files": report.project.total_files,
            "project_dirs": report.project.total_dirs,
            "knowledge_docs": report.knowledge.total_documents,
            "graph_nodes": s["nodes"],
            "graph_edges": s["edges"],
            "graph_density": s["density"],
            "improvements_count": len(report.improvements),
        }

        content = json.dumps(entry, indent=2)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            filepath = os.path.join(log_dir, "evolution_log.json")

            # Append to existing log or create new
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        existing = json.load(f)
                        if isinstance(existing, list):
                            existing.append(entry)
                        else:
                            existing = [existing, entry]
                    except json.JSONDecodeError:
                        existing = [entry]
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump([entry], f, indent=2)

        return content


# ──────────────────────────────────────────────
#  WORKSPACE AGENT (Main Class)
# ──────────────────────────────────────────────

class WorkspaceAgent:
    """
    Autonomous workspace analysis agent.

    Orchestrates project scanning, knowledge analysis,
    concept linking, graph building, and documentation.

    Usage:
        agent = WorkspaceAgent("path/to/project")
        agent.analyze()
        agent.print_report()
        agent.generate_docs()
    """

    def __init__(self, project_path: Optional[str] = None):
        self.project_path = project_path or _project_root
        self._scanner = ProjectScanner()
        self._analyzer = KnowledgeAnalyzer()
        self._suggester = ImprovementSuggester()
        self._doc_gen = DocGenerator()
        self._concept_linker = ConceptLinker()

        # Results
        self._structure: Optional[ProjectStructure] = None
        self._knowledge: Optional[KnowledgeSummary] = None
        self._concept_graph: Optional[ConceptGraph] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._improvements: list[Improvement] = []
        self._report: Optional[AnalysisReport] = None
        self._documents: list[Document] = []

    def analyze(self):
        """Run the full analysis pipeline."""
        # Step 1: Scan structure
        self._structure = self._scanner.scan(self.project_path)

        # Step 2: Load and analyze markdown documents
        md_dir = os.path.join(self.project_path, "workspace", "markdown")
        if os.path.exists(md_dir):
            self._documents = load_markdown(md_dir)
            self._knowledge = self._analyzer.analyze(self._documents)
        else:
            self._documents = []
            self._knowledge = KnowledgeSummary()

        # Step 3: Build concept graph
        if self._documents:
            self._concept_linker.scan_documents(self._documents)
            self._concept_graph = self._concept_linker.build_graph()
        else:
            self._concept_graph = ConceptGraph()

        # Step 4: Build knowledge graph
        self._knowledge_graph = KnowledgeGraph()
        if self._concept_graph and self._concept_graph.concepts:
            self._knowledge_graph.build_from_concept_graph(self._concept_graph)

        # Step 5: Generate suggestions
        self._improvements = self._suggester.suggest(
            self._structure,
            self._knowledge,
            self._knowledge_graph,
        )

        # Step 6: Build report
        self._report = AnalysisReport(
            project=self._structure,
            knowledge=self._knowledge,
            concepts=self._concept_graph,
            graph=self._knowledge_graph,
            improvements=self._improvements,
        )

    def get_structure(self) -> Optional[ProjectStructure]:
        return self._structure

    def get_knowledge(self) -> Optional[KnowledgeSummary]:
        return self._knowledge

    def get_suggestions(self) -> list[Improvement]:
        return self._improvements

    def get_report(self) -> Optional[AnalysisReport]:
        return self._report

    def generate_docs(
        self,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Generate documentation from analysis.

        Args:
            output_dir: Where to save docs

        Returns:
            Markdown content
        """
        if not self._report:
            raise RuntimeError("Run analyze() first.")

        out_dir = output_dir or DOCS_DIR
        return self._doc_gen.generate_report(self._report, out_dir)

    def log_evolution(self, log_dir: Optional[str] = None) -> str:
        """
        Log this analysis for evolution tracking.

        Args:
            log_dir: Where to save log

        Returns:
            JSON log entry
        """
        if not self._report:
            raise RuntimeError("Run analyze() first.")

        log_dir = log_dir or os.path.join(_project_root, "workspace")
        return self._doc_gen.generate_evolution_log(self._report, log_dir)

    def print_report(self):
        """Pretty-print the analysis report."""
        if not self._report:
            print("No analysis available. Run analyze() first.")
            return

        r = self._report

        print("\n" + "=" * 60)
        print("  🤖 WORKSPACE AGENT — Analysis Report")
        print("=" * 60)

        # Structure
        print(f"\n📁 Project: {r.project.root}")
        print(f"   Files: {r.project.total_files} | Dirs: {r.project.total_dirs}")
        print(f"   Python: {'✅' if r.project.is_python_project else '❌'}  "
              f"Tests: {'✅' if r.project.has_tests else '❌'}  "
              f"Docs: {'✅' if r.project.has_docs else '❌'}")

        # File types
        if r.project.file_types:
            print(f"\n   File types:")
            for ext, count in list(r.project.file_types.items())[:5]:
                print(f"     {ext or '(no ext)'}: {count}")

        # Knowledge
        print(f"\n📚 Knowledge Base:")
        print(f"   Documents: {r.knowledge.total_documents}")
        print(f"   Concepts:  {len(r.concepts.concepts)}")
        print(f"   Links:     {len(r.concepts.links)}")
        print(f"   Density:   {r.knowledge.concept_density:.1f} concepts/doc")
        print(f"   Coverage:  {r.knowledge.coverage_score:.0%}")

        # Key concepts
        if r.knowledge.key_concepts:
            print(f"\n🏷️  Key Concepts:")
            for concept, freq in r.knowledge.key_concepts[:5]:
                display = concept.replace("_", " ").title()
                print(f"     • {display} ({freq}x)")

        # Graph
        s = r.graph.summary()
        print(f"\n🕸️  Knowledge Graph:")
        print(f"   Nodes: {s['nodes']} | Edges: {s['edges']} | Density: {s['density']:.4f}")

        # Improvements
        if r.improvements:
            print(f"\n💡 Suggestions ({len(r.improvements)}):")
            for imp in r.improvements:
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(imp.priority, "•")
                print(f"   {icon} [{imp.priority}] {imp.title}")
                print(f"       {imp.action}")

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
    print("  WORKSPACE AGENT — Quick Test")
    print("=" * 60 + "\n")

    # Analyze the current project
    agent = WorkspaceAgent(_project_root)
    agent.analyze()
    agent.print_report()

    # Generate docs
    print("=" * 60)
    print("  GENERATING DOCUMENTATION")
    print("=" * 60 + "\n")

    report_content = agent.generate_docs()
    print(f"  Report generated: {len(report_content)} characters")

    # Log evolution
    log_content = agent.log_evolution()
    print(f"  Evolution log: {log_content[:100]}...")
