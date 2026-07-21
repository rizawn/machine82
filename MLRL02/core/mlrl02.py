"""
MLRL02 — Unified System Orchestrator

Single entry point for the entire MLRL02 adaptive AI system.

This module provides:
    - One import to access all subsystems
    - Clean facade pattern over complex internal architecture
    - Initialization, health checks, and system-wide coordination
    - Unified CLI commands: chat, analyze, ingest, search, graph, stats

Usage:
    from core.mlrl02 import MLRL02

    system = MLRL02()
    system.boot()

    # Chat with AI
    response = system.chat("Apa itu embeddings?")

    # Analyze project
    report = system.analyze()

    # Search memory
    results = system.search("semantic search")

    # Get system stats
    print(system.status())
"""

import os
import sys
from typing import Optional

import chromadb

# Project root sys.path hack removed from module level (M-1)

# ── Layer 2: Memory ──
from core.memory.vector_store import VectorStore
from core.memory.memory_context import MemoryContext
from core.memory.memory_reinforcement import MemoryReinforcement
from core.memory.loader import Document, load_markdown

# ── Layer 3: Reasoning ──
from core.reasoning.query import RetrievalEngine
from core.reasoning.query_classifier import QueryClassifier
from core.reasoning.reasoning_engine import ReasoningEngine
from core.reasoning.reflection_engine import ReflectionEngine
from core.reasoning.prompt_builder import PromptBuilder
from core.reasoning.concept_linker import ConceptLinker, ConceptGraph
from core.reasoning.knowledge_graph import KnowledgeGraph

# ── Layer 4: Agents ──
from core.agents.chat_engine import ChatEngine
from core.agents.task_planner import TaskPlanner
from core.agents.agent_loop import AgentLoop
from core.agents.workspace_agent import WorkspaceAgent


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
MARKDOWN_DIR = os.path.join(WORKSPACE_DIR, "markdown")
EMBEDDINGS_DIR = os.path.join(WORKSPACE_DIR, "embeddings")


# ──────────────────────────────────────────────
#  SYSTEM ORCHESTRATOR
# ──────────────────────────────────────────────

class MLRL02:
    """
    Unified orchestrator for the MLRL02 adaptive AI system.

    Facade pattern: exposes clean high-level methods while
    managing all internal module wiring.

    Subsystems available:
    - memory:     Vector storage + context retrieval + reinforcement
    - reasoning:  Multi-step reasoning + reflection + prompts
    - knowledge:  Concept linking + knowledge graph
    - agents:     Chat + autonomous loop + workspace analysis
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        model: str = "llama3",
        verbose: bool = True,
    ):
        self.workspace_dir = workspace_dir or WORKSPACE_DIR
        self.markdown_dir = os.path.join(self.workspace_dir, "markdown")
        self.embeddings_dir = os.path.join(self.workspace_dir, "embeddings")
        self.model = model
        self.verbose = verbose

        # Initialized on boot()
        self._booted = False

        # Layer 2: Memory
        self.vector_store: Optional[VectorStore] = None
        self.memory: Optional[MemoryContext] = None
        self.reinforcement: Optional[MemoryReinforcement] = None
        self.retrieval: Optional[RetrievalEngine] = None

        # Layer 3: Reasoning
        self.prompt_builder: Optional[PromptBuilder] = None
        self.classifier: Optional[QueryClassifier] = None
        self.reasoning: Optional[ReasoningEngine] = None
        self.reflection: Optional[ReflectionEngine] = None

        # Knowledge
        self.concept_linker: Optional[ConceptLinker] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None

        # Layer 4: Agents
        self.chat: Optional[ChatEngine] = None
        self.planner: Optional[TaskPlanner] = None
        self.agent_loop: Optional[AgentLoop] = None
        self.workspace_agent: Optional[WorkspaceAgent] = None

    # ──────────────────────────────────────────
    #  BOOT
    # ──────────────────────────────────────────

    def boot(self, verbose: bool = True):
        """
        Initialize all subsystems.

        This is lazy — nothing loads until boot() is called.

        Args:
            verbose: Print initialization progress
        """
        if self._booted:
            if verbose:
                print("[MLRL02] Already booted.")
            return

        if verbose:
            print("=" * 60)
            print("  🧠 MLRL02 — Booting Adaptive AI System")
            print("=" * 60 + "\n")

        # Layer 2: Memory
        if verbose:
            print("[1/4] Initializing memory subsystem...")

        # Create a single shared ChromaDB client for all modules
        os.makedirs(self.embeddings_dir, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=self.embeddings_dir)

        self.vector_store = VectorStore(
            persist_dir=self.embeddings_dir,
            client=self._chroma_client,
        )
        self.memory = MemoryContext(
            persist_dir=self.embeddings_dir,
            client=self._chroma_client,
        )
        self.reinforcement = MemoryReinforcement(
            memory=self.memory,
        )
        self.retrieval = RetrievalEngine(
            vector_store=self.vector_store,
        )

        # Layer 3: Reasoning
        if verbose:
            print("[2/4] Initializing reasoning subsystem...")
        self.prompt_builder = PromptBuilder(model=self.model)
        self.classifier = QueryClassifier(verbose=self.verbose)
        self.reasoning = ReasoningEngine(
            memory=self.memory,
            prompt_builder=self.prompt_builder,
        )
        self.reflection = ReflectionEngine()

        # Knowledge
        if verbose:
            print("[3/4] Initializing knowledge subsystem...")
        self.concept_linker = ConceptLinker()
        self.knowledge_graph = KnowledgeGraph()

        # Layer 4: Agents
        if verbose:
            print("[4/4] Initializing agent subsystem...")
        self.chat = ChatEngine(
            model=self.model,
            client=self._chroma_client,
        )
        self.planner = TaskPlanner()
        self.agent_loop = AgentLoop(
            memory=self.memory,
            reasoning=self.reasoning,
            reflection=self.reflection,
            planner=self.planner,
            chat=self.chat,
        )
        self.workspace_agent = WorkspaceAgent(
            project_path=BASE_DIR,
        )

        self._booted = True

        if verbose:
            print("\n[MLRL02] ✅ All subsystems ready.\n")

    # ──────────────────────────────────────────
    #  HIGH-LEVEL OPERATIONS
    # ──────────────────────────────────────────

    def route_query(self, question: str) -> str:
        """
        Intelligently route a query to the appropriate path.
        
        If the query is project-related, it routes to the reasoning engine (memory).
        Otherwise, it routes to the chat engine (general).

        Returns:
            AI response string
        """
        self._ensure_booted()
        
        category = self.classifier.classify(question)
        
        if category == "memory":
            if self.verbose:
                print(f"[MLRL02] 🧠 Routing to memory: '{question}'")
            # Use reasoning engine with LLM generation for better answers
            result = self.reasoning.reason(question, use_llm=True)
            return result.answer
        else:
            if self.verbose:
                print(f"[MLRL02] 🌐 Routing to general: '{question}'")
            return self.chat.chat(question, use_memory=False)

    def chat_with_ai(self, question: str, use_agent: bool = False, external_context: str = None) -> str:
        """
        Chat with the AI system.

        Args:
            question:    User's question
            use_agent:   If True, use full AgentLoop (slower but more thorough).
                         If False, use direct ChatEngine (faster).
            external_context: Optional additional context.

        Returns:
            AI response string
        """
        self._ensure_booted()

        if use_agent:
            result = self.agent_loop.execute(question)
            return result.final_answer
        else:
            return self.chat.chat(question, external_context=external_context)

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        use_reinforcement: bool = True,
    ) -> list:
        """
        Search semantic memory.

        Args:
            query:             Search query
            top_k:             Number of results
            use_reinforcement: Apply usage-based scoring boost

        Returns:
            List of ContextItem (if reinforcement) or dict (if raw)
        """
        self._ensure_booted()

        if use_reinforcement:
            return self.reinforcement.retrieve(query, top_k=top_k)
        else:
            return self.memory.retrieve(query, top_k=top_k)

    def reason_about(self, question: str) -> dict:
        """
        Deep reasoning about a question.

        Returns:
            Dict with answer, reasoning steps, and confidence
        """
        self._ensure_booted()

        result = self.reasoning.reason(question, use_llm=False)

        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "intent": result.analysis.intent,
            "key_terms": result.analysis.key_terms,
            "steps": len(result.reasoning_steps),
            "contexts_used": len(result.contexts_used),
            "concept_links": len(result.concept_links),
        }

    def ingest(self, markdown_dir: Optional[str] = None) -> dict:
        """
        Ingest markdown files into memory.

        Args:
            markdown_dir: Override default markdown directory

        Returns:
            Dict with ingestion stats
        """
        self._ensure_booted()

        md_dir = markdown_dir or self.markdown_dir

        # Ingest into vector store
        self.vector_store.ingest_markdown(md_dir)

        # Build knowledge graph
        self.concept_linker.scan_directory(md_dir)
        concept_graph = self.concept_linker.build_graph()
        self.knowledge_graph.build_from_concept_graph(concept_graph)

        return {
            "vector_count": self.vector_store.count(),
            "concept_count": len(concept_graph.concepts),
            "link_count": len(concept_graph.links),
            "graph_nodes": self.knowledge_graph.node_count,
        }

    def analyze_project(self, project_path: Optional[str] = None) -> dict:
        """
        Analyze a project directory.

        Args:
            project_path: Override default project path

        Returns:
            Dict with analysis summary
        """
        self._ensure_booted()

        path = project_path or BASE_DIR
        self.workspace_agent = WorkspaceAgent(project_path=path)
        self.workspace_agent.analyze()

        structure = self.workspace_agent.get_structure()
        knowledge = self.workspace_agent.get_knowledge()
        suggestions = self.workspace_agent.get_suggestions()

        return {
            "files": structure.total_files if structure else 0,
            "dirs": structure.total_dirs if structure else 0,
            "knowledge_docs": knowledge.total_documents if knowledge else 0,
            "concepts": len(self.knowledge_graph._nodes) if self.knowledge_graph else 0,
            "suggestions": len(suggestions),
            "improvements": [
                {
                    "title": s.title,
                    "priority": s.priority,
                    "action": s.action,
                }
                for s in suggestions
            ],
        }

    def export_knowledge_graph(self, filepath: Optional[str] = None) -> str:
        """
        Export knowledge graph as JSON.

        Args:
            filepath: Override default export path

        Returns:
            Path to exported file
        """
        self._ensure_booted()

        out_path = filepath or os.path.join(
            self.workspace_dir, "knowledge_graph.json"
        )
        self.knowledge_graph.export_json(out_path)
        return out_path

    def status(self) -> dict:
        """
        Get system-wide status.

        Returns:
            Dict with all subsystem stats
        """
        self._ensure_booted()

        return {
            "booted": self._booted,
            "model": self.model,
            "memory": {
                "vector_count": self.vector_store.count() if self.vector_store else 0,
                "available": self.memory.is_available() if self.memory else False,
            },
            "classifier": "ready" if self.classifier else "not_initialized",
            "reinforcement": self.reinforcement.stats() if self.reinforcement else {},
            "knowledge": {
                "graph_nodes": self.knowledge_graph.node_count if self.knowledge_graph else 0,
                "graph_edges": self.knowledge_graph.edge_count if self.knowledge_graph else 0,
            },
            "chat_history_turns": len(self.chat.history) if self.chat else 0,
            "agent_iterations": (
                self.reasoning._total_reasoned
                if self.reasoning
                else 0
            ),
        }

    # ──────────────────────────────────────────
    #  INTERNAL
    # ──────────────────────────────────────────

    def _ensure_booted(self):
        """Ensure system is booted, auto-boot if not."""
        if not self._booted:
            self.boot(verbose=False)


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure project root is on path for standalone execution
    import os
    import sys
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    system = MLRL02()
    system.boot()

    print("=" * 60)
    print("  MLRL02 — System Status")
    print("=" * 60 + "\n")

    stats = system.status()
    for section, data in stats.items():
        print(f"  {section}:")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"    {k}: {v}")
        else:
            print(f"    {data}")
        print()

    print("=" * 60)
    print("  Quick Search Test")
    print("=" * 60 + "\n")

    results = system.search_memory("apa itu embeddings", top_k=3)
    for i, ctx in enumerate(results, 1):
        print(f"  [{i}] {ctx.source} (score: {ctx.score:.0%})")
        print(f"      {ctx.text[:80]}...")
        print()
