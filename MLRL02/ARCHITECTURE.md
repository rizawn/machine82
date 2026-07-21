# MLRL02 — System Architecture

> Adaptive AI Memory & Reasoning Workspace
'''
author : Riza Wahyu Nugraha
'''

```
Version: 2.0
Date: 2026-05-24
Status: Production Design
```

---

## 1. System Overview

MLRL02 is a **local-first, modular AI system** that combines:
- **Persistent semantic memory** (ChromaDB + embeddings)
- **Multi-step reasoning** (analysis → retrieval → synthesis → reflection)
- **Autonomous agent workflows** (goal → plan → act → reflect → iterate)
- **Knowledge graph** (concept extraction, linking, community detection)
- **Adaptive reinforcement** (usage-based memory scoring + decay)

All powered by **local models via Ollama** — no cloud API required.

---

## 2. Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 4: ORCHESTRATION (Agents)                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ AgentLoop    │  │ Workspace    │  │ TaskPlanner   │ │
│  │ (autonomous) │  │ Agent        │  │ (goal decomp) │ │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘ │
│         │                 │                   │         │
│  ┌──────┴─────────────────┴───────────────────┴──────┐ │
│  │  MLRL02 Orchestrator (unified entry point)        │ │
│  └─────────────────────┬─────────────────────────────┘ │
└────────────────────────┼────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────┐
│  LAYER 3: REASONING & COGNITION                         │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ Reasoning    │  │ Reflection   │  │ PromptBuilder │ │
│  │ Engine       │  │ Engine       │  │ (templating)  │ │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘ │
│         │                 │                   │         │
│  ┌──────┴─────────────────┴───────────────────┴──────┐ │
│  │  ConceptLinker → KnowledgeGraph                    │ │
│  │  (concept extraction, linking, graph analysis)     │ │
│  └─────────────────────┬─────────────────────────────┘ │
└────────────────────────┼────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────┐
│  LAYER 2: MEMORY & RETRIEVAL                            │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ VectorStore  │  │ MemoryCtx    │  │ MemoryReinf.  │ │
│  │ (ChromaDB +  │  │ (retrieve +  │  │ (usage track  │ │
│  │  embedding)  │  │  dedup +     │  │  + decay)     │ │
│  │              │  │  rank)       │  │               │ │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘ │
│         │                 │                   │         │
│  ┌──────┴─────────────────┴───────────────────┴──────┐ │
│  │  Loader (markdown → Document)                      │ │
│  └─────────────────────┬─────────────────────────────┘ │
└────────────────────────┼────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────┐
│  LAYER 1: EXTERNAL SERVICES                             │
│                                                         │
│  ┌───────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │ ChromaDB  │  │ Ollama (LLM)  │  │ sentence-      │  │
│  │ (vectors) │  │ llama3/etc.   │  │ transformers   │  │
│  └───────────┘  └───────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Key design rule:** Each layer only depends on layers below it.
Layer 4 → Layer 3 → Layer 2 → Layer 1 (never upward or sideways).

---

## 3. Module Interaction Flow

### 3.1 Knowledge Ingestion Pipeline

```
workspace/markdown/*.md
       │
       ▼
  ┌────────────┐
  │  loader.py │  Document objects with page_content + metadata
  └─────┬──────┘
        │
   ┌────┴─────┐
   │          │
   ▼          ▼
┌──────────┐  ┌──────────────┐
│ Vector   │  │ ConceptLinker│
│ Store    │  │              │
│(embedding│  │ (n-grams,    │
│ → Chroma)│  │  links)      │
└─────┬────┘  └──────┬───────┘
      │              │
      ▼              ▼
┌───────────┐  ┌──────────────┐
│workspace/ │  │KnowledgeGraph│
│embeddings/│  │(nodes, edges,│
└───────────┘  │ centrality)  │
               └──────┬───────┘
                      │
                      ▼
               workspace/
               knowledge_graph.json
```

### 3.2 Autonomous Agent Cycle

```
                    ┌─────────────┐
                    │   GOAL      │  User question or autonomous objective
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   PLAN      │  TaskPlanner: decompose into steps
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  RETRIEVE   │  MemoryContext: fetch relevant memory
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   REASON    │  ReasoningEngine: analyze, chain, synthesize
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    ACT      │  Generate answer (context-based or LLM)
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │       REFLECT           │  ReflectionEngine: score quality
              │  ┌───────────────────┐  │
              │  │ Score ≥ 0.65?    │  │
              │  └──────┬──────┬────┘  │
              │     YES │      │ NO    │
              │         │      │       │
              │    ┌────▼──┐  ┌▼──────┐│
              │    │ACCEPT │  │REFINE ││
              │    │answer │  │context││
              │    └───────┘  └───┬───┘│
              │                   │    │
              │     ┌─────────────┘    │
              │     │ (iterate, max 3) │
              │     ▼                  │
              └────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   RESULT    │  AgentResult: answer + trace + confidence
                    └─────────────┘
```

### 3.3 Memory Reinforcement Loop

```
User Query
    │
    ▼
MemoryReinforcement.retrieve()
    │
    ├──→ MemoryContext.retrieve()  (base vector search)
    │         │
    │         ▼
    │    ┌─────────────┐
    │    │ Apply boost │  reinforced_score = base × (1 + usage_weight × decay)
    │    │ + decay     │
    │    └──────┬──────┘
    │           │
    ▼           ▼
Ranked Contexts → LLM generates answer
                      │
                      ▼
              record_usage(chunk_id, quality)
                      │
                      ▼
              ┌───────────────┐
              │ memory_stats  │  Persisted: access_count, avg_quality,
              │ .json         │  total_boost, last_access
              └───────────────┘
```

---

## 4. Current Module Map

| Layer | Module | Classes | Lines |
|-------|--------|---------|-------|
| **L4: Orchestration** | `agent_loop.py` | `AgentLoop`, `AgentResult`, `ToolRegistry` | 741 |
| | `workspace_agent.py` | `WorkspaceAgent`, `ProjectScanner`, `DocGenerator` | 853 |
| | `task_planner.py` | `TaskPlanner`, `GoalDecomposer`, `Plan` | 679 |
| **L3: Reasoning** | `reasoning_engine.py` | `ReasoningEngine`, `QuestionAnalyzer`, `MemoryExplorer` | 917 |
| | `reflection_engine.py` | `ReflectionEngine`, `RelevanceScorer`, `AccuracyScorer` | 952 |
| | `prompt_builder.py` | `PromptBuilder`, `LLM_PROFILES` | 494 |
| | `concept_linker.py` | `ConceptLinker`, `ConceptGraph`, `ConceptExtractor` | 836 |
| | `knowledge_graph.py` | `KnowledgeGraph`, `GraphNode`, `GraphEdge` | 847 |
| | `query_classifier.py` | `QueryClassifier` | 144 |
| | `constants.py` | `STOP_WORDS` | 32 |
| **L2: Memory** | `vector_store.py` | `VectorStore` | 271 |
| | `memory_context.py` | `MemoryContext`, `ContextItem` | 514 |
| | `memory_reinforcement.py` | `MemoryReinforcement`, `MemoryRecord` | 602 |
| | `loader.py` | `Document`, `load_markdown()`, `extract_frontmatter()` | 148 |
| **L4: Chat** | `chat_engine.py` | `ChatEngine` | 319 |
| **L2: Query** | `query.py` | `RetrievalEngine` | 191 |
| **Total** | **16 files** | **66 classes** | **~8,540 lines** |

---

## 5. Integration Issues & Fixes

### 5.1 Layering Violation
**Problem:** `reasoning_engine.py` (L3) imports `prompt_builder.py` from `agents/` (L4).
**Fix:** Move `prompt_builder.py` to `core/reasoning/` — it's a reasoning utility, not an agent.

### 5.2 Isolated ChatEngine
**Problem:** `ChatEngine` uses full LangChain stack but isn't wired into `AgentLoop`.
**Fix:** `AgentLoop._stage_act()` should delegate to `ChatEngine` when `use_llm=True`.

### 5.3 Missing Package Init
**Problem:** `core/agents/__init__.py` doesn't exist — inconsistent with other subpackages.
**Fix:** Create `__init__.py` with clean exports.

### 5.4 Dead Code in main.py
**Problem:** `main.py` imports unused `VectorStore` and only uses the old `RetrievalEngine` path.
**Fix:** Rewrite `main.py` as unified CLI that exposes all system capabilities.

---

## 6. Recommended Folder Structure (Target)

```
MLRL02/
├── main.py                          ← Unified CLI entry point
├── requirements.txt
├── ARCHITECTURE.md                  ← This document
│
├── core/
│   ├── __init__.py
│   ├── mlrl02.py                    ← Unified orchestrator (new)
│   │
│   ├── memory/
│   │   ├── __init__.py              ← Exports: VectorStore, MemoryContext, MemoryReinforcement
│   │   ├── loader.py                ← [moved from reasoning/] Markdown → Document
│   │   ├── vector_store.py          ← ChromaDB + embeddings
│   │   ├── memory_context.py        ← Context retrieval + ranking
│   │   └── memory_reinforcement.py  ← Usage-based adaptive scoring
│   │
│   ├── reasoning/
│   │   ├── __init__.py              ← Exports: all reasoning modules
│   │   ├── prompt_builder.py        ← [moved from agents/] Structured prompts
│   │   ├── query.py                 ← Semantic search (RetrievalEngine)
│   │   ├── reasoning_engine.py      ← Multi-step reasoning pipeline
│   │   ├── reflection_engine.py     ← Self-evaluation
│   │   ├── concept_linker.py        ← Concept extraction + linking
│   │   └── knowledge_graph.py       ← Cognitive graph analysis
│   │
│   └── agents/
│       ├── __init__.py              ← Exports: AgentLoop, WorkspaceAgent, TaskPlanner, ChatEngine
│       ├── chat_engine.py           ← LangChain + Ollama chat
│       ├── task_planner.py          ← Goal decomposition
│       ├── agent_loop.py            ← Autonomous agent cycle
│       └── workspace_agent.py       ← Project analysis
│
├── workspace/
│   ├── markdown/                    ← Knowledge base (source .md files)
│   ├── embeddings/                  ← ChromaDB persistent storage
│   ├── docs/                        ← Generated documentation
│   ├── memory_stats.json            ← Reinforcement usage data
│   ├── knowledge_graph.json         ← Exported graph
│   └── evolution_log.json           ← Project tracking log
│
└── tests/                           ← [future] Unit + integration tests
    ├── test_loader.py
    ├── test_memory_context.py
    ├── test_reasoning.py
    └── test_agent_loop.py
```

---

## 7. Recommended Improvements (Priority Order)

### High Priority
1. **Move `loader.py` to `core/memory/`** — It's an ingestion utility, not reasoning logic
2. **Move `prompt_builder.py` to `core/reasoning/`** — Fixes layering violation
3. **Integrate `ChatEngine` into `AgentLoop`** — Unify LLM paths
4. **Create `core/mlrl02.py` orchestrator** — Single import for the whole system
5. **Rewrite `main.py`** — Unified CLI with all subsystem access

### Medium Priority
6. **Add `tests/` directory** — Unit tests for each module
7. **Add configuration file** (`config.yaml`) — Centralize paths, model names, thresholds
8. **Add async support** — `asyncio` for concurrent memory retrieval
9. **Add logging framework** — Structured logging instead of print statements
10. **Add type stubs** — Better IDE support for the large codebase

### Low Priority
11. **Add plugin system** — Dynamic tool registration for agents
12. **Add REST API** — FastAPI server for external integration
13. **Add web UI** — Streamlit/Graviton dashboard for visualization
14. **Add vector index backup** — Export/import ChromaDB snapshots
15. **Add model registry** — Swap embedding/LLM models without code changes

---

## 8. Future Scaling Ideas

### Short-term (1-3 months)
| Feature | Description |
|---------|-------------|
| **Async Agent Loop** | `async def execute()` for non-blocking multi-query retrieval |
| **Config-driven system** | YAML config for all paths, models, thresholds |
| **Structured logging** | Replace `print()` with Python `logging` module |
| **Unit test coverage** | 70%+ coverage for core modules |
| **CLI completion** | `mlrl02 chat`, `mlrl02 analyze`, `mlrl02 graph` subcommands |

### Medium-term (3-6 months)
| Feature | Description |
|---------|-------------|
| **Tool ecosystem** | File I/O, web search, code execution as agent tools |
| **Multi-model support** | Swap between llama3, mistral, qwen via config |
| **REST API server** | FastAPI endpoint for external applications |
| **Memory compression** | Summarize old memories to save vector space |
| **Cross-document reasoning** | Reason across multiple knowledge bases |

### Long-term (6-12 months)
| Feature | Description |
|---------|-------------|
| **Distributed memory** | Multi-machine ChromaDB cluster for large knowledge bases |
| **Learning from feedback** | User corrections update memory reinforcement weights |
| **Autonomous projects** | Agent loop runs continuously, improving knowledge base |
| **Knowledge graph visualization** | Interactive web UI with D3.js force-directed graph |
| **Multi-agent collaboration** | Multiple specialized agents sharing the same memory |

---

## 9. System Configuration (Proposed)

```yaml
# config.yaml (proposed)
system:
  name: MLRL02
  version: 2.0

paths:
  workspace: ./workspace
  markdown: ./workspace/markdown
  embeddings: ./workspace/embeddings
  docs: ./workspace/docs

memory:
  model: all-MiniLM-L6-v2
  collection: mlrl_memory
  top_k: 10
  min_similarity: 0.15
  max_context_tokens: 4000

reinforcement:
  max_boost: 1.0
  decay_rate: 0.01
  decay_half_life: 30
  weight: 0.4

reasoning:
  max_iterations: 3
  acceptance_threshold: 0.65
  model: llama3
  temperature: 0.7

graph:
  min_edge_weight: 0.15
  pagerank_iterations: 50
  pagerank_damping: 0.85
```

---

## 10. Data Contracts

### Document (from loader)
```python
{
    "page_content": str,      # Cleaned, chunked text
    "metadata": {
        "source": str,        # Filename
        "path": str,          # Full file path
        # + any frontmatter fields
    }
}
```

### ContextItem (from memory_context)
```python
{
    "text": str,              # Context text
    "source": str,            # Origin file
    "score": float,           # 0.0 – 1.0 relevance
    "chunk_id": str,          # Vector store ID
    "quality": float,         # 0.0 – 1.0 content quality
    "token_est": int,         # Estimated token count
}
```

### ReasoningResult (from reasoning_engine)
```python
{
    "answer": str,
    "question": str,
    "analysis": QuestionAnalysis,
    "reasoning_steps": list[ReasoningStep],
    "concept_links": list[ConceptLink],
    "contexts_used": list[ContextItem],
    "confidence": float,      # 0.0 – 1.0
}
```

### AgentResult (from agent_loop)
```python
{
    "goal": str,
    "final_answer": str,
    "plan": Plan,
    "reasoning_result": ReasoningResult,
    "reflection_result": ReflectionResult,
    "iterations": int,
    "tools_used": list[ToolCall],
    "logs": list[AgentLog],
    "session_id": str,
}
```

---

*End of Architecture Document*