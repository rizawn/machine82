"""
AGENT LOOP — Autonomous Agent Workspace

Architecture:
    Goal → Planning → Memory Retrieval → Reasoning → Action → Reflection
    ↑                                                        ↓
    └──────────────────── Iterate if needed ←─────────────────┘

The agent loop orchestrates all MLRL02 modules into a
coherent autonomous reasoning cycle:

    1. GOAL       — Receive and understand the objective
    2. PLAN       — Decompose into actionable tasks (TaskPlanner)
    3. RETRIEVE   — Fetch relevant memory (MemoryContext)
    4. REASON     — Build reasoning chain (ReasoningEngine)
    5. ACT        — Execute and generate response
    6. REFLECT    — Evaluate quality (ReflectionEngine)
    7. ITERATE    — Loop if quality is insufficient

Features:
    - Modular: each stage uses a dedicated module
    - Transparent: every step is logged and traceable
    - Self-correcting: reflection can trigger re-reasoning
    - Tool-ready: placeholder for future tool integration
    - Task-aware: can decompose complex goals into sub-tasks

Usage:
    agent = AgentLoop()
    result = agent.execute("Explain how embeddings enable semantic search")
    print(result.final_answer)
    agent.print_execution_log()
"""

import os
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime

# Project root sys.path hack removed from module level (M-1)

from core.memory.memory_context import MemoryContext, ContextItem
from core.reasoning.reasoning_engine import ReasoningEngine, ReasoningResult
from core.reasoning.reflection_engine import ReflectionEngine, ReflectionResult
from core.agents.task_planner import TaskPlanner, Plan, TaskStatus
from core.reasoning.prompt_builder import PromptBuilder
from core.agents.chat_engine import ChatEngine


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

# Maximum reflection iterations before stopping
MAX_ITERATIONS = 3

# Minimum reflection score to accept an answer
ACCEPTANCE_THRESHOLD = 0.65

# Log format
LOG_FMT = "[{time}] {stage}: {message}"


# ──────────────────────────────────────────────
#  ENUMS
# ──────────────────────────────────────────────

class AgentStage(Enum):
    """Stages in the agent loop."""
    IDLE = "idle"
    GOAL = "goal"
    PLAN = "plan"
    RETRIEVE = "retrieve"
    REASON = "reason"
    ACT = "act"
    REFLECT = "reflect"
    ITERATE = "iterate"
    COMPLETE = "complete"
    FAILED = "failed"


class ToolType(Enum):
    """Available tool types (future extensibility)."""
    SEARCH = "search"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_CODE = "execute_code"
    CUSTOM = "custom"


# ──────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class AgentLog:
    """
    A single log entry from the agent execution.

    Attributes:
        timestamp:  When this event occurred
        stage:      Which agent stage produced this log
        message:    What happened
        data:       Optional structured data
    """
    timestamp: str
    stage: AgentStage
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    """
    A tool execution record.

    Attributes:
        tool_type:  What kind of tool
        name:       Tool identifier
        input:      Tool input parameters
        output:     Tool output/result
        success:    Whether execution succeeded
    """
    tool_type: ToolType
    name: str
    input: dict
    output: str = ""
    success: bool = False


@dataclass
class AgentResult:
    """
    Final output from the agent loop.

    Attributes:
        goal:              Original goal/question
        final_answer:      The accepted response
        plan:              Execution plan used
        reasoning_result:  Reasoning pipeline output
        reflection_result: Final evaluation
        iterations:        How many cycles were needed
        tools_used:        List of tool calls made
        logs:              Full execution log
        session_id:        Unique session identifier
    """
    goal: str
    final_answer: str
    plan: Optional[Plan] = None
    reasoning_result: Optional[ReasoningResult] = None
    reflection_result: Optional[ReflectionResult] = None
    iterations: int = 1
    tools_used: list[ToolCall] = field(default_factory=list)
    logs: list[AgentLog] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


# ──────────────────────────────────────────────
#  TOOL REGISTRY (Future Extensibility)
# ──────────────────────────────────────────────

class ToolRegistry:
    """
    Registry for tools the agent can use.

    Currently a placeholder — ready for future
    tool integration (file I/O, code execution, etc.).
    """

    def __init__(self):
        self._tools: dict[str, callable] = {}

    def register(self, name: str, func: callable):
        """Register a tool function."""
        self._tools[name] = func

    def execute(
        self,
        tool_type: ToolType,
        name: str,
        **kwargs,
    ) -> ToolCall:
        """
        Execute a registered tool.

        Args:
            tool_type: Type of tool
            name:      Tool name
            **kwargs:  Tool arguments

        Returns:
            ToolCall record
        """
        call = ToolCall(
            tool_type=tool_type,
            name=name,
            input=kwargs,
        )

        func = self._tools.get(name)
        if func:
            try:
                result = func(**kwargs)
                call.output = str(result)
                call.success = True
            except Exception as e:
                call.output = f"Error: {e}"
                call.success = False
        else:
            call.output = f"Tool '{name}' not registered"
            call.success = False

        return call


# ──────────────────────────────────────────────
#  AGENT LOOP
# ──────────────────────────────────────────────

class AgentLoop:
    """
    Autonomous agent orchestration loop.

    Coordinates all MLRL02 modules into a self-correcting
    reasoning cycle.

    Flow:
        1. Receive goal
        2. Create plan (TaskPlanner)
        3. Retrieve memory (MemoryContext)
        4. Reason about the goal (ReasoningEngine)
        5. Generate answer (Action)
        6. Evaluate quality (ReflectionEngine)
        7. If quality is low → iterate with refined approach
        8. Return best answer

    Usage:
        agent = AgentLoop()
        result = agent.execute("What is RAG?")
        print(result.final_answer)
    """

    def __init__(
        self,
        memory: Optional[MemoryContext] = None,
        reasoning: Optional[ReasoningEngine] = None,
        reflection: Optional[ReflectionEngine] = None,
        planner: Optional[TaskPlanner] = None,
        chat: Optional[ChatEngine] = None,
        max_iterations: int = MAX_ITERATIONS,
        acceptance_threshold: float = ACCEPTANCE_THRESHOLD,
    ):
        # Core modules
        self.memory = memory or MemoryContext()
        self.reasoning = reasoning or ReasoningEngine(memory=self.memory)
        self.reflection = reflection or ReflectionEngine()
        self.planner = planner or TaskPlanner()
        self.prompt_builder = PromptBuilder()
        self.chat = chat or ChatEngine()

        # Config
        self.max_iterations = max_iterations
        self.acceptance_threshold = acceptance_threshold

        # State
        self.tools = ToolRegistry()
        self._current_stage = AgentStage.IDLE
        self._logs: list[AgentLog] = []
        self._tools_used: list[ToolCall] = []

    # ──────────────────────────────────────────
    #  MAIN EXECUTION
    # ──────────────────────────────────────────

    def execute(self, goal: str, use_llm: bool = True) -> AgentResult:
        """
        Run the full agent loop.

        Args:
            goal:    The objective or question
            use_llm: If True, use LLM (ChatEngine) for final answer generation.
                     If False, use ReasoningEngine's internal answer.

        Returns:
            AgentResult with answer and full trace
        """
        self._logs.clear()
        self._tools_used.clear()
        session_id = str(uuid.uuid4())[:8]

        self._log(AgentStage.GOAL, f"Received goal: {goal[:80]}...")

        # Stage 1: Plan
        self._current_stage = AgentStage.PLAN
        plan = self._stage_plan(goal)

        # Stage 2: Retrieve memory
        self._current_stage = AgentStage.RETRIEVE
        contexts = self._stage_retrieve(goal)

        # Stage 3-7: Reasoning loop with reflection
        best_result = None
        best_score = 0.0
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            self._current_stage = AgentStage.REASON
            self._log(AgentStage.REASON, f"Reasoning iteration {iteration}")

            # Reason
            reasoning_result = self._stage_reason(goal, contexts)

            # Act (generate answer)
            self._current_stage = AgentStage.ACT
            answer = self._stage_act(goal, reasoning_result, contexts, use_llm=use_llm)

            # Reflect
            self._current_stage = AgentStage.REFLECT
            reflection_result = self._stage_reflect(
                goal, answer, contexts
            )

            score = reflection_result.overall_score

            # Track best
            if score > best_score:
                best_score = score
                best_result = AgentResult(
                    goal=goal,
                    final_answer=answer,
                    plan=plan,
                    reasoning_result=reasoning_result,
                    reflection_result=reflection_result,
                    iterations=iteration,
                    tools_used=list(self._tools_used),
                    logs=list(self._logs),
                    session_id=session_id,
                )

            # Check if good enough
            if score >= self.acceptance_threshold:
                self._log(
                    AgentStage.COMPLETE,
                    f"Answer accepted (score: {score:.0%})",
                )
                best_result.iterations = iteration
                break

            # Iterate if needed
            if iteration < self.max_iterations:
                self._current_stage = AgentStage.ITERATE
                self._log(
                    AgentStage.ITERATE,
                    f"Score {score:.0%} below threshold "
                    f"({self.acceptance_threshold:.0%}). Refining...",
                )
                # Refine context for next iteration
                contexts = self._refine_context(
                    goal, contexts, reflection_result
                )
            else:
                self._log(
                    AgentStage.COMPLETE,
                    f"Max iterations reached. Best score: {best_score:.0%}",
                )

        if not best_result:
            best_result = AgentResult(
                goal=goal,
                final_answer="Unable to generate a satisfactory answer.",
                plan=plan,
                iterations=iteration,
                tools_used=list(self._tools_used),
                logs=list(self._logs),
                session_id=session_id,
            )

        self._current_stage = AgentStage.COMPLETE
        return best_result

    # ──────────────────────────────────────────
    #  STAGE IMPLEMENTATIONS
    # ──────────────────────────────────────────

    def _stage_plan(self, goal: str) -> Plan:
        """
        Stage: PLAN
        Decompose the goal into actionable tasks.
        """
        self._log(AgentStage.PLAN, "Creating execution plan")

        plan = self.planner.plan(goal)
        self._log(
            AgentStage.PLAN,
            f"Plan created: {len(plan.tasks)} tasks, "
            f"strategy: {plan.strategy.value}",
        )

        for task in plan.tasks[:3]:  # Log first 3 tasks
            self._log(
                AgentStage.PLAN,
                f"  Task {task.id}: [{task.priority.value}] {task.title}",
            )

        return plan

    def _stage_retrieve(self, goal: str) -> list[ContextItem]:
        """
        Stage: RETRIEVE
        Fetch relevant memory for the goal.
        """
        self._log(AgentStage.RETRIEVE, "Retrieving relevant memory")

        contexts = self.memory.retrieve(goal)
        self._log(
            AgentStage.RETRIEVE,
            f"Retrieved {len(contexts)} contexts "
            f"(memory has {self.memory.count()} documents)",
        )

        for i, ctx in enumerate(contexts[:3], 1):
            self._log(
                AgentStage.RETRIEVE,
                f"  [{i}] {ctx.source} (score: {ctx.score:.0%})",
            )

        return contexts

    def _stage_reason(
        self,
        goal: str,
        contexts: list[ContextItem],
    ) -> ReasoningResult:
        """
        Stage: REASON
        Build a reasoning chain for the goal.

        Note: We explicitly pass use_llm=False here to perform context analysis
        and concept linking locally. The actual LLM text generation is deferred
        to the ACT stage (_stage_act) to prevent duplicate LLM calls.
        """
        self._log(AgentStage.REASON, "Building reasoning chain")

        # Local cognitive reasoning to build contexts & concept links
        result = self.reasoning.reason(goal, use_llm=False)

        self._log(
            AgentStage.REASON,
            f"Reasoning complete: {len(result.reasoning_steps)} steps, "
            f"{len(result.contexts_used)} contexts used, "
            f"confidence: {result.confidence:.0%}",
        )

        return result

    def _stage_act(
        self,
        goal: str,
        reasoning_result: ReasoningResult,
        contexts: list[ContextItem],
        use_llm: bool = True,
    ) -> str:
        """
        Stage: ACT
        Generate the answer from reasoning.
        """
        self._log(AgentStage.ACT, "Generating response")

        if use_llm:
            # Build enriched context from reasoning results to pass to ChatEngine
            self._log(AgentStage.ACT, "Using ChatEngine for LLM generation")

            # Combine reasoning contexts into an external_context string
            context_parts = []
            used_contexts = reasoning_result.contexts_used or contexts
            for ctx in used_contexts[:5]:
                context_parts.append(
                    f"[{ctx.source}] (relevance: {ctx.score:.0%})\n{ctx.text}"
                )

            # Add concept links if discovered
            if reasoning_result.concept_links:
                link_parts = []
                for link in reasoning_result.concept_links:
                    link_parts.append(
                        f"- {link.source_concept} ↔ {link.related_concept} "
                        f"(strength: {link.connection_strength:.0%})"
                    )
                context_parts.append(
                    "Related concepts:\n" + "\n".join(link_parts)
                )

            external_ctx = "\n\n---\n\n".join(context_parts) if context_parts else None
            answer = self.chat.chat(goal, external_context=external_ctx)
        else:
            # Use the reasoning engine's answer (already generated)
            answer = reasoning_result.answer

            # If the answer is just a fallback message, try to build better from context
            if "don't have specific information" in answer.lower():
                answer = self._build_context_answer(goal, contexts)

        self._log(
            AgentStage.ACT,
            f"Response generated ({len(answer.split())} words)",
        )

        return answer

    def _stage_reflect(
        self,
        goal: str,
        answer: str,
        contexts: list[ContextItem],
    ) -> ReflectionResult:
        """
        Stage: REFLECT
        Evaluate the answer quality.
        """
        self._log(AgentStage.REFLECT, "Evaluating response quality")

        result = self.reflection.evaluate(goal, answer, contexts)

        self._log(
            AgentStage.REFLECT,
            f"Score: {result.overall_score:.0%} (Grade: {result.grade}), "
            f"{'Strong ✅' if result.is_strong else 'Weak ⚠️' if result.is_weak else 'Moderate'}",
        )

        if result.improvements:
            for imp in result.improvements[:2]:
                self._log(
                    AgentStage.REFLECT,
                    f"  Improvement [{imp.priority}]: {imp.suggestion[:80]}",
                )

        return result

    # ──────────────────────────────────────────
    #  CONTEXT REFINEMENT (Iterative Improvement)
    # ──────────────────────────────────────────

    def _refine_context(
        self,
        goal: str,
        current_contexts: list[ContextItem],
        reflection: ReflectionResult,
    ) -> list[ContextItem]:
        """
        Refine context for the next iteration based on reflection feedback.

        Strategy:
        - If concepts are missing, search for them specifically
        - If answer was incomplete, broaden the search
        """
        self._log(AgentStage.ITERATE, "Refining context for next iteration")

        new_contexts = list(current_contexts)

        # Search for missing concepts
        for concept in reflection.missing_concepts[:2]:
            additional = self.memory.retrieve(concept, top_k=2)
            existing_ids = {c.chunk_id for c in new_contexts}
            for ctx in additional:
                if ctx.chunk_id not in existing_ids:
                    new_contexts.append(ctx)
                    existing_ids.add(ctx.chunk_id)
                    self._log(
                        AgentStage.ITERATE,
                        f"  Added context for missing concept: {concept}",
                    )

        # Also search with broader query
        broader = self.memory.retrieve(goal, top_k=3, max_tokens=6000)
        existing_ids = {c.chunk_id for c in new_contexts}
        for ctx in broader:
            if ctx.chunk_id not in existing_ids:
                new_contexts.append(ctx)

        self._log(
            AgentStage.ITERATE,
            f"Context refined: {len(new_contexts)} contexts "
            f"(was {len(current_contexts)})",
        )

        return new_contexts

    # ──────────────────────────────────────────
    #  FALLBACK ANSWER BUILDER
    # ──────────────────────────────────────────

    def _build_context_answer(
        self,
        goal: str,
        contexts: list[ContextItem],
    ) -> str:
        """Build an answer directly from contexts."""
        if not contexts:
            return (
                f"I don't have specific information about this topic "
                f"in my memory yet."
            )

        parts = [f"Based on my knowledge about this topic:\n"]

        for ctx in contexts[:3]:
            parts.append(f"\nFrom {ctx.source}:")
            parts.append(ctx.text)

        return "\n".join(parts)

    # ──────────────────────────────────────────
    #  TOOL USAGE (Future)
    # ──────────────────────────────────────────

    def use_tool(
        self,
        tool_type: ToolType,
        name: str,
        **kwargs,
    ) -> ToolCall:
        """
        Execute a tool and record the call.

        Args:
            tool_type: Type of tool
            name:      Tool name
            **kwargs:  Tool arguments

        Returns:
            ToolCall record
        """
        self._log(AgentStage.ACT, f"Using tool: {name}")
        result = self.tools.execute(tool_type, name, **kwargs)
        self._tools_used.append(result)
        return result

    # ──────────────────────────────────────────
    #  LOGGING
    # ──────────────────────────────────────────

    def _log(self, stage: AgentStage, message: str, data: dict = None):
        """Add a log entry."""
        entry = AgentLog(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            stage=stage,
            message=message,
            data=data or {},
        )
        self._logs.append(entry)

    def get_logs(self) -> list[AgentLog]:
        """Get all execution logs."""
        return list(self._logs)

    def print_execution_log(self):
        """Pretty-print the full execution log."""
        print("\n" + "=" * 60)
        print("  📋 AGENT EXECUTION LOG")
        print("=" * 60)

        stage_icons = {
            AgentStage.GOAL: "🎯",
            AgentStage.PLAN: "📋",
            AgentStage.RETRIEVE: "🔍",
            AgentStage.REASON: "🧠",
            AgentStage.ACT: "⚡",
            AgentStage.REFLECT: "🔎",
            AgentStage.ITERATE: "🔄",
            AgentStage.COMPLETE: "✅",
            AgentStage.FAILED: "❌",
            AgentStage.IDLE: "⏸️",
        }

        for entry in self._logs:
            icon = stage_icons.get(entry.stage, "•")
            print(f"  {icon} [{entry.timestamp}] {entry.stage.value.upper()}")
            print(f"     {entry.message}")

        print()

    def print_result(self, result: AgentResult):
        """Pretty-print the agent result."""
        print("\n" + "=" * 60)
        print(f"  🤖 AGENT RESULT (Session: {result.session_id})")
        print("=" * 60)

        print(f"\n🎯 Goal: {result.goal}")
        print(f"\n💡 Answer:")
        print(f"   {result.final_answer[:500]}")
        if len(result.final_answer) > 500:
            print(f"   ...")

        if result.reflection_result:
            r = result.reflection_result
            print(f"\n📊 Quality: {r.overall_score:.0%} (Grade: {r.grade})")
            print(f"   Iterations: {result.iterations}")
            print(f"   Contexts used: {len(result.reasoning_result.contexts_used) if result.reasoning_result else 0}")

        if result.plan:
            print(f"\n📋 Plan progress: {result.plan.progress:.0%}")

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
    print("  AGENT LOOP — Quick Test")
    print("=" * 60 + "\n")

    agent = AgentLoop()

    goals = [
        "Apa itu embeddings?",
        "Bagaimana semantic search bekerja?",
    ]

    for goal in goals:
        print(f"\n{'=' * 60}")
        print(f"🎯 GOAL: {goal}")
        print(f"{'=' * 60}")

        result = agent.execute(goal)
        agent.print_result(result)
        agent.print_execution_log()

        print("-" * 60)
