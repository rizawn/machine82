"""
PROMPT BUILDER — Structured Prompt Construction for LLMs

Responsibilities:
    - Build layered prompts (system + memory + history + question)
    - Inject semantic memory context from retrieval
    - Add reasoning instructions for better LLM output
    - Support multiple local LLM profiles (llama3, mistral, etc.)
    - Keep prompts clean, structured, and extensible for agents

Design:
    Each prompt is built from composable sections:
        1. System Instructions  → AI persona, rules, behavior
        2. Reasoning Guide      → How to think, step-by-step logic
        3. Retrieved Memory     → Context from ChromaDB vector search
        4. Conversation History → Past turns for continuity
        5. User Question        → Current input

Usage:
    builder = PromptBuilder(model="llama3")
    builder.set_system("You are a helpful assistant.")
    builder.add_memory("Embeddings are vector representations...")
    builder.add_history("user", "What are embeddings?")
    builder.add_history("ai", "They are...")
    prompt = builder.build(question="Can you explain more?")
"""

import os
from typing import Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# ──────────────────────────────────────────────
#  LLM PROFILES
# ──────────────────────────────────────────────

# Each profile defines how the LLM prefers to receive prompts.
# Local models have different token limits and formatting needs.

LLM_PROFILES = {
    "llama3": {
        "max_context_tokens": 8000,
        "reasoning_style": "step-by-step",
        "context_first": True,
        "temperature_range": (0.1, 1.0),
    },
    "mistral": {
        "max_context_tokens": 8000,
        "reasoning_style": "concise",
        "context_first": False,
        "temperature_range": (0.1, 1.0),
    },
    "qwen": {
        "max_context_tokens": 32000,
        "reasoning_style": "structured",
        "context_first": True,
        "temperature_range": (0.1, 1.0),
    },
    "default": {
        "max_context_tokens": 4000,
        "reasoning_style": "balanced",
        "context_first": True,
        "temperature_range": (0.1, 1.0),
    },
}

# ──────────────────────────────────────────────
#  REASONING INSTRUCTIONS
# ──────────────────────────────────────────────

# Templates that teach the LLM HOW to think before answering.
# Different styles for different use cases.

REASONING_STYLES = {
    "step-by-step": """\
Before answering, think through the problem:
1. Identify what the user is really asking
2. Review the available context
3. Connect relevant facts
4. Form a clear, logical answer""",

    "concise": """\
Answer directly and concisely. No unnecessary elaboration.
Only use information from the context when relevant.""",

    "structured": """\
Structure your response:
- Start with a direct answer
- Support with evidence from context
- Mention uncertainty if context is incomplete
- Offer follow-up suggestions if appropriate""",

    "balanced": """\
Provide a balanced response:
- Be accurate and cite context when possible
- Acknowledge what you don't know
- Keep explanations accessible but not oversimplified""",
}


# ──────────────────────────────────────────────
#  DEFAULT SYSTEM PROMPTS
# ──────────────────────────────────────────────

SYSTEM_DEFAULT = """\
You are MLRL02, an AI assistant with long-term semantic memory.

Your knowledge comes from two sources:
1. Retrieved context from your memory system (vector database)
2. Your general training knowledge

Always prioritize retrieved context when answering.
If the context doesn't cover the topic, use your general knowledge
but mention that you're speaking from general understanding."""

SYSTEM_RESEARCHER = """\
You are MLRL02 Research Agent, specialized in analyzing and \
synthesizing information from stored knowledge.

You think critically, cite sources from context, and \
distinguish between facts and speculation."""

SYSTEM_REASONER = """\
You are MLRL02 Reasoning Agent. Your job is to work through \
complex problems step by step.

Break down questions into components, reason through each one,
and build your answer logically."""


# ──────────────────────────────────────────────
#  PROMPT BUILDER
# ──────────────────────────────────────────────

class PromptBuilder:
    """
    Builds structured, layered prompts for LLM interaction.

    The builder separates concerns into distinct sections:
    - System: Who the AI is and how it should behave
    - Reasoning: How the AI should think before answering
    - Memory: Retrieved context from semantic search
    - History: Past conversation turns
    - Question: Current user input

    This modular approach makes it easy to:
    - Swap system personas
    - Adjust reasoning depth
    - Inject different memory sources
    - Extend for autonomous agent loops
    """

    def __init__(
        self,
        model: str = "llama3",
        system_prompt: Optional[str] = None,
        reasoning_style: Optional[str] = None,
    ):
        self.model = model
        self.profile = LLM_PROFILES.get(model, LLM_PROFILES["default"])

        # State
        self._system = system_prompt or SYSTEM_DEFAULT
        self._reasoning = reasoning_style or self.profile["reasoning_style"]
        self._memory_contexts: list[dict] = []  # [{text, source, score}]
        self._history: list[dict] = []          # [{"role": ..., "content": ...}]
        self._extra_instructions: list[str] = []

    # ──────────────────────────────────────────
    #  CONFIGURATION
    # ──────────────────────────────────────────

    def set_system(self, prompt: str) -> "PromptBuilder":
        """Set the system-level instruction."""
        self._system = prompt
        return self

    def set_reasoning(self, style: str) -> "PromptBuilder":
        """
        Set reasoning style.

        Args:
            style: One of 'step-by-step', 'concise', 'structured', 'balanced'
        """
        if style not in REASONING_STYLES:
            raise ValueError(
                f"Unknown reasoning style: {style}. "
                f"Available: {', '.join(REASONING_STYLES.keys())}"
            )
        self._reasoning = style
        return self

    def add_instruction(self, instruction: str) -> "PromptBuilder":
        """Add a custom instruction to the system prompt."""
        self._extra_instructions.append(instruction)
        return self

    # ──────────────────────────────────────────
    #  MEMORY / CONTEXT
    # ──────────────────────────────────────────

    def add_memory(
        self,
        text: str,
        source: str = "unknown",
        score: Optional[float] = None,
    ) -> "PromptBuilder":
        """
        Add a retrieved memory chunk.

        Args:
            text:   The context text from vector search
            source: Where it came from (filename, topic, etc.)
            score:  Similarity score (optional)
        """
        self._memory_contexts.append({
            "text": text,
            "source": source,
            "score": score,
        })
        return self

    def add_memories(self, contexts: list[dict]) -> "PromptBuilder":
        """
        Batch add memory contexts.

        Each dict should have keys: 'text', 'source', optional 'score'.
        """
        for ctx in contexts:
            self.add_memory(
                text=ctx.get("text", ""),
                source=ctx.get("source", "unknown"),
                score=ctx.get("score"),
            )
        return self

    def clear_memory(self) -> "PromptBuilder":
        """Remove all memory contexts."""
        self._memory_contexts.clear()
        return self

    # ──────────────────────────────────────────
    #  HISTORY
    # ──────────────────────────────────────────

    def add_history(self, role: str, content: str) -> "PromptBuilder":
        """
        Add a conversation turn.

        Args:
            role:    'user' or 'ai'
            content: Message content
        """
        self._history.append({"role": role, "content": content})
        return self

    def set_history(self, history: list[dict]) -> "PromptBuilder":
        """Replace entire conversation history."""
        self._history = history.copy()
        return self

    def clear_history(self) -> "PromptBuilder":
        """Remove all conversation history."""
        self._history.clear()
        return self

    # ──────────────────────────────────────────
    #  BUILD
    # ──────────────────────────────────────────

    def _build_system_section(self) -> str:
        """Assemble the full system prompt."""
        parts = [self._system]

        # Add reasoning instructions
        reasoning_text = REASONING_STYLES.get(self._reasoning, "")
        if reasoning_text:
            parts.append(f"\nReasoning approach:\n{reasoning_text}")

        # Add custom instructions
        if self._extra_instructions:
            extra = "\n".join(f"- {inst}" for inst in self._extra_instructions)
            parts.append(f"\nAdditional instructions:\n{extra}")

        # Add date for temporal awareness
        parts.append(f"\nCurrent date: {datetime.now().strftime('%Y-%m-%d')}")

        return "\n".join(parts)

    def _build_memory_section(self) -> str:
        """Format retrieved contexts into a clean block."""
        if not self._memory_contexts:
            return "No relevant context found in memory."

        parts = []
        for i, ctx in enumerate(self._memory_contexts, 1):
            source = ctx["source"]
            text = ctx["text"]
            score_str = ""
            if ctx.get("score") is not None:
                score_str = f" [relevance: {ctx['score']:.2%}]"

            parts.append(f"[Context {i}] (source: {source}){score_str}\n{text}")

        return "\n\n---\n\n".join(parts)

    def _build_prompt_string(self, question: str) -> str:
        """
        Build a raw prompt string — useful for debugging or
        models that don't support structured message formats.
        """
        system = self._build_system_section()
        memory = self._build_memory_section()

        # History as text
        history_text = ""
        if self._history:
            history_lines = []
            for msg in self._history:
                role = "User" if msg["role"] == "user" else "AI"
                history_lines.append(f"{role}: {msg['content']}")
            history_text = "\n".join(history_lines)

        # Assemble
        blocks = [
            f"<system>\n{system}\n</system>",
            f"<memory>\n{memory}\n</memory>",
        ]

        if history_text:
            blocks.append(f"<history>\n{history_text}\n</history>")

        blocks.append(f"<question>\n{question}\n</question>")

        blocks.append(
            "\nProvide a clear, well-reasoned answer based on "
            "the context above."
        )

        return "\n\n".join(blocks)

    def build(self, question: str) -> ChatPromptTemplate:
        """
        Build a LangChain ChatPromptTemplate ready for LLM invocation.

        Args:
            question: The user's current question

        Returns:
            ChatPromptTemplate with all sections wired in
        """
        system_text = self._build_system_section()

        # Determine context placement based on model profile
        if self.profile.get("context_first", True):
            # Memory before question
            human_template = (
                "Here is the relevant context from memory:\n\n"
                "{memory}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        else:
            human_template = (
                "Here is relevant context:\n\n"
                "{memory}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )

        return ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder(variable_name="history"),
            ("human", human_template),
        ])

    def build_with_values(self, question: str) -> dict:
        """
        Build prompt with all values resolved — ready for chain.invoke().

        Args:
            question: The user's current question

        Returns:
            Dict of template variables
        """
        history_messages = []
        for msg in self._history:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                history_messages.append(AIMessage(content=msg["content"]))

        return {
            "memory": self._build_memory_section(),
            "history": history_messages,
            "question": question,
        }

    # ──────────────────────────────────────────
    #  DEBUG / INSPECT
    # ──────────────────────────────────────────

    def preview(self, question: str) -> str:
        """
        Show the raw prompt text — useful for debugging.

        Args:
            question: The user's question

        Returns:
            Full prompt as a string
        """
        return self._build_prompt_string(question)

    def get_stats(self) -> dict:
        """Info about the current prompt state."""
        system_tokens = len(self._system.split())
        memory_tokens = sum(
            len(ctx["text"].split()) for ctx in self._memory_contexts
        )
        history_tokens = sum(
            len(msg["content"].split()) for msg in self._history
        )

        return {
            "model": self.model,
            "reasoning_style": self._reasoning,
            "system_tokens": system_tokens,
            "memory_contexts": len(self._memory_contexts),
            "memory_tokens": memory_tokens,
            "history_turns": len(self._history) // 2,
            "history_tokens": history_tokens,
            "total_estimated_tokens": system_tokens + memory_tokens + history_tokens,
            "max_context_tokens": self.profile["max_context_tokens"],
        }

    def reset(self) -> "PromptBuilder":
        """Clear everything except system prompt and model config."""
        self._memory_contexts.clear()
        self._history.clear()
        self._extra_instructions.clear()
        return self


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PROMPT BUILDER — Quick Test")
    print("=" * 60 + "\n")

    # ── Test 1: Basic usage ──
    print("--- Test 1: Basic Prompt ---\n")
    builder = PromptBuilder(model="llama3")
    builder.add_memory(
        text="Embeddings are numerical representations of text. "
             "Vector databases store them for semantic search.",
        source="ai_notes.md",
        score=0.85,
    )
    builder.add_history("user", "What are embeddings?")
    builder.add_history("ai", "They are vector representations of text.")

    prompt_preview = builder.preview("How does semantic search use them?")
    print(prompt_preview)

    # ── Test 2: Stats ──
    print("\n" + "=" * 60)
    print("--- Stats ---\n")
    for k, v in builder.get_stats().items():
        print(f"  {k}: {v}")

    # ── Test 3: Different reasoning style ──
    print("\n" + "=" * 60)
    print("--- Test 2: Researcher Profile ---\n")
    builder2 = (
        PromptBuilder(model="mistral")
        .set_system(SYSTEM_RESEARCHER)
        .set_reasoning("structured")
        .add_instruction("Always cite which context block you reference")
        .add_memory(
            text="RAG combines retrieval with generation. "
                 "First find relevant docs, then generate an answer.",
            source="architecture.md",
        )
        .add_history("user", "What is RAG?")
    )
    print(builder2.preview("Explain how RAG works step by step"))
