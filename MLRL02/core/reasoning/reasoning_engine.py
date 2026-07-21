"""
REASONING ENGINE — Lightweight Cognitive Reasoning Pipeline

System goals:
    - Multi-step reasoning before answering
    - Context chaining (retrieve → expand → synthesize)
    - Concept linking (detect related ideas from memory)
    - Semantic understanding (analyze intent, not just keywords)
    - Memory-aware responses (grounded in stored knowledge)

Architecture:
    This module sits between the user and the LLM. It acts as a
    cognitive reasoning layer that:

        1. ANALYZE    — Break down the question (intent, key terms, complexity)
        2. EXPAND     — Find related concepts via memory queries
        3. CHAIN      — Build a reasoning chain from retrieved contexts
        4. SYNTHESIZE — Prepare structured context for the LLM
        5. RESPOND    — Return a reasoning-grounded answer

    Flow:
        User Question
          → QuestionAnalyzer (intent, concepts, complexity)
          → MemoryExplorer (multi-query retrieval + concept linking)
          → ReasoningChain (build logical chain from contexts)
          → Synthesized Context (for LLM prompt injection)
          → Response

Usage:
    engine = ReasoningEngine()
    result = engine.reason("How do embeddings enable semantic search?")
    print(result.answer)
    print(result.reasoning_steps)  # See the thinking process
"""

import os
import sys
import re
from dataclasses import dataclass, field
from typing import Optional

# Project root sys.path hack removed from module level (M-1)

from core.memory.memory_context import MemoryContext, ContextItem
from core.reasoning.prompt_builder import PromptBuilder


# ──────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class QuestionAnalysis:
    """
    Structured analysis of a user question.

    Attributes:
        raw:          Original question text
        key_terms:    Important nouns/concepts extracted
        intent:       What the user wants (explain, compare, how-to, etc.)
        complexity:   Simple / moderate / complex
        needs_context: Whether this question benefits from memory lookup
        expansion_queries: Follow-up queries to find related concepts
    """
    raw: str
    key_terms: list[str] = field(default_factory=list)
    intent: str = "unknown"
    complexity: str = "simple"
    needs_context: bool = True
    expansion_queries: list[str] = field(default_factory=list)


@dataclass
class ReasoningConceptLink:
    """
    A discovered relationship between concepts in memory.

    Attributes:
        source_concept: The original concept from the question
        related_concept: A related concept found in memory
        connection_strength: How strongly they're related (0.0 – 1.0)
        evidence_text: The context text that shows this connection
    """
    source_concept: str
    related_concept: str
    connection_strength: float
    evidence_text: str


@dataclass
class ReasoningStep:
    """
    One step in the reasoning chain.

    Attributes:
        step_number: Position in the chain (1-based)
        step_type:   'analyze', 'retrieve', 'connect', 'synthesize'
        description: What this step does
        evidence:    Supporting context or data
    """
    step_number: int
    step_type: str
    description: str
    evidence: str = ""


@dataclass
class ReasoningResult:
    """
    Final output from the reasoning engine.

    Attributes:
        answer:           The generated response
        question:         Original question
        analysis:         QuestionAnalysis object
        reasoning_steps:  List of steps taken
        concept_links:    Discovered concept relationships
        contexts_used:    Memory contexts that informed the answer
        confidence:       Self-assessed confidence (0.0 – 1.0)
    """
    answer: str
    question: str
    analysis: QuestionAnalysis
    reasoning_steps: list[ReasoningStep] = field(default_factory=list)
    concept_links: list[ReasoningConceptLink] = field(default_factory=list)
    contexts_used: list[ContextItem] = field(default_factory=list)
    confidence: float = 0.0


# ──────────────────────────────────────────────
#  QUESTION ANALYZER
# ──────────────────────────────────────────────

class QuestionAnalyzer:
    """
    Stage 1: Analyze the user question.

    Extracts key terms, detects intent, assesses complexity,
    and generates expansion queries for deeper retrieval.
    """

    # Stop words to filter out when extracting key terms
    STOP_WORDS = {
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
        "apa", "itu", "yang", "dan", "atau", "dari", "untuk",
        "dengan", "pada", "dalam", "adalah", "bagaimana", "cara",
        "bisa", "tidak", "ini", "itu", "jika", "karena", "oleh",
        "tentang", "juga", "sudah", "akan", "secara",
    }

    # Intent detection patterns
    INTENT_PATTERNS = {
        "explain": [
            r"\b(what|apa)\b.*\b(is|itu|are)\b",
            r"\b(explain|jelaskan|describe|deskripsikan)\b",
            r"\b(tell me about|ceritakan tentang)\b",
        ],
        "how-to": [
            r"\b(how|bagaimana)\b.*\b(do|make|create|build|kerja)\b",
            r"\b(cara)\b",
            r"\b(step|langkah)\b",
        ],
        "compare": [
            r"\b(difference|perbedaan|vs\.?|versus|verses|versus)\b",
            r"\b(compare|bandingkan)\b.*\b(and|dengan)\b",
            r"\b(similar|sama)\b.*\b(different|beda)\b",
        ],
        "why": [
            r"\b(why|kenapa|mengapa)\b",
            r"\b(reason|alasan)\b.*\b(for|untuk)\b",
        ],
        "list": [
            r"\b(list|daftar)\b.*\b(of|dari)\b",
            r"\b(types?|jenis)\b.*\b(of|dari)\b",
            r"\b(examples?|contoh)\b",
        ],
    }

    def analyze(self, question: str) -> QuestionAnalysis:
        """
        Full analysis pipeline.

        Args:
            question: The user's question

        Returns:
            QuestionAnalysis with extracted insights
        """
        analysis = QuestionAnalysis(raw=question)
        analysis.key_terms = self._extract_key_terms(question)
        analysis.intent = self._detect_intent(question)
        analysis.complexity = self._assess_complexity(question, analysis.key_terms)
        analysis.needs_context = self._needs_context(analysis.intent, analysis.key_terms)
        analysis.expansion_queries = self._generate_expansion_queries(
            question, analysis.key_terms, analysis.intent
        )

        return analysis

    def _extract_key_terms(self, question: str) -> list[str]:
        """
        Extract meaningful terms from the question.

        Strategy: split on word boundaries, filter stop words,
        keep words with 3+ characters that are alphabetic or
        technical terms (contain underscores, hyphens).
        """
        # Split on non-word characters
        tokens = re.findall(r"[\w\-]+", question.lower())

        # Filter stop words and short tokens
        terms = [
            t for t in tokens
            if len(t) >= 3
            and t not in self.STOP_WORDS
            and not t.isdigit()
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        return unique

    def _detect_intent(self, question: str) -> str:
        """Detect what the user wants to do."""
        q_lower = question.lower()

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q_lower):
                    return intent

        return "general"

    def _assess_complexity(
        self, question: str, key_terms: list[str]
    ) -> str:
        """
        Assess question complexity based on:
        - Number of key concepts
        - Question length
        - Presence of relational words
        """
        score = 0

        # More key terms = more complex
        if len(key_terms) >= 4:
            score += 2
        elif len(key_terms) >= 2:
            score += 1

        # Longer questions tend to be more complex
        if len(question.split()) > 15:
            score += 1

        # Relational words indicate multi-concept reasoning
        relational = {"and", "or", "versus", "vs", "compare", "difference",
                      "how", "why", "relationship", "between", "connect"}
        question_lower = question.lower()
        if any(w in question_lower for w in relational):
            score += 1

        if score >= 3:
            return "complex"
        elif score >= 1:
            return "moderate"
        return "simple"

    def _needs_context(
        self, intent: str, key_terms: list[str]
    ) -> bool:
        """Whether this question benefits from memory lookup."""
        # Factual, explanatory questions need context
        if intent in ("explain", "how-to", "why", "list"):
            return True
        # Questions with multiple key terms likely need context
        if len(key_terms) >= 2:
            return True
        return False

    def _generate_expansion_queries(
        self,
        question: str,
        key_terms: list[str],
        intent: str,
    ) -> list[str]:
        """
        Generate follow-up queries to find related concepts.

        For a question like "How do embeddings enable semantic search?",
        we want to also search for:
        - "embeddings"
        - "semantic search"
        - "vector databases" (related concept)
        """
        queries = [question]  # Always include the original

        # Add queries for each key term
        for term in key_terms[:3]:  # Max 3 expansion queries
            queries.append(term)

        # For "how" and "why" questions, add mechanism-focused query
        if intent in ("how-to", "why") and key_terms:
            primary = key_terms[0]
            queries.append(f"how {primary} works")

        # For comparison questions, query both sides
        if intent == "compare" and len(key_terms) >= 2:
            queries.append(key_terms[0])
            queries.append(key_terms[1])

        # Deduplicate
        seen = set()
        unique = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)

        return unique[:5]  # Cap at 5 expansion queries


# ──────────────────────────────────────────────
#  MEMORY EXPLORER
# ──────────────────────────────────────────────

class MemoryExplorer:
    """
    Stage 2: Explore memory for related concepts.

    Takes expansion queries from the analyzer, retrieves
    contexts for each, and discovers concept links between them.
    """

    def __init__(self, memory: MemoryContext):
        self.memory = memory

    def explore(
        self,
        analysis: QuestionAnalysis,
        max_contexts: int = 8,
    ) -> tuple[list[ContextItem], list[ReasoningConceptLink]]:
        """
        Run multiple retrieval queries and merge results.

        Args:
            analysis:     The analyzed question
            max_contexts: Max total contexts to return

        Returns:
            Tuple of (merged contexts, discovered concept links)
        """
        if not analysis.needs_context or not self.memory.is_available():
            return [], []

        all_contexts: list[ContextItem] = []
        seen_ids: set[str] = set()

        # Run each expansion query
        for query in analysis.expansion_queries:
            results = self.memory.retrieve(query, top_k=5)
            for ctx in results:
                if ctx.chunk_id not in seen_ids:
                    seen_ids.add(ctx.chunk_id)
                    all_contexts.append(ctx)

        if not all_contexts:
            return [], []

        # Sort by combined score
        all_contexts.sort(
            key=lambda c: c.score * 0.6 + c.quality * 0.4,
            reverse=True,
        )

        # Discover concept links between contexts
        concept_links = self._find_concept_links(
            analysis.key_terms, all_contexts
        )

        # Limit to max_contexts
        return all_contexts[:max_contexts], concept_links

    def _find_concept_links(
        self,
        key_terms: list[str],
        contexts: list[ContextItem],
    ) -> list[ReasoningConceptLink]:
        """
        Find relationships between concepts across contexts.

        Strategy: for each key term, check which other terms
        appear in the same context chunk. Co-occurrence = link.
    """
        links: list[ReasoningConceptLink] = []

        for ctx in contexts:
            text_lower = ctx.text.lower()

            # Find which key terms appear in this context
            present_terms = [
                t for t in key_terms if t.lower() in text_lower
            ]

            # If multiple key terms co-occur, that's a link
            if len(present_terms) >= 2:
                for i, term_a in enumerate(present_terms):
                    for term_b in present_terms[i + 1:]:
                        # Strength based on context quality + score
                        strength = (ctx.score * 0.5 + ctx.quality * 0.5)

                        # Check if this link already exists
                        exists = any(
                            (l.source_concept == term_a and
                             l.related_concept == term_b)
                            or
                            (l.source_concept == term_b and
                             l.related_concept == term_a)
                            for l in links
                        )
                        if not exists:
                            links.append(ReasoningConceptLink(
                                source_concept=term_a,
                                related_concept=term_b,
                                connection_strength=round(strength, 3),
                                evidence_text=ctx.text[:200],
                            ))

        return links


# ──────────────────────────────────────────────
#  REASONING CHAIN BUILDER
# ──────────────────────────────────────────────

class ReasoningChainBuilder:
    """
    Stage 3: Build a logical reasoning chain from retrieved contexts.

    Creates a step-by-step chain that shows:
    - What was understood about the question
    - What memories were retrieved
    - How concepts connect
    - What synthesis was performed
    """

    def build(
        self,
        analysis: QuestionAnalysis,
        contexts: list[ContextItem],
        concept_links: list[ReasoningConceptLink],
    ) -> list[ReasoningStep]:
        """
        Build the full reasoning chain.

        Args:
            analysis:     Question analysis
            contexts:     Retrieved memory contexts
            concept_links: Discovered concept relationships

        Returns:
            List of ReasoningStep objects
        """
        steps: list[ReasoningStep] = []
        step_num = 0

        # Step 1: Analysis
        step_num += 1
        steps.append(ReasoningStep(
            step_number=step_num,
            step_type="analyze",
            description=(
                f"Analyzed question: intent='{analysis.intent}', "
                f"complexity='{analysis.complexity}', "
                f"key terms: {', '.join(analysis.key_terms[:5])}"
            ),
        ))

        # Step 2: Retrieval summary
        step_num += 1
        if contexts:
            sources = list(set(c.source for c in contexts))
            avg_score = sum(c.score for c in contexts) / len(contexts)
            steps.append(ReasoningStep(
                step_number=step_num,
                step_type="retrieve",
                description=(
                    f"Retrieved {len(contexts)} relevant contexts "
                    f"(avg relevance: {avg_score:.0%}) "
                    f"from sources: {', '.join(sources)}"
                ),
                evidence=self._summarize_contexts(contexts),
            ))
        else:
            steps.append(ReasoningStep(
                step_number=step_num,
                step_type="retrieve",
                description="No relevant contexts found in memory. "
                           "Will answer from general knowledge.",
            ))

        # Step 3: Concept linking
        if concept_links:
            step_num += 1
            link_descriptions = []
            for link in concept_links:
                link_descriptions.append(
                    f"{link.source_concept} ↔ {link.related_concept} "
                    f"(strength: {link.connection_strength:.0%})"
                )
            steps.append(ReasoningStep(
                step_number=step_num,
                step_type="connect",
                description=(
                    f"Found {len(concept_links)} concept relationships: "
                    + "; ".join(link_descriptions)
                ),
            ))

        # Step 4: Synthesis
        step_num += 1
        steps.append(ReasoningStep(
            step_number=step_num,
            step_type="synthesize",
            description=(
                f"Synthesizing answer for '{analysis.intent}' question "
                f"using {len(contexts)} context(s) and "
                f"{len(concept_links)} concept link(s)"
            ),
        ))

        return steps

    def _summarize_contexts(self, contexts: list[ContextItem]) -> str:
        """Create a brief summary of retrieved contexts."""
        summaries = []
        for ctx in contexts[:3]:  # Top 3 only
            preview = ctx.text[:100].replace("\n", " ")
            summaries.append(f"[{ctx.source}] {preview}...")
        return "\n".join(summaries)


# ──────────────────────────────────────────────
#  REASONING ENGINE
# ──────────────────────────────────────────────

class ReasoningEngine:
    """
    Main reasoning engine — orchestrates the full cognitive pipeline.

    Flow:
        Question → Analyze → Explore Memory → Build Chain → Synthesize → Answer

    This engine is designed to be:
    - Modular: each stage can be swapped or extended
    - Transparent: every reasoning step is recorded
    - Memory-aware: answers are grounded in retrieved knowledge
    - Agent-ready: the pipeline can be called autonomously
    """

    def __init__(
        self,
        memory: Optional[MemoryContext] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        model: str = "llama3",
    ):
        self.memory = memory or MemoryContext()
        self.prompt_builder = prompt_builder or PromptBuilder(model=model)
        self.model = self.prompt_builder.model

        # Pipeline stages (created per call for clean state)
        self._analyzer = QuestionAnalyzer()
        self._explorer = MemoryExplorer(self.memory)
        self._chain_builder = ReasoningChainBuilder()

        # Track total reasoning operations
        self._total_reasoned = 0

    # ──────────────────────────────────────────
    #  MAIN ENTRY POINT
    # ──────────────────────────────────────────

    def reason(
        self,
        question: str,
        use_llm: bool = False,
    ) -> ReasoningResult:
        """
        Full reasoning pipeline.

        Args:
            question: The user's question
            use_llm:  If True, requires LLM for answer generation.
                      If False, builds a rule-based answer from context.

        Returns:
            ReasoningResult with answer, steps, and metadata
        """
        if not question or not question.strip():
            return self._empty_result(question)

        self._total_reasoned += 1

        # Stage 1: Analyze
        analysis = self._analyzer.analyze(question)

        # Stage 2: Explore memory
        contexts, concept_links = self._explorer.explore(analysis)

        # Stage 3: Build reasoning chain
        reasoning_steps = self._chain_builder.build(
            analysis, contexts, concept_links
        )

        # Stage 4: Generate answer
        if use_llm:
            answer = self._generate_llm_answer(
                question, analysis, contexts, concept_links
            )
        else:
            answer = self._generate_context_answer(
                question, analysis, contexts, concept_links
            )

        # Calculate confidence
        confidence = self._calculate_confidence(
            contexts, concept_links, analysis
        )

        return ReasoningResult(
            answer=answer,
            question=question,
            analysis=analysis,
            reasoning_steps=reasoning_steps,
            concept_links=concept_links,
            contexts_used=contexts,
            confidence=confidence,
        )

    # ──────────────────────────────────────────
    #  ANSWER GENERATION
    # ──────────────────────────────────────────

    def _generate_context_answer(
        self,
        question: str,
        analysis: QuestionAnalysis,
        contexts: list[ContextItem],
        concept_links: list[ReasoningConceptLink],
    ) -> str:
        """
        Generate an answer using only retrieved context (no LLM).

        Useful for testing, debugging, or when LLM is unavailable.
        """
        if not contexts:
            return (
                f"I don't have specific information about '{question}' "
                f"in my memory. This topic might not be covered in my "
                f"knowledge base yet."
            )

        # Build answer from top contexts
        parts = []

        # Opening based on intent
        if analysis.intent == "explain":
            parts.append(f"Based on my memory, here's what I know:\n")
        elif analysis.intent == "how-to":
            parts.append(f"Here's how it works based on stored knowledge:\n")
        elif analysis.intent == "compare":
            parts.append(f"Based on my knowledge, here's the comparison:\n")
        elif analysis.intent == "why":
            parts.append(f"Here's what my memory suggests about why:\n")
        else:
            parts.append(f"Based on relevant context:\n")

        # Add content from contexts
        for i, ctx in enumerate(contexts[:3], 1):
            parts.append(f"\n**From {ctx.source}** (relevance: {ctx.score:.0%}):")
            parts.append(ctx.text)

        # Add concept links if found
        if concept_links:
            parts.append("\n**Related concepts found in memory:**")
            for link in concept_links:
                parts.append(
                    f"- {link.source_concept} ↔ {link.related_concept} "
                    f"(strength: {link.connection_strength:.0%})"
                )

        return "\n".join(parts)

    def _generate_llm_answer(
        self,
        question: str,
        analysis: QuestionAnalysis,
        contexts: list[ContextItem],
        concept_links: list[ReasoningConceptLink],
    ) -> str:
        """
        Generate answer using LLM with built prompt.

        This requires Ollama to be running.
        """
        # Build prompt with reasoning context
        pb = self.prompt_builder
        pb.reset()

        # Add retrieved contexts
        for ctx in contexts[:5]:
            pb.add_memory(
                text=ctx.text,
                source=ctx.source,
                score=ctx.score,
            )

        # Add concept links as extra instruction
        if concept_links:
            link_text = "Related concepts in memory:\n"
            for link in concept_links:
                link_text += (
                    f"- {link.source_concept} is related to "
                    f"{link.related_concept} "
                    f"(based on: {link.evidence_text[:100]}...)\n"
                )
            pb.add_instruction(link_text.strip())

        # Build chain and invoke
        prompt = pb.build(question)
        values = pb.build_with_values(question)

        try:
            from langchain_ollama.llms import OllamaLLM
            llm = OllamaLLM(model=self.model, temperature=0.7)
            chain = prompt | llm
            response = chain.invoke(values)
            return response
        except Exception as e:
            return (
                f"⚠️ LLM unavailable: {e}\n\n"
                f"{self._generate_context_answer(question, analysis, contexts, concept_links)}"
            )

    # ──────────────────────────────────────────
    #  CONFIDENCE
    # ──────────────────────────────────────────

    def _calculate_confidence(
        self,
        contexts: list[ContextItem],
        concept_links: list[ReasoningConceptLink],
        analysis: QuestionAnalysis,
    ) -> float:
        """
        Estimate confidence in the answer.

        Factors:
        - Number and quality of retrieved contexts
        - Strength of concept links
        - Question complexity vs available knowledge
        """
        confidence = 0.0

        # Context coverage (0.0 – 0.5)
        if contexts:
            avg_score = sum(c.score for c in contexts) / len(contexts)
            avg_quality = sum(c.quality for c in contexts) / len(contexts)
            context_factor = (avg_score * 0.6 + avg_quality * 0.4)
            # More contexts = more confidence, up to a point
            context_count_factor = min(1.0, len(contexts) / 5)
            confidence += context_factor * context_count_factor * 0.5

        # Concept links add confidence (0.0 – 0.3)
        if concept_links:
            avg_link_strength = sum(
                l.connection_strength for l in concept_links
            ) / len(concept_links)
            confidence += avg_link_strength * 0.3

        # Intent match (0.0 – 0.2)
        if analysis.needs_context and contexts:
            confidence += 0.2
        elif not analysis.needs_context:
            confidence += 0.1

        return round(min(1.0, confidence), 3)

    def _empty_result(self, question: str) -> ReasoningResult:
        """Return empty result for invalid questions."""
        return ReasoningResult(
            answer="Please provide a valid question.",
            question=question,
            analysis=QuestionAnalysis(raw=question),
            confidence=0.0,
        )

    # ──────────────────────────────────────────
    #  CHAIN METHOD (LangChain compatible)
    # ──────────────────────────────────────────

    def reason_chain(
        self,
        question: str,
        history: Optional[list[dict]] = None,
    ) -> ReasoningResult:
        """
        Reason with conversation history for continuity.

        Args:
            question: Current question
            history:  List of {"role": "user"/"ai", "content": "..."}

        Returns:
            ReasoningResult
        """
        # Inject history into prompt builder
        if history:
            self.prompt_builder.set_history(history)

        return self.reason(question)

    # ──────────────────────────────────────────
    #  STATS
    # ──────────────────────────────────────────

    def get_stats(self) -> dict:
        """Engine statistics."""
        return {
            "total_reasoning_operations": self._total_reasoned,
            "memory_available": self.memory.is_available(),
            "memory_documents": self.memory.count(),
        }

    def print_reasoning_trace(self, result: ReasoningResult):
        """Pretty-print the full reasoning trace."""
        print("\n" + "=" * 60)
        print(f"  🧠 REASONING TRACE")
        print("=" * 60)

        print(f"\n📝 Question: {result.question}")
        print(f"   Intent: {result.analysis.intent}")
        print(f"   Complexity: {result.analysis.complexity}")
        print(f"   Key terms: {', '.join(result.analysis.key_terms[:5])}")

        print(f"\n📋 Reasoning Steps:")
        for step in result.reasoning_steps:
            icon = {
                "analyze": "🔍",
                "retrieve": "📚",
                "connect": "🔗",
                "synthesize": "🧩",
            }.get(step.step_type, "•")
            print(f"   {icon} Step {step.step_number} [{step.step_type}]")
            print(f"      {step.description}")
            if step.evidence:
                for line in step.evidence.split("\n"):
                    print(f"      → {line}")

        if result.concept_links:
            print(f"\n🔗 Concept Links ({len(result.concept_links)}):")
            for link in result.concept_links:
                print(f"   {link.source_concept} ↔ {link.related_concept} "
                      f"({link.connection_strength:.0%})")

        print(f"\n📊 Contexts used: {len(result.contexts_used)}")
        print(f"   Confidence: {result.confidence:.0%}")

        print(f"\n💡 Answer:")
        print(f"   {result.answer[:300]}{'...' if len(result.answer) > 300 else ''}")
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
    print("  REASONING ENGINE — Quick Test")
    print("=" * 60 + "\n")

    engine = ReasoningEngine()
    print(f"Engine stats: {engine.get_stats()}\n")

    questions = [
        "Apa itu embeddings?",
        "Bagaimana semantic search bekerja?",
        "Apa hubungan antara embeddings dan vector database?",
    ]

    for q in questions:
        result = engine.reason(q)
        engine.print_reasoning_trace(result)
        print("-" * 60)
