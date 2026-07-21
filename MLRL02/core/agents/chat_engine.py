"""
CHAT ENGINE — LangChain + ChromaDB + Ollama LLM

Flow:
    User Question → Embed → ChromaDB Search → Retrieve Context
    → Build Prompt (context + history + question)
    → Ollama (llama3) → Formatted Response

Architecture:
    This module bridges semantic memory (ChromaDB) with a local LLM (Ollama).
    It maintains conversation history so the AI can reference past turns.

Usage:
    engine = ChatEngine()
    response = engine.chat("Apa itu embeddings?")
    print(response)
"""

import os
from typing import Optional
from datetime import datetime

from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import chromadb


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PERSIST_DIR = os.path.join(BASE_DIR, "workspace", "embeddings")
COLLECTION_NAME = "mlrl_memory"

DEFAULT_MODEL = "llama3"
DEFAULT_TOP_K = 3  # Jumlah context dokumen yang diretriev
MAX_HISTORY_TURNS = 10  # Maximum conversation turns to keep

# System prompt — instruksi dasar untuk AI behavior
SYSTEM_PROMPT = """\
You are MLRL02, an AI assistant with long-term semantic memory.
You answer based on the retrieved context when available.
If context is irrelevant or empty, use your general knowledge.
Always be clear, concise, and helpful.

Current date: {date}
"""


# ──────────────────────────────────────────────
#  CHAT ENGINE
# ──────────────────────────────────────────────

class ChatEngine:
    """
    Main chat interface — menghubungkan memory + LLM.

    Setiap pertanyaan user akan:
    1. Dicari context serupa dari ChromaDB
    2. Context + history + question digabung jadi prompt
    3. Dikirim ke Ollama (llama3)
    4. Response dikembalikan dan history disimpan
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        top_k: int = DEFAULT_TOP_K,
        client=None,
    ):
        self.persist_dir = persist_dir or PERSIST_DIR
        self.model = model
        self.top_k = top_k
        self._client = client  # Shared ChromaDB client
        self.history: list[dict] = []  # [{"role": "user"/"ai", "content": "..."}]

        self._init_llm()
        self._init_vectorstore()
        self._init_prompt()

    # ──────────────────────────────────────────
    #  INIT
    # ──────────────────────────────────────────

    def _init_llm(self):
        """Setup Ollama LLM + embeddings."""
        print(f"[ChatEngine] Loading Ollama model: {self.model}")
        self.llm = OllamaLLM(
            model=self.model,
            temperature=0.7,
        )
        # Use HuggingFace embeddings to match VectorStore (384 dimensions)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _init_vectorstore(self):
        """Connect ke ChromaDB yang sudah ada."""
        print(f"[ChatEngine] Connecting to ChromaDB at: {self.persist_dir}")

        if not os.path.exists(self.persist_dir):
            print(f"[ChatEngine] ⚠️  Persist dir not found — search will return no results.")
            self.vectorstore = None
            return

        try:
            chroma_client = self._client or chromadb.PersistentClient(path=self.persist_dir)
            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                client=chroma_client,
            )
            count = self.vectorstore._collection.count()
            print(f"[ChatEngine] ChromaDB ready — {count} documents in memory.")
        except Exception as e:
            print(f"[ChatEngine] ⚠️  Failed to connect ChromaDB: {e}")
            self.vectorstore = None

    def _init_prompt(self):
        """Bangun prompt template dari system + context + history + question."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", """Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""),
        ])

    # ──────────────────────────────────────────
    #  RETRIEVE CONTEXT
    # ──────────────────────────────────────────

    def _retrieve_context(self, question: str) -> str:
        """
        Cari dokumen relevan dari ChromaDB.

        Returns:
            Formatted string dari dokumen yang ditemukan.
        """
        if not self.vectorstore:
            return ""

        try:
            docs = self.vectorstore.similarity_search(question, k=self.top_k)
        except Exception as e:
            print(f"[ChatEngine] ⚠️  Search error: {e}")
            return ""

        if not docs:
            return ""

        # Format context dengan source info
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[{i}] (source: {source})\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    # ──────────────────────────────────────────
    #  BUILD CHAIN
    # ──────────────────────────────────────────

    def _build_chain(self, question: str, context: str):
        """
        Rakit prompt + LLM jadi chain yang bisa di-execute.
        """
        # Convert internal history ke LangChain messages
        langchain_history = []
        for msg in self.history:
            if msg["role"] == "user":
                langchain_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                langchain_history.append(AIMessage(content=msg["content"]))

        chain = self.prompt | self.llm

        return chain.invoke({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "history": langchain_history,
            "context": context,
            "question": question,
        })

    # ──────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────

    def chat(self, question: str, use_memory: bool = True, external_context: str = None) -> str:
        """
        Main entry point — tanya sesuatu ke AI.

        Args:
            question: Pertanyaan user
            use_memory: Jika False, tidak mencari di vector memory.
            external_context: Optional additional context.

        Returns:
            AI response string
        """
        if not question or not question.strip():
            return "Please ask a question."

        # Step 1: Retrieve context dari memory
        context = ""
        if use_memory:
            context = self._retrieve_context(question)
            if context:
                print(f"[ChatEngine] Retrieved {self.top_k} context document(s).")
            else:
                print("[ChatEngine] No context found — answering from general knowledge.")
        else:
            print("[ChatEngine] Memory bypass — answering directly.")

        if external_context:
            if context:
                context += "\n\n" + external_context
            else:
                context = external_context

        # Step 2: Generate response
        try:
            response = self._build_chain(question, context)
        except Exception as e:
            return f"❌ Error generating response: {e}"

        # Step 3: Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "ai", "content": response})

        # Truncate history to prevent unbounded growth / token overflow
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            self.history = self.history[-(MAX_HISTORY_TURNS * 2):]

        return response

    def reset_history(self):
        """Hapus semua conversation history."""
        self.history.clear()
        print("[ChatEngine] Conversation history cleared.")

    def show_history(self):
        """Print conversation history."""
        if not self.history:
            print("[ChatEngine] No conversation history.")
            return

        print("\n" + "=" * 50)
        print("  CONVERSATION HISTORY")
        print("=" * 50)
        for i, msg in enumerate(self.history, 1):
            role = "🧑 User" if msg["role"] == "user" else "🤖 AI"
            preview = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
            print(f"\n[{i}] {role}:")
            print(f"    {preview}")
        print()

    def get_stats(self) -> dict:
        """Info tentang engine dan memory."""
        doc_count = 0
        if self.vectorstore:
            try:
                doc_count = self.vectorstore._collection.count()
            except Exception:
                pass

        return {
            "model": self.model,
            "top_k": self.top_k,
            "memory_documents": doc_count,
            "conversation_turns": len(self.history) // 2,
            "has_context": self.vectorstore is not None,
        }


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CHAT ENGINE — Quick Test")
    print("=" * 60 + "\n")

    engine = ChatEngine()

    # Test questions
    questions = [
        "Apa itu embeddings?",
        "Bagaimana cara kerja semantic search?",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"🧑 User: {q}")
        print(f"{'=' * 60}")
        response = engine.chat(q)
        print(f"\n🤖 AI: {response}")
        print()

    # Stats
    print("=" * 60)
    print("  STATS")
    print("=" * 60)
    for k, v in engine.get_stats().items():
        print(f"  {k}: {v}")

    engine.show_history()
