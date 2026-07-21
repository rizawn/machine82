"""
VECTOR STORE — Long Term Memory untuk AI

Flow:
    Markdown → Loader → Text → Embedding Model → Vectors → Vector Database

Module ini menyediakan:
    - Ingest markdown files ke vector database (ChromaDB)
    - Simpan embeddings ke workspace/embeddings/
    - Semantic search berdasarkan makna, bukan keyword
"""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

# Path default — relatif terhadap root project MLRL02
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "workspace", "embeddings")
MARKDOWN_DIR = os.path.join(BASE_DIR, "workspace", "markdown")

# Model embedding — all-MiniLM-L6-v2 ringan & cukup akurat
DEFAULT_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mlrl_memory"


# ──────────────────────────────────────────────
#  VECTOR STORE CLASS
# ──────────────────────────────────────────────

class VectorStore:
    """
    Vector Store — otak long-term memory AI.

    Simpan knowledge sebagai embedding vectors di ChromaDB.
    Bisa dicari secara semantic (berdasarkan makna).
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        collection_name: str = COLLECTION_NAME,
        client: Optional[chromadb.ClientAPI] = None,
    ):
        self.persist_dir = persist_dir or EMBEDDINGS_DIR
        self.collection_name = collection_name

        # Pastikan folder persist ada
        os.makedirs(self.persist_dir, exist_ok=True)

        # ── Load Embedding Model ──
        print(f"[VectorStore] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # ── Init ChromaDB (persistent) ──
        if client:
            self.client = client
        else:
            print(f"[VectorStore] Initializing ChromaDB at: {self.persist_dir}")
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        print(f"[VectorStore] Collection '{self.collection_name}' ready — "
              f"{self.collection.count()} documents stored.")

    # ──────────────────────────────────────────
    #  TEXT → EMBEDDING
    # ──────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Ubah text jadi vector embedding."""
        return self.model.encode(text).tolist()

    # ──────────────────────────────────────────
    #  ADD DOCUMENTS
    # ──────────────────────────────────────────

    def add_document(self, doc_id: str, text: str, metadata: Optional[dict] = None):
        """
        Simpan satu dokumen ke vector database.

        Args:
            doc_id:   ID unik dokumen
            text:     Isi teks dokumen
            metadata: Info tambahan (misal: source file, topic)
        """
        embedding = self.embed_text(text)
        meta = metadata or {}

        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )
        print(f"[VectorStore] Added/updated doc: {doc_id}")

    def add_documents(self, documents: list[dict]):
        """
        Batch add dokumen.

        Args:
            documents: List of dict dengan keys: 'id', 'text', 'metadata' (optional)
        """
        if not documents:
            return

        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        embeddings = [self.embed_text(t) for t in texts]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"[VectorStore] Added/updated {len(documents)} documents.")

    # ──────────────────────────────────────────
    #  INGEST MARKDOWN FILES
    # ──────────────────────────────────────────

    def ingest_markdown(self, markdown_dir: Optional[str] = None):
        """
        Baca semua .md files via loader, lalu simpan ke vector DB.

        Loader menangani:
            - Recursive file discovery
            - Frontmatter extraction
            - Header-based chunking
            - Text sanitization

        Args:
            markdown_dir: Folder berisi file markdown
        """
        # Lazy import untuk hindari circular import
        from core.memory.loader import load_markdown

        md_dir = markdown_dir or MARKDOWN_DIR
        docs = load_markdown(md_dir)

        if not docs:
            print(f"[VectorStore] No markdown files found in: {md_dir}")
            return

        print(f"[VectorStore] Found {len(docs)} document(s) to ingest via loader.")

        documents = []
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            # Deterministic ID: hash of source + content for stable, unique IDs
            import hashlib
            content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()[:12]
            doc_id = f"{source}__{content_hash}"
            documents.append({
                "id": doc_id,
                "text": doc.page_content,
                "metadata": doc.metadata,
            })

        self.add_documents(documents)
        print(f"[VectorStore] Ingestion complete — {len(documents)} chunks stored.")

    # ──────────────────────────────────────────
    #  SEMANTIC SEARCH (SIMILARITY SEARCH)
    # ──────────────────────────────────────────

    def similarity_search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Cari dokumen berdasarkan MAKNA (semantic similarity).

        "bagaimana AI menyimpan knowledge?" 
        → bisa nemu "Vector databases store embeddings."

        Args:
            query:     Pertanyaan / query text
            n_results: Jumlah hasil yang dikembalikan

        Returns:
            List of dict: {'id', 'text', 'metadata', 'distance'}
        """
        query_embedding = self.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        # Format output
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })

        return output

    def delete_document(self, doc_id: str):
        """Hapus dokumen berdasarkan ID."""
        self.collection.delete(ids=[doc_id])
        print(f"[VectorStore] Deleted doc: {doc_id}")

    def reset(self):
        """Hapus semua data di collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VectorStore] Collection '{self.collection_name}' has been reset.")

    # ──────────────────────────────────────────
    #  STATS
    # ──────────────────────────────────────────

    def count(self) -> int:
        """Jumlah dokumen di collection."""
        return self.collection.count()

    def info(self) -> dict:
        """Info tentang vector store."""
        return {
            "collection": self.collection_name,
            "document_count": self.count(),
            "persist_dir": self.persist_dir,
            "model": DEFAULT_MODEL,
        }



# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VECTOR STORE — Quick Test")
    print("=" * 60)

    store = VectorStore()

    # Add sample documents
    store.add_document("test_1", "Vector databases store embeddings for semantic search.")
    store.add_document("test_2", "AI agents use memory to remember past interactions.")
    store.add_document("test_3", "Reinforcement learning rewards good behavior.")

    # Semantic search
    print("\n🔍 Query: 'bagaimana AI menyimpan knowledge?'")
    results = store.similarity_search("bagaimana AI menyimpan knowledge?", n_results=3)

    for r in results:
        print(f"  [{r['distance']:.4f}] {r['text']}")

    print(f"\n📊 Total documents: {store.count()}")
    print(f"📂 Info: {store.info()}")
