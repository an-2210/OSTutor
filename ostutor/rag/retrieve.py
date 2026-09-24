"""
FAISS Vector Retrieval Module for OSTutorLLM RAG Pipeline.

Builds, persists, loads, and queries FAISS vector indexes (IndexFlatIP with L2-normalized vectors)
for cosine similarity search over OS context chunks with optional metadata filtering.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from config import CHUNKS_METADATA_PATH, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, INDEX_DIR
from rag.embed import embed_query

logger = logging.getLogger(__name__)


class VectorStoreIndex:
    """
    FAISS Vector Store wrapper linking vector index positions to chunk metadata records.
    """

    def __init__(self, index: Any, chunks: List[Dict[str, Any]]):
        """
        Initialize vector store index instance.

        Args:
            index: FAISS index object (IndexFlatIP).
            chunks: List of chunk metadata records corresponding 1:1 with vector rows.
        """
        self.index = index
        self.chunks = chunks

    @property
    def total_vectors(self) -> int:
        """Total number of vectors in the FAISS index."""
        return self.index.ntotal if self.index is not None else 0

    @property
    def vector_dimension(self) -> int:
        """Dimension of vectors in the index."""
        return self.index.d if self.index is not None else 0


def build_index(chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> VectorStoreIndex:
    """
    Build a FAISS IndexFlatIP index over L2-normalized embedding vectors.

    Args:
        chunks: List of chunk records.
        embeddings: 2D float32 NumPy array of shape (N, D).

    Returns:
        VectorStoreIndex instance.
    """
    if faiss is None:
        raise ImportError("faiss-cpu package is required. Install via `pip install faiss-cpu`.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})."
        )

    if len(embeddings) == 0:
        logger.warning("Empty embeddings array provided to build_index.")
        index = faiss.IndexFlatIP(384)
        return VectorStoreIndex(index, [])

    num_vectors, dim = embeddings.shape

    # Ensure embeddings are float32 contiguous array
    embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

    # IndexFlatIP computes inner product. With L2-normalized vectors, IP = Cosine Similarity.
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings_f32)

    logger.info(f"Built FAISS IndexFlatIP index containing {faiss_index.ntotal} vectors of dimension {dim}.")
    return VectorStoreIndex(faiss_index, chunks)


def save_index(
    vector_store: VectorStoreIndex,
    index_dir: Path = INDEX_DIR,
    faiss_filename: str = "faiss.index",
    chunks_filename: str = "chunks.json",
) -> Tuple[Path, Path]:
    """
    Save FAISS vector index binary and chunk metadata JSON to index_dir.

    Args:
        vector_store: VectorStoreIndex instance.
        index_dir: Directory path to save artifacts.
        faiss_filename: Filename for FAISS binary index.
        chunks_filename: Filename for metadata JSON.

    Returns:
        Tuple of (saved_faiss_path, saved_chunks_path).
    """
    if faiss is None:
        raise ImportError("faiss-cpu package is required.")

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = index_dir / faiss_filename
    chunks_path = index_dir / chunks_filename

    # Save FAISS binary index
    faiss.write_index(vector_store.index, str(faiss_path))

    # Save metadata records
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(vector_store.chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved FAISS index to {faiss_path} and metadata to {chunks_path}.")
    return faiss_path, chunks_path


def load_index(
    index_dir: Path = INDEX_DIR,
    faiss_filename: str = "faiss.index",
    chunks_filename: str = "chunks.json",
) -> VectorStoreIndex:
    """
    Load FAISS vector index binary and chunk metadata JSON from index_dir.

    Args:
        index_dir: Directory path containing index artifacts.
        faiss_filename: Filename for FAISS binary index.
        chunks_filename: Filename for metadata JSON.

    Returns:
        VectorStoreIndex instance.
    """
    if faiss is None:
        raise ImportError("faiss-cpu package is required.")

    faiss_path = index_dir / faiss_filename
    chunks_path = index_dir / chunks_filename

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index file missing at: {faiss_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Metadata JSON file missing at: {chunks_path}")

    index = faiss.read_index(str(faiss_path))

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded FAISS index ({index.ntotal} vectors) and metadata ({len(chunks)} chunks) from {index_dir}.")
    return VectorStoreIndex(index, chunks)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    unit: Optional[int] = None,
    topic: Optional[str] = None,
    index_dir: Path = INDEX_DIR,
    vector_store: Optional[VectorStoreIndex] = None,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> List[Dict[str, Any]]:
    """
    Query the vector store and return ranked relevant OS context chunks.

    Args:
        query: Natural language query string.
        top_k: Number of top relevant chunks to return (default: 5).
        unit: Optional integer OS unit filter (1-7).
        topic: Optional topic string filter.
        index_dir: Directory path containing saved index.
        vector_store: Pre-loaded VectorStoreIndex instance (optional).
        model_name: Embedding model identifier.

    Returns:
        List of result dictionaries containing rank, score, chunk_id, metadata, and text.
    """
    if not query.strip():
        return []

    if vector_store is None:
        vector_store = load_index(index_dir)

    total_chunks = vector_store.total_vectors
    if total_chunks == 0:
        logger.warning("Retriever index is empty.")
        return []

    # Embed query vector
    query_vector = embed_query(query, model_name=model_name)
    query_matrix = np.ascontiguousarray([query_vector], dtype=np.float32)

    # Fetch candidate pool size (larger if filtering is active)
    fetch_k = min(total_chunks, max(top_k * 10, 50)) if (unit is not None or topic is not None) else min(top_k, total_chunks)

    # Search FAISS index
    scores, indices = vector_store.index.search(query_matrix, fetch_k)

    raw_scores = scores[0]
    raw_indices = indices[0]

    results: List[Dict[str, Any]] = []
    rank = 1

    for score, idx in zip(raw_scores, raw_indices):
        if idx < 0 or idx >= len(vector_store.chunks):
            continue

        chunk_meta = vector_store.chunks[idx]

        # Apply metadata filtering
        if unit is not None and chunk_meta.get("unit") != unit:
            continue
        if topic is not None and topic.lower() not in chunk_meta.get("topic", "").lower():
            continue

        item = {
            "rank": rank,
            "score": round(float(score), 4),
            "chunk_id": chunk_meta.get("chunk_id"),
            "unit": chunk_meta.get("unit"),
            "topic": chunk_meta.get("topic"),
            "subtopic": chunk_meta.get("subtopic"),
            "source": chunk_meta.get("filename") or chunk_meta.get("source"),
            "page": chunk_meta.get("page"),
            "slide": chunk_meta.get("slide"),
            "text": chunk_meta.get("text"),
        }
        results.append(item)
        rank += 1

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing VectorStoreIndex wrapper...")
    dummy_chunks = [
        {"chunk_id": "c1", "unit": 3, "topic": "CPU Scheduling", "text": "Round Robin scheduling algorithm"},
        {"chunk_id": "c2", "unit": 4, "topic": "Semaphores", "text": "Binary semaphore for mutual exclusion"},
    ]
    dummy_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    vstore = build_index(dummy_chunks, dummy_vecs)
    print(f"Index built: {vstore.total_vectors} vectors.")
