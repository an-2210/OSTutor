"""
Embedding Generator Module for OSTutorLLM RAG Pipeline.

Uses SentenceTransformers to generate dense vector embeddings for OS context chunks.
Supports CPU and Apple Silicon (MPS) acceleration, batch processing, L2 normalization,
and deterministic inference.
"""

import logging
from typing import List, Optional, Union
import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    torch = None

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


def get_device() -> str:
    """
    Detect optimal available computing device (CUDA, MPS, or CPU).

    Returns:
        Device name string ('cuda', 'mps', or 'cpu').
    """
    if torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


class EmbeddingManager:
    """
    Singleton / cached manager for loading and running SentenceTransformers models.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the embedding manager with model name.

        Args:
            model_name: HuggingFace model path or identifier.
        """
        self.model_name = model_name
        self.device = get_device()
        self._model: Optional[Any] = None

    def load_model(self) -> Any:
        """
        Lazy load the SentenceTransformer model onto the target device.

        Returns:
            SentenceTransformer model instance.
        """
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers package is required. Install via `pip install sentence-transformers`."
                )
            logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            dim_func = getattr(self._model, "get_embedding_dimension", None) or getattr(self._model, "get_sentence_embedding_dimension")
            logger.info(f"Model '{self.model_name}' loaded successfully. Embedding dimension: {dim_func()}")
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding vector dimension."""
        model = self.load_model()
        dim_func = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension")
        return dim_func()

    def embed_texts(
        self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> np.ndarray:
        """
        Generate L2-normalized embedding vectors for a list of text strings.

        Args:
            texts: List of text content strings.
            batch_size: Batch size for model inference.

        Returns:
            2D NumPy array of shape (N, D) and dtype float32.
        """
        if not texts:
            dim = self.embedding_dimension
            return np.empty((0, dim), dtype=np.float32)

        model = self.load_model()
        # Encode with normalize_embeddings=True so inner product equals cosine similarity
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate L2-normalized embedding vector for a single query string.

        Args:
            query: Natural language query string.

        Returns:
            1D NumPy array of shape (D,) and dtype float32.
        """
        if not query.strip():
            dim = self.embedding_dimension
            return np.zeros(dim, dtype=np.float32)

        embeddings = self.embed_texts([query], batch_size=1)
        return embeddings[0]


# Global default embedder instance
_default_manager: Optional[EmbeddingManager] = None


def get_embedder(model_name: str = EMBEDDING_MODEL_NAME) -> EmbeddingManager:
    """Get or create singleton EmbeddingManager instance."""
    global _default_manager
    if _default_manager is None or _default_manager.model_name != model_name:
        _default_manager = EmbeddingManager(model_name=model_name)
    return _default_manager


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> Any:
    """Load and return sentence transformer embedding model instance."""
    manager = get_embedder(model_name)
    return manager.load_model()


def embed_texts(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """Convenience function to embed a list of texts."""
    manager = get_embedder(model_name)
    return manager.embed_texts(texts, batch_size=batch_size)


def embed_query(query: str, model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Convenience function to embed a query string."""
    manager = get_embedder(model_name)
    return manager.embed_query(query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing embedding generator...")
    sample_texts = [
        "Operating systems manage CPU scheduling and memory allocation.",
        "Round Robin uses a time quantum to preempt processes.",
    ]
    vecs = embed_texts(sample_texts)
    print("Embedded vectors shape:", vecs.shape)
    print("Embedding norm:", np.linalg.norm(vecs[0]))
