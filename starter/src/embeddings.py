"""EmbeddingGemma 2 adapter for local document/query embeddings."""
from typing import Protocol

from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Use a real Sentence Transformers model with task-specific prompts."""

    def __init__(self, model_name: str, truncate_dim: int | None = None) -> None:
        self.model_name = model_name
        self.truncate_dim = truncate_dim
        self.model = SentenceTransformer(
            model_name,
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        native_dimensions = self.model.get_embedding_dimension()
        self.dimensions = truncate_dim or native_dimensions
        if self.dimensions not in {128, 256, 512, 768}:
            raise ValueError(f"Unsupported embedding dimension: {self.dimensions}")

    def _encode(self, texts: list[str], prompt_name: str) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            prompt_name=prompt_name,
            normalize_embeddings=True,
            truncate_dim=self.truncate_dim,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "Document")

    def embed_query(self, question: str, prompt_name: str = "QuestionAnswering") -> list[float]:
        return self._encode([question], prompt_name)[0]
