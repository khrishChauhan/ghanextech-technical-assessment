"""Google Gemini Embedding client with batching, normalization, and error handling."""

import time
from typing import List, Optional
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("embeddings")


class GeminiAPIKeyError(Exception):
    """Raised when Gemini API key is missing or invalid."""
    pass


class GeminiEmbeddingError(Exception):
    """Raised when the embedding generation encounters an API or network error."""
    pass


class GeminiEmbedder:
    """Wrapper for generating dense vector embeddings via Google Gemini API.
    
    Uses `models/text-embedding-004` (768 dimensions).
    Applies L2 normalization so that Inner Product equals Cosine Similarity.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/text-embedding-004",
        batch_size: int = 20,
        max_retries: int = 3,
        dimension: int = 768,
    ):
        self.api_key = api_key
        # Ensure model name is formatted properly for Google GenAI SDK
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.dimension = dimension
        self._client = None

        if self.api_key:
            self._init_client()

    def set_api_key(self, api_key: str):
        """Update or set the API key dynamically."""
        self.api_key = api_key
        self._init_client()

    def _init_client(self):
        """Initialize Google GenAI client."""
        if not self.api_key:
            raise GeminiAPIKeyError(
                "Gemini API key is required to initialize embedding client."
            )
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Google GenAI client with model '{self.model_name}'.")
        except ImportError:
            # Fallback check for google-generativeai
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self._client = legacy_genai
                logger.info("Initialized legacy google-generativeai client fallback.")
            except ImportError as e:
                raise ImportError(
                    f"Neither google-genai nor google-generativeai is installed: {e}"
                )

    def _ensure_client(self):
        """Ensure client is instantiated before making API calls."""
        if self._client is None:
            if not self.api_key:
                raise GeminiAPIKeyError(
                    "No Gemini API key provided. Please configure GEMINI_API_KEY in "
                    ".env or provide it through the sidebar."
                )
            self._init_client()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings in batches and return L2-normalized float32 vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension) with dtype float32.
        """
        self._ensure_client()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_embeddings: List[List[float]] = []

        # Batch texts to respect rate limits and request payload sizes
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_vectors)

        vectors = np.array(all_embeddings, dtype=np.float32)

        # L2-normalize vectors so that Dot Product / Inner Product equals Cosine Similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-12, norms)
        normalized_vectors = vectors / norms

        return normalized_vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query string and return an L2-normalized 1D or 2D vector.

        Args:
            query: User search query.

        Returns:
            numpy array of shape (1, dimension) with dtype float32.
        """
        self._ensure_client()
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        vectors = self.embed_texts([query.strip()])
        return vectors  # shape (1, dimension)

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """Call embedding API for a batch of texts with exponential backoff."""
        clean_batch = [t if t.strip() else " " for t in batch]

        # Model identifier normalization
        clean_model = self.model_name
        if clean_model.startswith("models/"):
            model_id = clean_model
            bare_model_id = clean_model.replace("models/", "")
        else:
            model_id = f"models/{clean_model}"
            bare_model_id = clean_model

        for attempt in range(1, self.max_retries + 1):
            try:
                # Check modern google.genai Client
                if hasattr(self._client, "models"):
                    # google-genai Client
                    response = self._client.models.embed_content(
                        model=bare_model_id,
                        contents=clean_batch,
                    )
                    # response.embeddings is a list of ContentEmbedding objects
                    embeddings = [e.values for e in response.embeddings]
                    return embeddings
                else:
                    # Legacy google.generativeai
                    result = self._client.embed_content(
                        model=model_id,
                        content=clean_batch,
                    )
                    return result["embedding"]
            except Exception as e:
                err_str = str(e)
                if "API_KEY_INVALID" in err_str or "unregistered" in err_str:
                    raise GeminiAPIKeyError(f"Invalid Google Gemini API key: {e}")

                logger.warning(
                    f"Embedding request attempt {attempt}/{self.max_retries} failed: {e}"
                )
                if attempt == self.max_retries:
                    raise GeminiEmbeddingError(
                        f"Failed to generate embeddings after {self.max_retries} attempts: {e}"
                    )
                time.sleep(2 ** attempt)

        raise GeminiEmbeddingError("Failed to embed batch.")
