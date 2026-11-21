"""Vector storage subpackage."""
from src.vectorstore.faiss_store import FAISSVectorStore, SearchResult

__all__ = ["FAISSVectorStore", "SearchResult"]
