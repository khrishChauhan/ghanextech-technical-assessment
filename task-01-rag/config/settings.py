"""Centralized application settings and configuration management."""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load variables from .env file if available
load_dotenv()


@dataclass
class Settings:
    """Application settings with runtime overrides and validation."""

    app_name: str = "Ghanex DocAssist RAG"
    app_version: str = "1.0.0"

    # LLM Settings
    default_llm_model: str = "gemini-2.0-flash"
    available_llm_models: List[str] = field(
        default_factory=lambda: [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
    )

    # Embedding Settings
    default_embedding_model: str = "models/text-embedding-004"
    embedding_dimension: int = 768

    # Chunking Defaults
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    min_chunk_size: int = 200
    max_chunk_size: int = 4000

    # Retrieval Defaults
    default_top_k: int = 4
    default_temperature: float = 0.2
    max_top_k: int = 10

    # Ingestion constraints
    max_file_size_mb: int = 25
    allowed_extensions: List[str] = field(default_factory=lambda: [".pdf"])

    # API Keys
    api_key_env_var: str = "GEMINI_API_KEY"
    alternate_api_key_env_var: str = "GOOGLE_API_KEY"

    def get_api_key(self, user_override: Optional[str] = None) -> Optional[str]:
        """Resolve Gemini API key in prioritized order:
        1. User UI override (explicitly provided)
        2. Environment variable GEMINI_API_KEY
        3. Environment variable GOOGLE_API_KEY
        4. Streamlit secrets (if available in Streamlit runtime)
        """
        if user_override and user_override.strip():
            return user_override.strip()

        env_key = os.getenv(self.api_key_env_var) or os.getenv(self.alternate_api_key_env_var)
        if env_key and env_key.strip():
            return env_key.strip()

        # Try Streamlit secrets if running inside streamlit
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                if self.api_key_env_var in st.secrets:
                    return str(st.secrets[self.api_key_env_var]).strip()
                if self.alternate_api_key_env_var in st.secrets:
                    return str(st.secrets[self.alternate_api_key_env_var]).strip()
        except Exception:
            pass

        return None


def get_settings() -> Settings:
    """Factory function for retrieving default settings."""
    return Settings()
