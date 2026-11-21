"""End-to-end RAG Engine coordinating retrieval, prompt formatting, and LLM generation."""

from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple
import numpy as np

from config.settings import Settings, get_settings
from src.embeddings.gemini_embeddings import GeminiEmbedder
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt, format_context_block
from src.utils.logger import get_logger
from src.vectorstore.faiss_store import FAISSVectorStore, SearchResult

logger = get_logger("rag_engine")


@dataclass
class RAGResponse:
    """Encapsulates the generated answer and associated citation context."""
    answer: str
    query: str
    sources: List[SearchResult] = field(default_factory=list)
    model_used: str = ""
    context_text: str = ""


class RAGEngine:
    """Core RAG coordinator handling semantic retrieval and grounded generation."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: GeminiEmbedder,
        settings: Optional[Settings] = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.settings = settings or get_settings()

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """Perform query embedding and vector search.

        Args:
            query: User's natural language question.
            top_k: Number of chunks to retrieve.

        Returns:
            List of ranked SearchResult objects.
        """
        if self.vector_store.count == 0:
            logger.warning("Vector store is empty. No chunks to retrieve.")
            return []

        logger.info(f"Embedding query: '{query}'")
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        logger.info(f"Retrieved {len(results)} context chunks for query.")
        return results

    def generate_answer_stream(
        self,
        query: str,
        top_k: int = 4,
        temperature: float = 0.2,
        model_name: Optional[str] = None,
    ) -> Tuple[Generator[str, None, None], List[SearchResult]]:
        """Retrieve relevant context and stream the grounded answer token by token.

        Args:
            query: User query.
            top_k: Number of chunks to retrieve.
            temperature: LLM sampling temperature.
            model_name: Optional model override.

        Returns:
            Tuple of:
                - Generator yielding string chunks for streaming UI
                - List of SearchResult objects for citation display
        """
        model = model_name or self.settings.default_llm_model
        sources = self.retrieve(query, top_k=top_k)

        if not sources:
            def empty_stream():
                yield "I cannot answer this question because no documents have been uploaded or indexed yet. Please upload a PDF document first."
            return empty_stream(), []

        context_block = format_context_block(sources)
        user_prompt = build_user_prompt(query, context_block)

        api_key = self.embedder.api_key or self.settings.get_api_key()
        if not api_key:
            def missing_key_stream():
                yield "Error: Google Gemini API Key is missing. Please set GEMINI_API_KEY in .env or via the sidebar."
            return missing_key_stream(), sources

        stream_generator = self._stream_gemini_tokens(
            prompt=user_prompt,
            system_instruction=SYSTEM_PROMPT,
            model_name=model,
            temperature=temperature,
            api_key=api_key,
        )

        return stream_generator, sources

    def generate_answer(
        self,
        query: str,
        top_k: int = 4,
        temperature: float = 0.2,
        model_name: Optional[str] = None,
    ) -> RAGResponse:
        """Non-streaming generation returning complete RAGResponse."""
        model = model_name or self.settings.default_llm_model
        stream_gen, sources = self.generate_answer_stream(
            query=query,
            top_k=top_k,
            temperature=temperature,
            model_name=model,
        )

        collected_tokens = []
        for token in stream_gen:
            collected_tokens.append(token)

        full_answer = "".join(collected_tokens)
        context_block = format_context_block(sources)

        return RAGResponse(
            answer=full_answer,
            query=query,
            sources=sources,
            model_used=model,
            context_text=context_block,
        )

    def _stream_gemini_tokens(
        self,
        prompt: str,
        system_instruction: str,
        model_name: str,
        temperature: float,
        api_key: str,
    ) -> Generator[str, None, None]:
        """Low-level streaming generator using google-genai or google-generativeai."""
        # Clean model name (e.g. gemini-2.0-flash)
        clean_model = model_name.replace("models/", "")

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            )
            response = client.models.generate_content_stream(
                model=clean_model,
                contents=prompt,
                config=config,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except ImportError:
            # Fallback to legacy google.generativeai
            import google.generativeai as legacy_genai

            legacy_genai.configure(api_key=api_key)
            model = legacy_genai.GenerativeModel(
                model_name=clean_model,
                system_instruction=system_instruction,
            )
            response = model.generate_content(
                prompt,
                stream=True,
                generation_config={"temperature": temperature},
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"\n\n[Error generating response: {e}]"
