"""RAG Milvus Tool for CrewAI - Semantic search across internal knowledge base"""
import os
import json
import requests
from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from pymilvus import MilvusClient
from rapidfuzz import fuzz

from src.config.settings import get_settings
from src.config.constants import (
    RAG_COLLECTION_DISPLAY_NAMES,
    RAG_COLLECTION_INTERNAL_NAMES,
    DEFAULT_RAG_TOP_K,
    DEFAULT_EMBEDDING_TIMEOUT
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGMilvusToolSchema(BaseModel):
    """Input schema for RAGMilvusTool."""
    query: str = Field(..., description="Search query to find relevant information in the knowledge base")


class RAGMilvusTool(BaseTool):
    """
    Tool for searching internal knowledge base using semantic search.

    Uses Milvus vector database for similarity search across multiple collections
    including user income, DGE, Genie, and other internal documentation.
    """

    name: str = "Internal Knowledge Base Search"
    description: str = (
        "Searches the internal knowledge base using semantic search. "
        "Useful for finding relevant information about user income, user occupation, "
        "DGE, Genie, push notifications, and pills. "
        "Input should be a natural language query about any of these topics."
    )
    args_schema: Type[BaseModel] = RAGMilvusToolSchema

    # Pydantic fields for configuration
    db_path: str = ""
    model_name: str = ""
    embedding_endpoint: str = ""
    top_k: int = DEFAULT_RAG_TOP_K

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        db_path: str = None,
        model_name: str = None,
        embedding_endpoint: str = None,
        top_k: int = None,
        **kwargs
    ):
        """
        Initialize RAG Milvus tool.

        Args:
            db_path: Path to Milvus database file (optional, reads from settings)
            model_name: Embedding model name (optional, reads from settings)
            embedding_endpoint: URL for embedding generation (optional, reads from settings)
            top_k: Number of top results to return (optional, default from settings)
        """
        settings = get_settings()

        # Use provided values or fallback to settings
        super().__init__(
            db_path=db_path or settings.rag.db_path,
            model_name=model_name or settings.rag.embedding_model,
            embedding_endpoint=embedding_endpoint or settings.rag.embedding_endpoint,
            top_k=top_k or settings.rag.top_k,
            **kwargs
        )

        # Create class mapping dictionary (store as instance attribute)
        object.__setattr__(
            self,
            '_class_mapping_dict',
            dict(zip(RAG_COLLECTION_DISPLAY_NAMES, RAG_COLLECTION_INTERNAL_NAMES))
        )
        object.__setattr__(self, '_initialized', False)
        object.__setattr__(self, '_client', None)

        # Initialize Milvus client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Milvus client and verify database exists."""
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"Milvus database not found at {self.db_path}")
                return

            client = MilvusClient(self.db_path)
            object.__setattr__(self, '_client', client)
            logger.info(f"RAG Milvus client initialized with database: {self.db_path}")
            object.__setattr__(self, '_initialized', True)
        except Exception as e:
            logger.error(f"Failed to initialize Milvus client: {e}")

    def is_available(self) -> bool:
        """Check if RAG tool is available."""
        return getattr(self, '_initialized', False)

    def _mapping_group(self, string_to_check: str) -> str:
        """
        Map query to the most relevant collection using fuzzy matching.

        Args:
            string_to_check: Query string to map

        Returns:
            Collection name in lowercase format
        """
        query = string_to_check.upper()
        query_words = query.split()
        final_results = []

        for item in RAG_COLLECTION_DISPLAY_NAMES:
            # Compute partial_ratio for full query
            full_score = fuzz.partial_ratio(query, item)

            # Compute score for each word in query, take the highest
            word_scores = [fuzz.partial_ratio(word, item) for word in query_words]
            max_word_score = max(word_scores) if word_scores else 0

            # Take the higher of full_score vs max_word_score
            final_score = max(full_score, max_word_score)
            final_results.append((item, final_score))

        # Sort results by score descending
        final_results.sort(key=lambda x: x[1], reverse=True)

        # Return the mapped collection name
        top_item = final_results[0][0]
        class_mapping = getattr(self, '_class_mapping_dict', {})
        mapped_name = class_mapping[top_item]

        logger.debug(f"Mapped query to collection: {mapped_name} (score: {final_results[0][1]})")
        return mapped_name

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            RuntimeError: If embedding generation fails
        """
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float"
        }
        headers = {"Content-Type": "application/json"}

        try:
            logger.debug(f"Generating embedding for query: {text[:50]}...")
            response = requests.post(
                self.embedding_endpoint,
                json=payload,
                headers=headers,
                timeout=DEFAULT_EMBEDDING_TIMEOUT
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            logger.debug(f"Successfully generated embedding: {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            error_msg = f"Failed to generate embedding: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _run(self, query: str) -> str:
        """
        Search the knowledge base for relevant information.

        Args:
            query: Natural language search query

        Returns:
            JSON string with search results
        """
        if not self.is_available():
            error_msg = "RAG Milvus tool not available. Check database path and initialization."
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        try:
            logger.info(f"Searching knowledge base for: {query}")

            # Map query to collection
            collection_name = self._mapping_group(query)
            logger.info(f"Searching collection: {collection_name}")

            # Generate embedding for query
            query_vector = self._generate_embedding(query)

            # Search Milvus
            client = getattr(self, '_client', None)
            if not client:
                raise RuntimeError("Milvus client not initialized")

            search_results = client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=self.top_k,
                output_fields=["text"]
            )

            # Format results
            results = []
            for hit in search_results[0]:
                results.append({
                    "score": float(hit["distance"]),
                    "text": hit["entity"]["text"],
                    "collection": collection_name
                })

            logger.info(f"Found {len(results)} results in collection {collection_name}")
            return json.dumps({
                "query": query,
                "collection_searched": collection_name,
                "results_count": len(results),
                "results": results
            }, indent=2)

        except Exception as e:
            error_msg = f"Error searching knowledge base: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({
                "error": error_msg,
                "query": query
            })
