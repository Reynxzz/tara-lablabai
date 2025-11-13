"""RAG Milvus Tool for CrewAI - Semantic search across internal knowledge base"""
import os
import json
import requests
from typing import Any, Dict, Type, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from pymilvus import MilvusClient
from rapidfuzz import fuzz


class RAGMilvusToolSchema(BaseModel):
    """Input schema for RAGMilvusTool."""
    query: str = Field(..., description="Search query to find relevant information in the knowledge base")


class RAGMilvusTool(BaseTool):
    name: str = "Internal Knowledge Base Search"
    description: str = (
        "Searches the internal knowledge base using semantic search. "
        "Useful for finding relevant information about user income, user occupation, "
        "DGE, Genie, push notifications, and pills. "
        "Input should be a natural language query about any of these topics."
    )
    args_schema: Type[BaseModel] = RAGMilvusToolSchema

    # Pydantic fields for configuration
    db_path: str = "./milvus_demo_batch_bmth_v1.db"
    model_name: str = "google/embeddinggemma-300m"
    embedding_endpoint: str = "https://litellm-staging.gopay.sh/embeddings"
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    # Collection name mappings (class variables)
    col_name_list: List[str] = [
        "USER INCOME",
        "USER OCCUPATION",
        "DGE",
        "GENIE",
        "PN - PUSH NOTIFICATIONS",
        "PILLS"
    ]

    lower_col_name_list: List[str] = [
        "user_income",
        "user_occupation",
        "dge",
        "genie",
        "pn_push_notifications",
        "pills"
    ]

    def __init__(
        self,
        db_path: str = "./milvus_demo_batch_bmth_v1.db",
        model_name: str = "google/embeddinggemma-300m",
        embedding_endpoint: str = "https://litellm-staging.gopay.sh/embeddings",
        top_k: int = 5,
        **kwargs
    ):
        """
        Initialize RAG Milvus tool.

        Args:
            db_path: Path to the Milvus database file
            model_name: Embedding model name
            embedding_endpoint: URL for embedding generation
            top_k: Number of top results to return
        """
        super().__init__(
            db_path=db_path,
            model_name=model_name,
            embedding_endpoint=embedding_endpoint,
            top_k=top_k,
            **kwargs
        )

        # Create class mapping dictionary (store as instance attribute)
        object.__setattr__(self, '_class_mapping_dict', dict(zip(self.col_name_list, self.lower_col_name_list)))
        object.__setattr__(self, '_initialized', False)
        object.__setattr__(self, '_client', None)

        # Initialize Milvus client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Milvus client and verify database exists."""
        try:
            if not os.path.exists(self.db_path):
                print(f"Warning: Milvus database not found at {self.db_path}")
                object.__setattr__(self, '_initialized', False)
                return

            client = MilvusClient(self.db_path)
            object.__setattr__(self, '_client', client)
            print(f"✅ RAG Milvus client initialized with database: {self.db_path}")
            object.__setattr__(self, '_initialized', True)
        except Exception as e:
            print(f"Warning: Failed to initialize Milvus client: {e}")
            object.__setattr__(self, '_initialized', False)

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

        for item in self.col_name_list:
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
        return class_mapping[top_item]

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
            response = requests.post(
                self.embedding_endpoint,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def _run(self, query: str) -> str:
        """
        Search the knowledge base for relevant information.

        Args:
            query: Natural language search query

        Returns:
            JSON string with search results
        """
        if not self.is_available():
            return json.dumps({
                "error": "RAG Milvus tool not available. Check database path and initialization."
            })

        try:
            # Map query to collection
            collection_name = self._mapping_group(query)
            print(f"📚 Searching collection: {collection_name}")

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

            return json.dumps({
                "query": query,
                "collection_searched": collection_name,
                "results_count": len(results),
                "results": results
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "error": str(e),
                "query": query
            })
