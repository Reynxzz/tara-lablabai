"""Test script for RAG Milvus Tool"""
import sys
from pathlib import Path
import json

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from src.tools import RAGMilvusTool
from src.config.settings import get_settings


def test_rag_tool():
    """Test RAG Milvus tool functionality."""
    print("=" * 60)
    print("RAG Milvus Tool Test")
    print("=" * 60)

    try:
        settings = get_settings()

        # Initialize the tool
        print("\nInitializing RAG Milvus tool...")
        tool = RAGMilvusTool()

        if not tool.is_available():
            print("\n❌ RAG Milvus tool is not available")
            print("Make sure the database file exists at:", settings.rag.db_path)
            return 1

        print("✅ RAG Milvus tool initialized successfully")
        print(f"Database: {settings.rag.db_path}")
        print(f"Model: {settings.rag.embedding_model}")
        print(f"Top K: {settings.rag.top_k}")

        # Test queries
        test_queries = [
            "What is user income data?",
            "Tell me about DGE",
            "How do push notifications work?",
            "Information about Genie"
        ]

        for i, query in enumerate(test_queries, 1):
            print("\n" + "=" * 60)
            print(f"TEST {i}: {query}")
            print("=" * 60)

            result = tool._run(query)
            result_data = json.loads(result)

            print("\n" + "-" * 60)
            print("Search Results:")
            print("-" * 60)
            print(json.dumps(result_data, indent=2))

            # Display top result details
            if result_data.get("results_count", 0) > 0:
                print("\n📌 Top Result:")
                top_result = result_data["results"][0]
                print(f"Score: {top_result['score']:.4f}")
                print(f"Collection: {top_result['collection']}")
                print(f"Text preview: {top_result['text'][:200]}...")

        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print("\nYou can now use this tool with your documentation agent:")
        print("python scripts/run_documentation_agent.py namespace/project --with-rag")

        return 0

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_rag_tool())
