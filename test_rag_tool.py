"""Test script for RAG Milvus Tool"""
import json
from rag_milvus_tool import RAGMilvusTool


def test_rag_tool():
    """Test RAG Milvus tool functionality"""
    print("=" * 60)
    print("RAG Milvus Tool Test")
    print("=" * 60)

    # Initialize the tool
    print("\nInitializing RAG Milvus tool...")
    tool = RAGMilvusTool(
        db_path="./milvus_demo_batch_bmth_v1.db",
        top_k=5
    )

    if not tool.is_available():
        print("\n❌ RAG Milvus tool is not available")
        print("Make sure the database file exists at: ./milvus_demo_batch_bmth_v1.db")
        return

    print("✅ RAG Milvus tool initialized successfully")

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

        try:
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

        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\nYou can now use this tool with your documentation agent.")


if __name__ == "__main__":
    test_rag_tool()
