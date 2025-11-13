"""Test script for Google Drive MCP integration"""
import os
import json
from dotenv import load_dotenv
from google_drive_mcp_tool import GoogleDriveMCPTool


def test_drive_mcp():
    """Test Google Drive MCP tool functionality"""

    # Load environment variables
    load_dotenv()

    print("=" * 60)
    print("Google Drive MCP Integration Test")
    print("=" * 60)

    # Get credentials from environment
    access_token = os.getenv("GOOGLE_DRIVE_TOKEN")
    mcp_url = os.getenv("MCP_DRIVE_URL", "http://localhost:9000")

    if not access_token:
        print("\n❌ Error: GOOGLE_DRIVE_TOKEN not found in environment")
        print("Please set GOOGLE_DRIVE_TOKEN in your .env file")
        return

    print(f"\n✅ Google Drive access token found (length: {len(access_token)} chars)")
    print(f"✅ MCP server URL: {mcp_url}")

    # Initialize the tool
    print("\n" + "-" * 60)
    print("Initializing Google Drive MCP tool...")
    print("-" * 60)

    tool = GoogleDriveMCPTool(access_token=access_token, mcp_url=mcp_url)

    if not tool.is_available():
        print("\n❌ Google Drive MCP tool is not available")
        print("Make sure the MCP server is running at:", mcp_url)
        print("\nTo start the MCP server, you typically need to:")
        print("1. Install the MCP server (e.g., npx -y @qais/tara-mcp-drive)")
        print("2. Run it with your access token")
        return

    print("\n✅ Google Drive MCP tool initialized successfully")

    # Test search functionality
    print("\n" + "=" * 60)
    print("TEST 1: Search for documents")
    print("=" * 60)

    search_query = input("\nEnter search query (or press Enter for 'documentation'): ").strip()
    if not search_query:
        search_query = "documentation"

    print(f"\nSearching for: '{search_query}'...")

    try:
        result = tool._run(search_query)
        result_data = json.loads(result)

        print("\n" + "-" * 60)
        print("Search Results:")
        print("-" * 60)
        print(json.dumps(result_data, indent=2))

        # Save to file
        output_file = "drive_search_results.json"
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"\n✅ Search results saved to: {output_file}")

        # Test file retrieval if any files found
        if result_data.get("files_found", 0) > 0:
            print("\n" + "=" * 60)
            print("TEST 2: Retrieve file content")
            print("=" * 60)

            first_file = result_data.get("files", [])[0]
            print(f"\nRetrieving content for: {first_file.get('name')}")
            print(f"URI: {first_file.get('uri')}")
            print(f"Content preview (first 500 chars):")
            print("-" * 60)
            content = first_file.get('content', '')
            print(content[:500])
            if len(content) > 500:
                print(f"\n... ({len(content) - 500} more characters)")

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    print("\nYou can now use Google Drive integration with your documentation agent:")
    print("python documentation_agent.py namespace/project --with-drive")


if __name__ == "__main__":
    test_drive_mcp()
