"""Test script for Google Drive MCP Tool"""
import sys
from pathlib import Path
import json

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from src.tools import GoogleDriveMCPTool
from src.config.settings import get_settings


def test_drive_tool():
    """Test Google Drive MCP tool functionality."""
    print("=" * 60)
    print("Google Drive MCP Integration Test")
    print("=" * 60)

    try:
        settings = get_settings()

        if not settings.google_drive.token:
            print("\n❌ Error: GOOGLE_DRIVE_TOKEN not found in environment")
            print("Please set GOOGLE_DRIVE_TOKEN in your .env file")
            return 1

        print(f"\n✅ Google Drive access token found (length: {len(settings.google_drive.token)} chars)")
        print(f"✅ MCP server URL: {settings.google_drive.mcp_url}")

        # Initialize the tool
        print("\n" + "-" * 60)
        print("Initializing Google Drive MCP tool...")
        print("-" * 60)

        tool = GoogleDriveMCPTool()

        if not tool.is_available():
            print("\n❌ Google Drive MCP tool is not available")
            print("Make sure the MCP server is running at:", settings.google_drive.mcp_url)
            return 1

        print("\n✅ Google Drive MCP tool initialized successfully")

        # Test search functionality
        print("\n" + "=" * 60)
        print("TEST: Search for documents")
        print("=" * 60)

        search_query = input("\nEnter search query (or press Enter for 'documentation'): ").strip()
        if not search_query:
            search_query = "documentation"

        print(f"\nSearching for: '{search_query}'...")

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

        # Display first file content preview
        if result_data.get("files_found", 0) > 0:
            print("\n" + "=" * 60)
            print("File Content Preview")
            print("=" * 60)

            first_file = result_data.get("files", [])[0]
            print(f"\nFile: {first_file.get('name')}")
            print(f"URI: {first_file.get('uri')}")
            print(f"\nContent preview (first 500 chars):")
            print("-" * 60)
            content = first_file.get('content', '')
            print(content[:500])
            if len(content) > 500:
                print(f"\n... ({len(content) - 500} more characters)")

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        print("\nYou can now use Google Drive integration with your documentation agent:")
        print("python scripts/run_documentation_agent.py namespace/project --with-drive")

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
    sys.exit(test_drive_tool())
