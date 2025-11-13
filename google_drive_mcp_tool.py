"""Google Drive MCP Tool for CrewAI"""
import os
import json
import asyncio
import requests
from typing import Any, Dict, Type, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class GoogleDriveMCPToolSchema(BaseModel):
    """Input schema for GoogleDriveMCPTool."""
    query: str = Field(..., description="Search query or file name to search in Google Drive")


class GoogleDriveMCPTool(BaseTool):
    name: str = "Google Drive Document Analyzer"
    description: str = (
        "Searches for and retrieves documents from Google Drive. "
        "Useful for finding documentation, specifications, or reference materials. "
        "Input should be a search query or document name."
    )
    args_schema: Type[BaseModel] = GoogleDriveMCPToolSchema

    def __init__(self, access_token: str = None, user_id: str = None, mcp_url: str = None, **kwargs):
        """
        Initialize Google Drive MCP tool.

        Args:
            access_token: Google Drive access token
            user_id: User identifier
            mcp_url: MCP server URL (defaults to MCP_DRIVE_URL env var or http://localhost:9000)
        """
        super().__init__(**kwargs)
        self._access_token = access_token or os.getenv("GOOGLE_DRIVE_TOKEN")
        self._user_id = user_id or os.getenv("GOOGLE_DRIVE_USER_ID", "default")
        self._mcp_url = mcp_url or os.getenv("MCP_DRIVE_URL", "http://localhost:9000")
        self._initialized = False

        if self._access_token:
            self._initialize_mcp()

    def _initialize_mcp(self):
        """Verify MCP server is reachable."""
        if not self._access_token:
            print("Warning: No Google Drive access token provided - MCP tools disabled")
            self._initialized = False
            return

        try:
            response = requests.post(
                self._mcp_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "tools/list"
                },
                timeout=5
            )
            if response.status_code == 200:
                print(f"✅ MCP Drive server reachable at {self._mcp_url}")
                self._initialized = True
            else:
                print(f"Warning: MCP server returned status {response.status_code}")
                self._initialized = False
        except requests.exceptions.RequestException as e:
            print(f"Warning: MCP server not reachable at {self._mcp_url}: {e}")
            self._initialized = False

    def is_available(self) -> bool:
        """Check if MCP tools are available."""
        return self._initialized

    def _run(self, query: str) -> str:
        """
        Search Google Drive and retrieve document content.

        Args:
            query: Search query or document name

        Returns:
            JSON string with search results and file contents
        """
        if not self.is_available():
            return json.dumps({"error": "Google Drive MCP tools not available. Check access token and MCP server."})

        try:
            # Search for files
            files = self._search_files(query)

            if not files:
                return json.dumps({
                    "query": query,
                    "files_found": 0,
                    "message": f"No files found matching '{query}'"
                })

            # Get content of first few files (limit to 3 to avoid overload)
            results = []
            for file_info in files[:3]:
                file_uri = file_info.get("uri")
                file_name = file_info.get("name")

                if file_uri:
                    file_content = self._get_file(file_uri)
                    results.append({
                        "name": file_name,
                        "uri": file_uri,
                        "mimeType": file_info.get("mimeType", ""),
                        "content": file_content.get("content", "")[:2000],  # Limit content to first 2000 chars
                        "full_content_length": len(file_content.get("content", ""))
                    })

            return json.dumps({
                "query": query,
                "files_found": len(files),
                "files_retrieved": len(results),
                "files": results
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _search_files(self, query: str) -> List[Dict[str, str]]:
        """
        Search for files in Google Drive using MCP search tool via HTTP.

        Args:
            query: Search query string

        Returns:
            List of dicts with file metadata
        """
        try:
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "query": query,
                        "access_token": self._access_token
                    }
                }
            }

            response = requests.post(
                self._mcp_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=90
            )

            if response.status_code != 200:
                print(f"Failed to search Drive: HTTP {response.status_code}")
                return []

            json_response = response.json()

            if "error" in json_response:
                print(f"MCP server returned error: {json_response['error']}")
                return []

            if "result" not in json_response:
                return []

            content = json_response["result"].get("content", [])
            if not content:
                return []

            text = content[0].get("text", "")

            try:
                search_data = json.loads(text)
                files = search_data.get("files", [])
                print(f"✅ Found {len(files)} files matching '{query}'")
                return files
            except json.JSONDecodeError:
                return []

        except Exception as e:
            print(f"Drive search failed: {e}")
            return []

    def _get_file(self, uri: str) -> Dict[str, Any]:
        """
        Get file content from Google Drive using MCP get_file tool via HTTP.

        Args:
            uri: File URI from search results (e.g., "gdrive:///fileId")

        Returns:
            Dict with file metadata and content
        """
        try:
            request_data = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_file",
                    "arguments": {
                        "uri": uri,
                        "access_token": self._access_token
                    }
                }
            }

            response = requests.post(
                self._mcp_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=180
            )

            if response.status_code != 200:
                print(f"Failed to get file: HTTP {response.status_code}")
                return {}

            json_response = response.json()

            if "error" in json_response:
                print(f"MCP server returned error: {json_response['error']}")
                return {}

            if "result" not in json_response:
                return {}

            content = json_response["result"].get("content", [])
            if not content:
                return {}

            text = content[0].get("text", "")

            try:
                file_data = json.loads(text)
                return file_data
            except json.JSONDecodeError:
                return {}

        except Exception as e:
            print(f"Drive get_file failed: {e}")
            return {}
