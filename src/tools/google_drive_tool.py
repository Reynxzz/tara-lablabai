"""Google Drive MCP Tool for CrewAI - Searches and retrieves Google Drive documents"""
import json
import requests
from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from src.config.settings import get_settings
from src.config.constants import DEFAULT_DRIVE_TOP_K
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GoogleDriveMCPToolSchema(BaseModel):
    """Input schema for GoogleDriveMCPTool."""
    query: str = Field(..., description="Search query or file name to search in Google Drive")


class GoogleDriveMCPTool(BaseTool):
    """
    Tool for searching and retrieving documents from Google Drive.

    Uses MCP (Model Context Protocol) server to interface with Google Drive API.
    """

    name: str = "Google Drive Document Analyzer"
    description: str = (
        "Searches for and retrieves documents from Google Drive. "
        "Useful for finding documentation, specifications, or reference materials. "
        "Input should be a search query or document name."
    )
    args_schema: Type[BaseModel] = GoogleDriveMCPToolSchema

    # Pydantic fields
    mcp_url: str = ""
    access_token: str = ""
    top_k: int = DEFAULT_DRIVE_TOP_K

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, access_token: str = None, mcp_url: str = None, **kwargs):
        """
        Initialize Google Drive MCP tool.

        Args:
            access_token: Google Drive access token (optional, reads from settings if not provided)
            mcp_url: MCP server URL (optional, reads from settings if not provided)
        """
        settings = get_settings()

        # Use provided values or fallback to settings
        token = access_token or settings.google_drive.token
        url = mcp_url or settings.google_drive.mcp_url

        # Ensure URL has proper protocol
        if url and not url.startswith('http'):
            url = f'https://{url}'

        super().__init__(
            mcp_url=url,
            access_token=token or "",
            **kwargs
        )

        # Verify MCP server is reachable
        object.__setattr__(self, '_initialized', False)
        if self.access_token:
            self._initialize_mcp()

    def _initialize_mcp(self):
        """Verify MCP server is reachable."""
        if not self.access_token:
            logger.warning("No Google Drive access token provided - MCP tools disabled")
            return

        try:
            response = requests.post(
                self.mcp_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "tools/list"
                },
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"MCP Drive server reachable at {self.mcp_url}")
                object.__setattr__(self, '_initialized', True)
            else:
                logger.warning(f"MCP server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"MCP server not reachable at {self.mcp_url}: {e}")

    def is_available(self) -> bool:
        """Check if MCP tools are available."""
        return getattr(self, '_initialized', False)

    def _run(self, query: str) -> str:
        """
        Search Google Drive and retrieve document content.

        Args:
            query: Search query or document name

        Returns:
            JSON string with search results and file contents
        """
        if not self.is_available():
            error_msg = "Google Drive MCP tools not available. Check access token and MCP server."
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        try:
            logger.info(f"Searching Google Drive for: {query}")

            # Search for files
            files = self._search_files(query)

            if not files:
                logger.info(f"No files found matching '{query}'")
                return json.dumps({
                    "query": query,
                    "files_found": 0,
                    "message": f"No files found matching '{query}'"
                })

            # Get content of first few files (limit to prevent overload)
            results = []
            for file_info in files[:self.top_k]:
                file_uri = file_info.get("uri")
                file_name = file_info.get("name")

                if file_uri:
                    logger.debug(f"Retrieving file: {file_name}")
                    file_content = self._get_file(file_uri)
                    content = file_content.get("content", "")
                    results.append({
                        "name": file_name,
                        "uri": file_uri,
                        "mimeType": file_info.get("mimeType", ""),
                        "content": content[:2000],  # Limit to first 2000 chars
                        "full_content_length": len(content)
                    })

            logger.info(f"Retrieved {len(results)} files from Google Drive")
            return json.dumps({
                "query": query,
                "files_found": len(files),
                "files_retrieved": len(results),
                "files": results
            }, indent=2)

        except Exception as e:
            error_msg = f"Error searching Google Drive: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg})

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
                        "access_token": self.access_token
                    }
                }
            }

            response = requests.post(
                self.mcp_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=90
            )

            if response.status_code != 200:
                logger.error(f"Failed to search Drive: HTTP {response.status_code}")
                return []

            json_response = response.json()

            if "error" in json_response:
                logger.error(f"MCP server returned error: {json_response['error']}")
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
                logger.info(f"Found {len(files)} files matching '{query}'")
                return files
            except json.JSONDecodeError:
                logger.error("Failed to parse search results as JSON")
                return []

        except requests.exceptions.RequestException as e:
            logger.error(f"Drive search request failed: {e}")
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
                        "access_token": self.access_token
                    }
                }
            }

            response = requests.post(
                self.mcp_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=180
            )

            if response.status_code != 200:
                logger.error(f"Failed to get file: HTTP {response.status_code}")
                return {}

            json_response = response.json()

            if "error" in json_response:
                logger.error(f"MCP server returned error: {json_response['error']}")
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
                logger.error("Failed to parse file data as JSON")
                return {}

        except requests.exceptions.RequestException as e:
            logger.error(f"Drive get_file request failed: {e}")
            return {}
