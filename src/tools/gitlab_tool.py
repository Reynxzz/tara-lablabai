"""GitLab MCP Tool for CrewAI - Fetches project information from GitLab"""
import json
import requests
from typing import Any, Dict, List, Type
from urllib.parse import quote_plus
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from src.config.settings import get_settings
from src.utils.logger import setup_logger
from src.utils.validators import validate_gitlab_project

logger = setup_logger(__name__)


class GitLabMCPToolSchema(BaseModel):
    """Input schema for GitLabMCPTool."""
    project: str = Field(..., description="Project in format 'namespace/project' or project ID")


class GitLabMCPTool(BaseTool):
    """
    Tool for analyzing GitLab projects.

    Extracts information about code structure, files, commits, and project metadata
    to assist in generating comprehensive documentation.
    """

    name: str = "GitLab Project Analyzer"
    description: str = (
        "⚠️ THIS IS THE PRIMARY GITLAB TOOL FOR LEARNING PATH GENERATION ⚠️\n"
        "Analyzes GitLab projects to extract metadata, file structure, commits, README, and code snippets. "
        "Use this tool for: learning path generation, project overview, contributor info, and file structure. "
        "DO NOT use this for answering specific code questions - use the Code Q&A tool for that. "
        "Input should be the project namespace and name in format 'namespace/project'. "
        "Returns: project info, file structure, commits, README, and code snippets."
    )
    args_schema: Type[BaseModel] = GitLabMCPToolSchema

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Initialize GitLab tool with configuration."""
        super().__init__(**kwargs)
        settings = get_settings()
        object.__setattr__(self, 'gitlab_url', settings.gitlab.url)
        object.__setattr__(self, 'token', settings.gitlab.token)
        object.__setattr__(self, 'headers', {'PRIVATE-TOKEN': settings.gitlab.token})
        logger.info(f"Initialized GitLabMCPTool with URL: {settings.gitlab.url}")

    def _run(self, project: str) -> str:
        """
        Fetch project information using GitLab REST API.

        Args:
            project: Project in format 'namespace/project' or project ID

        Returns:
            JSON string with project information
        """
        logger.info("=" * 80)
        logger.info("📊 GITLAB PROJECT ANALYZER TOOL CALLED")
        logger.info(f"Project: {project}")
        logger.info("This is the GitLab tool, NOT the Internal Knowledge Base Search tool!")
        logger.info("=" * 80)

        if not validate_gitlab_project(project):
            error_msg = f"Invalid project format: {project}. Expected format: namespace/project"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        try:
            logger.info(f"Fetching GitLab project information: {project}")

            # Get project information
            project_info = self._get_project_info(project)

            if "error" in project_info:
                logger.error(f"Failed to fetch project info: {project_info['error']}")
                return json.dumps({"error": project_info["error"]})

            # Get file structure
            file_structure = self._get_file_structure(project)

            # Get recent commits
            commits = self._get_recent_commits(project, limit=5)

            # Get README
            readme = self._get_readme(project)

            # Get code snippets from key files
            code_snippets = self._get_code_snippets(project, project_info.get("default_branch", "main"))

            result = {
                "project": project,
                "info": project_info,
                "file_structure": file_structure,
                "recent_commits": commits,
                "readme": readme,
                "code_snippets": code_snippets
            }

            logger.info(f"Successfully fetched data for project: {project}")
            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error fetching GitLab project data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg})

    def _get_project_info(self, project: str) -> Dict[str, Any]:
        """
        Get basic project information.

        Args:
            project: Project identifier

        Returns:
            Dictionary with project information
        """
        try:
            gitlab_url = getattr(self, 'gitlab_url')
            headers = getattr(self, 'headers')
            project_encoded = quote_plus(project)
            url = f'{gitlab_url}/api/v4/projects/{project_encoded}'

            logger.debug(f"Fetching project info from: {url}")
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return {
                    "id": data.get("id"),
                    "name": data.get("name"),
                    "path": data.get("path"),
                    "path_with_namespace": data.get("path_with_namespace"),
                    "description": data.get("description"),
                    "default_branch": data.get("default_branch"),
                    "visibility": data.get("visibility"),
                    "star_count": data.get("star_count"),
                    "forks_count": data.get("forks_count"),
                    "open_issues_count": data.get("open_issues_count"),
                    "topics": data.get("topics", []),
                    "created_at": data.get("created_at"),
                    "last_activity_at": data.get("last_activity_at"),
                    "web_url": data.get("web_url"),
                    "readme_url": data.get("readme_url"),
                    "license": data.get("license", {}).get("name") if data.get("license") else None
                }
            else:
                error_msg = f"Failed to fetch project info: HTTP {response.status_code}"
                logger.warning(error_msg)
                return {"error": error_msg}

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching project info: {e}")
            return {"error": str(e)}

    def _get_file_structure(self, project: str, path: str = "") -> Dict[str, Any]:
        """
        Get project file structure.

        Args:
            project: Project identifier
            path: Path within repository (default: root)

        Returns:
            Dictionary with file structure
        """
        try:
            gitlab_url = getattr(self, 'gitlab_url')
            headers = getattr(self, 'headers')
            project_encoded = quote_plus(project)
            url = f'{gitlab_url}/api/v4/projects/{project_encoded}/repository/tree'

            logger.debug(f"Fetching file structure from: {url}")
            response = requests.get(
                url,
                headers=headers,
                params={'path': path, 'per_page': 20},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                structure = []
                for item in data[:20]:  # Limit to first 20 items
                    structure.append({
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "type": item.get("type"),
                        "mode": item.get("mode")
                    })
                return {"files": structure}
            else:
                logger.warning(f"Failed to fetch file structure: HTTP {response.status_code}")
                return {"error": f"Failed to fetch file structure: {response.status_code}"}

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching file structure: {e}")
            return {"error": str(e)}

    def _get_recent_commits(self, project: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent commits.

        Args:
            project: Project identifier
            limit: Number of commits to fetch

        Returns:
            List of commit dictionaries
        """
        try:
            gitlab_url = getattr(self, 'gitlab_url')
            headers = getattr(self, 'headers')
            project_encoded = quote_plus(project)
            url = f'{gitlab_url}/api/v4/projects/{project_encoded}/repository/commits'

            logger.debug(f"Fetching recent commits from: {url}")
            response = requests.get(
                url,
                headers=headers,
                params={'per_page': limit},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                commits = []
                for commit in data:
                    commits.append({
                        "id": commit.get("id", "")[:7],
                        "short_id": commit.get("short_id", ""),
                        "title": commit.get("title", ""),
                        "message": commit.get("message", ""),
                        "author_name": commit.get("author_name", ""),
                        "authored_date": commit.get("authored_date", ""),
                        "web_url": commit.get("web_url", "")
                    })
                return commits
            else:
                logger.warning(f"Failed to fetch commits: HTTP {response.status_code}")
                return [{"error": f"Failed to fetch commits: {response.status_code}"}]

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching commits: {e}")
            return [{"error": str(e)}]

    def _get_readme(self, project: str) -> str:
        """
        Get project README.

        Args:
            project: Project identifier

        Returns:
            README content (truncated to 1000 chars)
        """
        try:
            gitlab_url = getattr(self, 'gitlab_url')
            headers = getattr(self, 'headers')
            project_encoded = quote_plus(project)
            readme_filenames = ['README.md', 'README', 'readme.md', 'Readme.md']

            for filename in readme_filenames:
                file_encoded = quote_plus(filename)

                # Try main branch first
                for branch in ['main', 'master']:
                    url = f'{gitlab_url}/api/v4/projects/{project_encoded}/repository/files/{file_encoded}/raw'

                    logger.debug(f"Trying to fetch README: {filename} on branch {branch}")
                    response = requests.get(
                        url,
                        headers=headers,
                        params={'ref': branch},
                        timeout=30
                    )

                    if response.status_code == 200:
                        content = response.text[:1000]
                        if len(response.text) > 1000:
                            content += "..."
                        logger.info(f"Successfully fetched README: {filename}")
                        return content

            logger.warning("README not found in repository")
            return "README not found or inaccessible"

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching README: {e}")
            return f"Error fetching README: {str(e)}"

    def _get_code_snippets(self, project: str, branch: str = "main") -> Dict[str, Any]:
        """
        Get code snippets from key files in the repository.

        Args:
            project: Project identifier
            branch: Branch name (default: main)

        Returns:
            Dictionary with code snippets from key files
        """
        try:
            gitlab_url = getattr(self, 'gitlab_url')
            headers = getattr(self, 'headers')
            project_encoded = quote_plus(project)

            # Key files to fetch (in priority order)
            key_files = [
                'main.py',
                'app.py',
                '__init__.py',
                'setup.py',
                'requirements.txt',
                'Dockerfile',
                'docker-compose.yml',
                'config.py',
                'settings.py'
            ]

            snippets = {}

            for filename in key_files:
                file_encoded = quote_plus(filename)
                url = f'{gitlab_url}/api/v4/projects/{project_encoded}/repository/files/{file_encoded}/raw'

                logger.debug(f"Trying to fetch code snippet: {filename}")
                response = requests.get(
                    url,
                    headers=headers,
                    params={'ref': branch},
                    timeout=30
                )

                if response.status_code == 200:
                    content = response.text
                    # Truncate to reasonable size (300 chars for snippets)
                    snippet = content[:300]
                    if len(content) > 300:
                        snippet += "\n... (truncated)"

                    # Create file link
                    file_link = f"{gitlab_url}/{project}/-/blob/{branch}/{filename}"

                    snippets[filename] = {
                        "content": snippet,
                        "link": file_link,
                        "full_length": len(content)
                    }
                    logger.info(f"Successfully fetched snippet: {filename}")

                    # Limit to 3 snippets to avoid too much data
                    if len(snippets) >= 3:
                        break

            if not snippets:
                logger.info("No key files found for code snippets")
                return {"message": "No key files found"}

            return snippets

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching code snippets: {e}")
            return {"error": str(e)}

    def _get_code_files_from_directory(self, project: str, branch: str = "main", directory: str = "src", max_files: int = 10) -> Dict[str, Any]:
        """
        Recursively fetch code files from a specific directory.

        Args:
            project: Project identifier
            branch: Branch name (default: main)
            directory: Directory to search (default: src)
            max_files: Maximum number of files to fetch (default: 10)

        Returns:
            Dictionary with code files and their contents
        """
        try:
            gitlab_url = getattr(self, 'gitlab_url')
            headers = getattr(self, 'headers')
            project_encoded = quote_plus(project)

            # Get directory tree recursively
            url = f'{gitlab_url}/api/v4/projects/{project_encoded}/repository/tree'

            logger.info(f"Fetching code files from {directory}/ directory")
            response = requests.get(
                url,
                headers=headers,
                params={'path': directory, 'recursive': True, 'per_page': 100, 'ref': branch},
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Failed to fetch directory tree: HTTP {response.status_code}")
                return {"error": f"Failed to fetch directory tree: {response.status_code}"}

            tree = response.json()

            # Filter for Python files (exclude .ipynb)
            python_files = [
                item for item in tree
                if item.get('type') == 'blob'
                and item.get('name', '').endswith('.py')
                and not item.get('name', '').endswith('.ipynb')
            ]

            if not python_files:
                logger.info(f"No Python files found in {directory}/")
                return {"message": f"No Python files found in {directory}/"}

            # Fetch content of files (limit to max_files)
            code_files = {}
            for file_item in python_files[:max_files]:
                file_path = file_item.get('path')
                file_name = file_item.get('name')

                # Fetch file content
                file_encoded = quote_plus(file_path)
                file_url = f'{gitlab_url}/api/v4/projects/{project_encoded}/repository/files/{file_encoded}/raw'

                logger.debug(f"Fetching code file: {file_path}")
                file_response = requests.get(
                    file_url,
                    headers=headers,
                    params={'ref': branch},
                    timeout=30
                )

                if file_response.status_code == 200:
                    content = file_response.text

                    # Create file link
                    file_link = f"{gitlab_url}/{project}/-/blob/{branch}/{file_path}"

                    # Limit content to reasonable size (first 1000 lines or 50KB)
                    lines = content.split('\n')
                    if len(lines) > 1000:
                        content = '\n'.join(lines[:1000]) + "\n... (truncated)"
                    elif len(content) > 50000:
                        content = content[:50000] + "\n... (truncated)"

                    code_files[file_path] = {
                        "name": file_name,
                        "path": file_path,
                        "link": file_link,
                        "content": content,
                        "lines": len(lines)
                    }
                    logger.info(f"Successfully fetched: {file_path} ({len(lines)} lines)")

            logger.info(f"Fetched {len(code_files)} code files from {directory}/")
            return {
                "directory": directory,
                "files_count": len(code_files),
                "files": code_files
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching code files: {e}")
            return {"error": str(e)}
