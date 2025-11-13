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
        "Analyzes GitLab projects to extract information about code structure, "
        "files, commits, and merge requests. Useful for generating comprehensive documentation. "
        "Input should be the project namespace and name in format 'namespace/project'."
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

            result = {
                "project": project,
                "info": project_info,
                "file_structure": file_structure,
                "recent_commits": commits,
                "readme": readme
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
