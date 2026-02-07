"""GitHub Tool for CrewAI - Fetches project information from GitHub"""
import json
import requests
import base64
from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from src.config.settings import get_settings
from src.utils.logger import setup_logger
from src.utils.validators import validate_github_repo

logger = setup_logger(__name__)


class GitHubToolSchema(BaseModel):
    """Input schema for GitHubTool."""
    repo: str = Field(..., description="Repository in format 'owner/repo'")


class GitHubTool(BaseTool):
    """
    Tool for analyzing GitHub repositories.

    Extracts information about code structure, files, commits, and project metadata
    to assist in generating comprehensive documentation.
    """

    name: str = "GitHub Project Analyzer"
    description: str = (
        "Analyzes GitHub repositories to extract metadata, file structure, commits, README, and code snippets. "
        "Use this tool for: learning path generation, project overview, contributor info, and file structure. "
        "DO NOT use this for answering specific code questions - use the Code Q&A tool for that. "
        "Input should be the repository in format 'owner/repo'. "
        "Returns: project info, file structure, commits, README, and code snippets."
    )
    args_schema: Type[BaseModel] = GitHubToolSchema

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Initialize GitHub tool with configuration."""
        super().__init__(**kwargs)
        settings = get_settings()
        object.__setattr__(self, 'api_url', settings.github.api_url.rstrip('/'))
        object.__setattr__(self, 'token', settings.github.token)
        object.__setattr__(self, 'headers', {
            'Authorization': f'Bearer {settings.github.token}',
            'Accept': 'application/vnd.github.v3+json',
            'X-GitHub-Api-Version': '2022-11-28'
        })
        logger.info(f"Initialized GitHubTool with API URL: {settings.github.api_url}")

    def _run(self, repo: str) -> str:
        """
        Fetch repository information using GitHub REST API.

        Args:
            repo: Repository in format 'owner/repo'

        Returns:
            JSON string with repository information
        """
        logger.info("=" * 80)
        logger.info("GITHUB PROJECT ANALYZER TOOL CALLED")
        logger.info(f"Repository: {repo}")
        logger.info("=" * 80)

        if not validate_github_repo(repo):
            error_msg = f"Invalid repository format: {repo}. Expected format: owner/repo"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        try:
            logger.info(f"Fetching GitHub repository information: {repo}")

            # Get repository information
            repo_info = self._get_repo_info(repo)

            if "error" in repo_info:
                logger.error(f"Failed to fetch repo info: {repo_info['error']}")
                return json.dumps({"error": repo_info["error"]})

            # Get file structure
            file_structure = self._get_file_structure(repo)

            # Get recent commits
            commits = self._get_recent_commits(repo, limit=5)

            # Get README
            readme = self._get_readme(repo)

            # Get code snippets from key files
            code_snippets = self._get_code_snippets(repo, repo_info.get("default_branch", "main"))

            result = {
                "repository": repo,
                "info": repo_info,
                "file_structure": file_structure,
                "recent_commits": commits,
                "readme": readme,
                "code_snippets": code_snippets
            }

            logger.info(f"Successfully fetched data for repository: {repo}")
            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error fetching GitHub repository data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg})

    def _get_repo_info(self, repo: str) -> Dict[str, Any]:
        """
        Get basic repository information.

        Args:
            repo: Repository identifier (owner/repo)

        Returns:
            Dictionary with repository information
        """
        try:
            api_url = getattr(self, 'api_url')
            headers = getattr(self, 'headers')
            url = f'{api_url}/repos/{repo}'

            logger.debug(f"Fetching repo info from: {url}")
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return {
                    "id": data.get("id"),
                    "name": data.get("name"),
                    "full_name": data.get("full_name"),
                    "description": data.get("description"),
                    "default_branch": data.get("default_branch"),
                    "visibility": "private" if data.get("private") else "public",
                    "stargazers_count": data.get("stargazers_count"),
                    "forks_count": data.get("forks_count"),
                    "open_issues_count": data.get("open_issues_count"),
                    "topics": data.get("topics", []),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "pushed_at": data.get("pushed_at"),
                    "html_url": data.get("html_url"),
                    "language": data.get("language"),
                    "license": data.get("license", {}).get("name") if data.get("license") else None
                }
            else:
                error_msg = f"Failed to fetch repo info: HTTP {response.status_code}"
                logger.warning(error_msg)
                return {"error": error_msg}

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching repo info: {e}")
            return {"error": str(e)}

    def _get_file_structure(self, repo: str, path: str = "") -> Dict[str, Any]:
        """
        Get repository file structure.

        Args:
            repo: Repository identifier
            path: Path within repository (default: root)

        Returns:
            Dictionary with file structure
        """
        try:
            api_url = getattr(self, 'api_url')
            headers = getattr(self, 'headers')
            url = f'{api_url}/repos/{repo}/contents/{path}'

            logger.debug(f"Fetching file structure from: {url}")
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                structure = []
                for item in data[:20]:  # Limit to first 20 items
                    structure.append({
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "type": item.get("type"),
                        "size": item.get("size")
                    })
                return {"files": structure}
            else:
                logger.warning(f"Failed to fetch file structure: HTTP {response.status_code}")
                return {"error": f"Failed to fetch file structure: {response.status_code}"}

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching file structure: {e}")
            return {"error": str(e)}

    def _get_recent_commits(self, repo: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent commits.

        Args:
            repo: Repository identifier
            limit: Number of commits to fetch

        Returns:
            List of commit dictionaries
        """
        try:
            api_url = getattr(self, 'api_url')
            headers = getattr(self, 'headers')
            url = f'{api_url}/repos/{repo}/commits'

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
                    commit_data = commit.get("commit", {})
                    author = commit_data.get("author", {})
                    commits.append({
                        "sha": commit.get("sha", "")[:7],
                        "message": commit_data.get("message", "").split('\n')[0],  # First line only
                        "author_name": author.get("name", ""),
                        "author_date": author.get("date", ""),
                        "html_url": commit.get("html_url", "")
                    })
                return commits
            else:
                logger.warning(f"Failed to fetch commits: HTTP {response.status_code}")
                return [{"error": f"Failed to fetch commits: {response.status_code}"}]

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching commits: {e}")
            return [{"error": str(e)}]

    def _get_readme(self, repo: str) -> str:
        """
        Get repository README.

        Args:
            repo: Repository identifier

        Returns:
            README content (truncated to 1000 chars)
        """
        try:
            api_url = getattr(self, 'api_url')
            headers = getattr(self, 'headers')
            url = f'{api_url}/repos/{repo}/readme'

            logger.debug(f"Fetching README from: {url}")
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                # Content is base64 encoded
                content_b64 = data.get("content", "")
                try:
                    content = base64.b64decode(content_b64).decode('utf-8')
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    logger.info("Successfully fetched README")
                    return content
                except Exception as e:
                    logger.warning(f"Error decoding README: {e}")
                    return "README found but could not be decoded"
            else:
                logger.warning("README not found in repository")
                return "README not found or inaccessible"

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching README: {e}")
            return f"Error fetching README: {str(e)}"

    def _get_code_snippets(self, repo: str, branch: str = "main") -> Dict[str, Any]:
        """
        Get code snippets from key files in the repository.

        Args:
            repo: Repository identifier
            branch: Branch name (default: main)

        Returns:
            Dictionary with code snippets from key files
        """
        try:
            api_url = getattr(self, 'api_url')
            headers = getattr(self, 'headers')

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
                'settings.py',
                'package.json',
                'index.js',
                'index.ts'
            ]

            snippets = {}

            for filename in key_files:
                url = f'{api_url}/repos/{repo}/contents/{filename}'

                logger.debug(f"Trying to fetch code snippet: {filename}")
                response = requests.get(
                    url,
                    headers=headers,
                    params={'ref': branch},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    content_b64 = data.get("content", "")
                    try:
                        content = base64.b64decode(content_b64).decode('utf-8')
                        # Truncate to reasonable size (300 chars for snippets)
                        snippet = content[:300]
                        if len(content) > 300:
                            snippet += "\n... (truncated)"

                        # Create file link
                        file_link = f"https://github.com/{repo}/blob/{branch}/{filename}"

                        snippets[filename] = {
                            "content": snippet,
                            "link": file_link,
                            "full_length": len(content)
                        }
                        logger.info(f"Successfully fetched snippet: {filename}")

                        # Limit to 3 snippets to avoid too much data
                        if len(snippets) >= 3:
                            break
                    except Exception as e:
                        logger.warning(f"Error decoding {filename}: {e}")

            if not snippets:
                logger.info("No key files found for code snippets")
                return {"message": "No key files found"}

            return snippets

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching code snippets: {e}")
            return {"error": str(e)}

    def _get_code_files_from_directory(self, repo: str, branch: str = "main", directory: str = "src", max_files: int = 10) -> Dict[str, Any]:
        """
        Recursively fetch code files from a specific directory.

        Args:
            repo: Repository identifier
            branch: Branch name (default: main)
            directory: Directory to search (default: src)
            max_files: Maximum number of files to fetch (default: 10)

        Returns:
            Dictionary with code files and their contents
        """
        try:
            api_url = getattr(self, 'api_url')
            headers = getattr(self, 'headers')

            # Get directory contents recursively using git trees API
            url = f'{api_url}/repos/{repo}/git/trees/{branch}?recursive=1'

            logger.info(f"Fetching code files from {directory}/ directory")
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch directory tree: HTTP {response.status_code}")
                return {"error": f"Failed to fetch directory tree: {response.status_code}"}

            tree = response.json().get("tree", [])

            # Supported code file extensions
            code_extensions = (
                '.py', '.js', '.ts', '.jsx', '.tsx',  # Python, JavaScript, TypeScript
                '.java', '.kt', '.scala',              # JVM languages
                '.go', '.rs', '.rb',                   # Go, Rust, Ruby
                '.php', '.c', '.cpp', '.h', '.hpp',    # PHP, C/C++
                '.cs', '.swift', '.m',                 # C#, Swift, Objective-C
                '.sql', '.sh', '.bash',                # SQL, Shell
                '.yaml', '.yml', '.json', '.toml',     # Config files
                '.md', '.rst', '.txt',                 # Documentation
                '.html', '.css', '.scss',              # Web files
            )

            # Handle root directory
            is_root = directory in ('.', '', '/')

            # Filter for code files in the specified directory
            code_files_list = []
            for item in tree:
                if item.get('type') != 'blob':
                    continue

                path = item.get('path', '')

                # Check if file has a supported extension
                if not any(path.endswith(ext) for ext in code_extensions):
                    continue

                # Check directory match
                if is_root:
                    # For root, include files at root level or in any subdirectory
                    code_files_list.append(item)
                elif path.startswith(f"{directory}/"):
                    code_files_list.append(item)

            if not code_files_list:
                logger.info(f"No code files found in {directory}/")
                return {"message": f"No code files found in {directory}/"}

            # Fetch content of files (limit to max_files)
            code_files = {}
            for file_item in code_files_list[:max_files]:
                file_path = file_item.get('path')
                file_name = file_path.split('/')[-1]

                # Fetch file content
                file_url = f'{api_url}/repos/{repo}/contents/{file_path}'

                logger.debug(f"Fetching code file: {file_path}")
                file_response = requests.get(
                    file_url,
                    headers=headers,
                    params={'ref': branch},
                    timeout=30
                )

                if file_response.status_code == 200:
                    data = file_response.json()
                    content_b64 = data.get("content", "")
                    try:
                        content = base64.b64decode(content_b64).decode('utf-8')

                        # Create file link
                        file_link = f"https://github.com/{repo}/blob/{branch}/{file_path}"

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
                    except Exception as e:
                        logger.warning(f"Error decoding {file_path}: {e}")

            logger.info(f"Fetched {len(code_files)} code files from {directory}/")
            return {
                "directory": directory,
                "files_count": len(code_files),
                "files": code_files
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching code files: {e}")
            return {"error": str(e)}
