"""GitHub MCP Tool for CrewAI"""
import os
import subprocess
import json
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class GitHubMCPToolSchema(BaseModel):
    """Input schema for GitHubMCPTool."""
    repo: str = Field(..., description="Repository in format 'owner/repo'")


class GitHubMCPTool(BaseTool):
    name: str = "GitHub Repository Analyzer"
    description: str = (
        "Analyzes GitHub repositories to extract information about code structure, "
        "files, commits, and issues. Useful for generating comprehensive documentation. "
        "Input should be the repository owner and name in format 'owner/repo'."
    )
    args_schema: Type[BaseModel] = GitHubMCPToolSchema

    def _run(self, repo: str) -> str:
        """
        Fetch repository information using MCP GitHub server

        Args:
            repo: Repository in format 'owner/repo'

        Returns:
            JSON string with repository information
        """
        try:
            # Parse owner and repo
            if '/' not in repo:
                return json.dumps({"error": "Repository must be in format 'owner/repo'"})

            owner, repo_name = repo.split('/', 1)

            # Get repository information
            repo_info = self._get_repo_info(owner, repo_name)

            # Get file structure
            file_structure = self._get_file_structure(owner, repo_name)

            # Get recent commits (limited)
            commits = self._get_recent_commits(owner, repo_name, limit=5)

            # Get README
            readme = self._get_readme(owner, repo_name)

            result = {
                "repository": repo,
                "info": repo_info,
                "file_structure": file_structure,
                "recent_commits": commits,
                "readme": readme
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_repo_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get basic repository information"""
        try:
            # Use GitHub API through requests
            import requests
            token = os.getenv('GITHUB_TOKEN', '')
            headers = {'Authorization': f'token {token}'} if token else {}

            response = requests.get(
                f'https://api.github.com/repos/{owner}/{repo}',
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "name": data.get("name"),
                    "full_name": data.get("full_name"),
                    "description": data.get("description"),
                    "language": data.get("language"),
                    "stars": data.get("stargazers_count"),
                    "forks": data.get("forks_count"),
                    "open_issues": data.get("open_issues_count"),
                    "topics": data.get("topics", []),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "homepage": data.get("homepage"),
                    "license": data.get("license", {}).get("name") if data.get("license") else None
                }
            else:
                return {"error": f"Failed to fetch repo info: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def _get_file_structure(self, owner: str, repo: str, path: str = "") -> Dict[str, Any]:
        """Get repository file structure"""
        try:
            import requests
            token = os.getenv('GITHUB_TOKEN', '')
            headers = {'Authorization': f'token {token}'} if token else {}

            response = requests.get(
                f'https://api.github.com/repos/{owner}/{repo}/contents/{path}',
                headers=headers
            )

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
                return {"error": f"Failed to fetch file structure: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def _get_recent_commits(self, owner: str, repo: str, limit: int = 5) -> list:
        """Get recent commits"""
        try:
            import requests
            token = os.getenv('GITHUB_TOKEN', '')
            headers = {'Authorization': f'token {token}'} if token else {}

            response = requests.get(
                f'https://api.github.com/repos/{owner}/{repo}/commits?per_page={limit}',
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                commits = []
                for commit in data:
                    commits.append({
                        "sha": commit.get("sha", "")[:7],
                        "message": commit.get("commit", {}).get("message", ""),
                        "author": commit.get("commit", {}).get("author", {}).get("name", ""),
                        "date": commit.get("commit", {}).get("author", {}).get("date", "")
                    })
                return commits
            else:
                return [{"error": f"Failed to fetch commits: {response.status_code}"}]
        except Exception as e:
            return [{"error": str(e)}]

    def _get_readme(self, owner: str, repo: str) -> str:
        """Get repository README"""
        try:
            import requests
            token = os.getenv('GITHUB_TOKEN', '')
            headers = {
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3.raw'
            } if token else {'Accept': 'application/vnd.github.v3.raw'}

            response = requests.get(
                f'https://api.github.com/repos/{owner}/{repo}/readme',
                headers=headers
            )

            if response.status_code == 200:
                # Limit README to first 1000 characters
                return response.text[:1000] + ("..." if len(response.text) > 1000 else "")
            else:
                return f"README not found or inaccessible"
        except Exception as e:
            return f"Error fetching README: {str(e)}"
