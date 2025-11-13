"""GitLab Code Q&A Tool - Deep dives into repository code to answer questions"""
import json
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from src.config.settings import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GitLabCodeQAToolSchema(BaseModel):
    """Input schema for GitLabCodeQATool."""
    project: str = Field(..., description="Project in format 'namespace/project'")
    question: str = Field(..., description="Question about the codebase")
    directory: str = Field(default="src", description="Directory to search (default: src)")


class GitLabCodeQATool(BaseTool):
    """
    Tool for answering questions about GitLab repository code.

    Fetches code files from specified directory and provides context for answering
    specific questions about the codebase (e.g., feature processing, architecture).
    """

    name: str = "GitLab Code Q&A"
    description: str = (
        "⚠️ ONLY FOR ANSWERING SPECIFIC CODE QUESTIONS - NOT FOR LEARNING PATH GENERATION ⚠️\n"
        "This tool answers specific questions about repository code by deep-diving into Python files. "
        "DO NOT use this tool for learning path generation - use 'GitLab Project Analyzer' instead. "
        "ONLY use this when a user asks a specific question like 'What feature processing does this do?' "
        "Input requires: project path, question, and optional directory (default: src). "
        "Returns: code files with full content and links for answering the specific question asked."
    )
    args_schema: Type[BaseModel] = GitLabCodeQAToolSchema

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Initialize GitLab Code Q&A tool with configuration."""
        super().__init__(**kwargs)
        settings = get_settings()
        object.__setattr__(self, 'gitlab_url', settings.gitlab.url)
        object.__setattr__(self, 'token', settings.gitlab.token)

        # Import GitLabMCPTool to reuse the code fetching method
        from src.tools.gitlab_tool import GitLabMCPTool
        gitlab_tool = GitLabMCPTool()
        object.__setattr__(self, '_gitlab_tool', gitlab_tool)

        logger.info("Initialized GitLabCodeQATool")

    def _run(self, project: str, question: str, directory: str = "src") -> str:
        """
        Fetch code files and provide context for answering the question.

        Args:
            project: Project in format 'namespace/project'
            question: Question about the codebase
            directory: Directory to search (default: src)

        Returns:
            JSON string with code files and their contents
        """
        logger.info("=" * 80)
        logger.info("💬 GITLAB CODE Q&A TOOL CALLED")
        logger.info(f"Project: {project}")
        logger.info(f"Question: {question}")
        logger.info(f"Directory: {directory}")
        logger.info("=" * 80)

        try:
            # Get project info to find default branch
            gitlab_tool = getattr(self, '_gitlab_tool')
            project_info = gitlab_tool._get_project_info(project)

            if "error" in project_info:
                return json.dumps({"error": project_info["error"]})

            branch = project_info.get("default_branch", "main")

            # Fetch code files from directory
            logger.info(f"Fetching code files from {directory}/ on branch {branch}")
            code_data = gitlab_tool._get_code_files_from_directory(
                project=project,
                branch=branch,
                directory=directory,
                max_files=10
            )

            if "error" in code_data:
                logger.error(f"Failed to fetch code files: {code_data['error']}")
                return json.dumps({
                    "error": code_data["error"],
                    "question": question
                })

            if "message" in code_data:
                logger.warning(f"No files found: {code_data['message']}")
                return json.dumps({
                    "message": code_data["message"],
                    "question": question,
                    "suggestion": f"Try a different directory or check if {directory}/ exists in the repository"
                })

            # Prepare result
            result = {
                "project": project,
                "question": question,
                "directory": directory,
                "branch": branch,
                "files_count": code_data.get("files_count", 0),
                "files": []
            }

            # Format files for easier consumption
            files_dict = code_data.get("files", {})
            for file_path, file_data in files_dict.items():
                result["files"].append({
                    "name": file_data.get("name"),
                    "path": file_data.get("path"),
                    "link": file_data.get("link"),
                    "content": file_data.get("content"),
                    "lines": file_data.get("lines")
                })

            logger.info(f"Successfully fetched {len(result['files'])} code files")
            logger.info("Agent should now analyze these files to answer the question")

            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error in Code Q&A tool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({
                "error": error_msg,
                "question": question
            })
