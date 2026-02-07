"""Core crew orchestration for documentation generation"""
import json
import re
from typing import Dict, Optional, List
from crewai import Agent, Task, Crew, Process

from src.config.settings import get_settings
from src.config.constants import LLMModel
from src.tools import GitHubTool, GoogleDriveMCPTool, GitHubCodeQATool
from src.utils.logger import setup_logger
from src.utils.validators import validate_github_repo, sanitize_filename

logger = setup_logger(__name__)


def extract_markdown_from_response(response: str) -> str:
    """Extract and clean markdown content from various response formats."""
    content = response.strip()

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            for key in ["markdown_documentation", "documentation", "content", "markdown"]:
                if key in data and data[key]:
                    content = data[key]
                    break
    except (json.JSONDecodeError, ValueError):
        pass

    if content.startswith('```markdown') or content.startswith('```md'):
        content = re.sub(r'^```(?:markdown|md)\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
    elif content.startswith('```'):
        content = re.sub(r'^```\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)

    return content.strip()


class DocumentationCrew:
    """
    CrewAI setup for generating documentation from GitHub repositories.
    Uses GPT-4o for reliable tool calling.
    """

    def __init__(self, enable_google_drive: bool = False):
        """Initialize DocumentationCrew."""
        logger.info("Initializing DocumentationCrew")

        self.settings = get_settings()
        self.enable_google_drive = enable_google_drive

        # Initialize tools
        logger.info("Initializing tools...")
        self.github_tool = GitHubTool()

        self.drive_tool = None
        if enable_google_drive:
            if self.settings.google_drive.is_configured():
                self.drive_tool = GoogleDriveMCPTool()
                if self.drive_tool.is_available():
                    logger.info("Google Drive integration enabled")
                else:
                    logger.warning("Google Drive integration disabled (MCP server not reachable)")
                    self.enable_google_drive = False
            else:
                logger.warning("Google Drive integration disabled (no GOOGLE_DRIVE_TOKEN)")
                self.enable_google_drive = False

        # Use GPT-4o model string - CrewAI will use OpenAI directly
        self.model = LLMModel.GPT_4O.value
        logger.info(f"Using model: {self.model}")

    def generate_documentation(self, repo: str) -> Dict:
        """Generate documentation for a GitHub repository."""
        if not validate_github_repo(repo):
            raise ValueError(f"Invalid repository format: {repo}. Expected format: owner/repo")

        logger.info("=" * 60)
        logger.info(f"Generating learning path for repository: {repo}")
        logger.info("=" * 60)

        # Create GitHub analyzer agent
        github_agent = Agent(
            role="GitHub Data Fetcher",
            goal="Fetch repository data using the GitHub Project Analyzer tool",
            backstory="You fetch data from GitHub. You MUST use the GitHub Project Analyzer tool. Do not make up any data.",
            tools=[self.github_tool],
            llm=self.model,
            verbose=True,
            allow_delegation=False
        )

        # Create writer agent
        writer_agent = Agent(
            role="Learning Path Writer",
            goal="Format the GitHub data into a readable learning path",
            backstory="You format data into Markdown. Use ONLY the data provided. Do not invent anything.",
            tools=[],
            llm=self.model,
            verbose=True,
            allow_delegation=False
        )

        # Create tasks
        fetch_task = Task(
            description=f"Use the GitHub Project Analyzer tool to fetch data for repository: {repo}. Return the complete JSON response.",
            expected_output="The JSON data from the GitHub Project Analyzer tool",
            agent=github_agent
        )

        write_task = Task(
            description="""Format the GitHub data into a Markdown learning path with these sections:
# Learning Path: [repository name]

## Overview
- Description, URL, language, stars, forks

## Recent Contributors
- List each commit with author and message

## Repository Structure
- List the files

## Code Snippets
- Show any code snippets from the data

## README
- Show the README content

## Getting Started
- Clone command and basic setup

Use ONLY data from the previous task. Do not invent anything.""",
            expected_output="A Markdown learning path using only the provided data",
            agent=writer_agent,
            context=[fetch_task]
        )

        # Create and run crew
        crew = Crew(
            agents=[github_agent, writer_agent],
            tasks=[fetch_task, write_task],
            process=Process.sequential,
            verbose=True
        )

        logger.info("Executing crew...")
        result = crew.kickoff()

        result_str = str(result)
        markdown_content = extract_markdown_from_response(result_str)

        logger.info("Successfully generated Learning Path")
        return {
            "repository": repo,
            "documentation": markdown_content,
            "format": "markdown"
        }

    def save_documentation(self, documentation: Dict, output_file: Optional[str] = None) -> str:
        """Save learning path to a Markdown file."""
        if not output_file:
            repo = documentation.get("repository", "unknown")
            safe_repo = sanitize_filename(repo.replace('/', '_'))
            output_file = f"learning_path_{safe_repo}.md"

        content = documentation.get("documentation", "")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Learning path saved to {output_file}")
        return output_file

    def answer_code_question(
        self,
        repo: str,
        question: str,
        directory: str = ".",
        chat_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Answer a question about the repository code in a conversational way.

        Args:
            repo: GitHub repository in format 'owner/repo'
            question: The user's question
            directory: Directory to search for code
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
        """
        if not validate_github_repo(repo):
            raise ValueError(f"Invalid repository format: {repo}. Expected format: owner/repo")

        logger.info("=" * 60)
        logger.info(f"Code Chat for repository: {repo}")
        logger.info(f"Question: {question}")
        logger.info(f"Directory: {directory}/")
        logger.info("=" * 60)

        code_qa_tool = GitHubCodeQATool()

        # Build conversation context
        context_str = ""
        if chat_history and len(chat_history) > 0:
            context_str = "\n\nPREVIOUS CONVERSATION:\n"
            for msg in chat_history[-6:]:  # Last 6 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:500]  # Truncate long messages
                context_str += f"{role}: {content}\n"
            context_str += "\nUse this context to understand follow-up questions.\n"

        code_qa_agent = Agent(
            role="Code Assistant",
            goal="Help users understand the codebase through conversation",
            backstory="""You are a friendly code assistant that helps developers understand codebases.
You have a conversation with the user, answering their questions about the code.
Use the GitHub Code Q&A tool to fetch actual code, then explain it clearly.
For project structure questions, use directory="." to search from the root.
If no files are found, try different directories like "src", "app", or "lib".
For follow-up questions, refer to the previous conversation context.
Be concise but helpful. Use code examples when relevant.""",
            tools=[code_qa_tool],
            llm=self.model,
            verbose=True,
            allow_delegation=False
        )

        qa_task = Task(
            description=f"""You are chatting with a user about the repository: {repo}
{context_str}
USER'S CURRENT QUESTION: {question}

Instructions:
1. Use the GitHub Code Q&A tool to fetch code from the "{directory}" directory
2. Answer the user's question based on the actual code
3. If this is a follow-up question, use the conversation context
4. Quote relevant code snippets in your answer
5. Be conversational and helpful""",
            expected_output="A helpful, conversational answer about the code with examples",
            agent=code_qa_agent
        )

        crew = Crew(
            agents=[code_qa_agent],
            tasks=[qa_task],
            process=Process.sequential,
            verbose=True
        )

        logger.info("Executing Code Chat agent...")
        result = crew.kickoff()

        return {
            "repository": repo,
            "question": question,
            "directory": directory,
            "answer": str(result)
        }
