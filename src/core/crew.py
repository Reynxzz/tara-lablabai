"""Core crew orchestration for documentation generation"""
import json
import re
from typing import Dict, Optional, List
from crewai import Agent, Task, Crew, Process

from src.config.settings import get_settings
from src.config.constants import LLMModel, DEFAULT_TEMPERATURE_TOOL_CALLING, DEFAULT_TEMPERATURE_WRITING
from src.tools import GitHubTool, GoogleDriveMCPTool, GitHubCodeQATool
from src.llm import create_tool_calling_llm, create_writing_llm
from src.agents import (
    create_github_analyzer_agent,
    create_drive_analyzer_agent,
    create_documentation_writer_agent,
    create_code_qa_agent
)
from src.utils.logger import setup_logger
from src.utils.validators import validate_github_repo, sanitize_filename

logger = setup_logger(__name__)


def extract_markdown_from_response(response: str) -> str:
    """
    Extract and clean markdown content from various response formats.

    Handles:
    - JSON responses with markdown content
    - Markdown wrapped in code blocks
    - Raw markdown text

    Args:
        response: The raw response string from the agent

    Returns:
        Clean markdown text
    """
    content = response.strip()

    # Try to parse as JSON first
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            # Look for markdown content in common keys
            for key in ["markdown_documentation", "documentation", "content", "markdown"]:
                if key in data and data[key]:
                    content = data[key]
                    break
    except (json.JSONDecodeError, ValueError):
        pass

    # Remove markdown code blocks if present
    if content.startswith('```markdown') or content.startswith('```md'):
        content = re.sub(r'^```(?:markdown|md)\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
    elif content.startswith('```'):
        content = re.sub(r'^```\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)

    return content.strip()


class DocumentationCrew:
    """
    CrewAI setup for generating documentation from GitHub repositories using collaborating agents.

    Architecture:
    - Agent 1 (GitHub Data Analyzer): Uses gpt-4o-mini LLM with GitHub tool to fetch project data
    - Agent 2 (Google Drive Analyzer): [Optional] Uses gpt-4o-mini LLM to search Google Drive for reference docs
    - Agent 3 (Documentation Writer): Uses gpt-4o LLM to write structured learning path documentation

    This design leverages gpt-4o-mini for tool calling capabilities (lower cost) and gpt-4o for
    documentation generation (better quality).
    """

    def __init__(
        self,
        enable_google_drive: bool = False
    ):
        """
        Initialize DocumentationCrew with optional integrations.

        Args:
            enable_google_drive: Whether to enable Google Drive search for reference documentation
        """
        logger.info("Initializing DocumentationCrew")

        settings = get_settings()
        self.enable_google_drive = enable_google_drive

        # Initialize tools
        logger.info("Initializing tools...")
        self.github_tool = GitHubTool()

        self.drive_tool = None
        if enable_google_drive:
            if settings.google_drive.is_configured():
                self.drive_tool = GoogleDriveMCPTool()
                if self.drive_tool.is_available():
                    logger.info("Google Drive integration enabled")
                else:
                    logger.warning("Google Drive integration disabled (MCP server not reachable)")
                    self.enable_google_drive = False
            else:
                logger.warning("Google Drive integration disabled (no GOOGLE_DRIVE_TOKEN)")
                self.enable_google_drive = False

        # Create LLM instances
        logger.info("Initializing dual-LLM architecture:")
        logger.info("  - gpt-4o-mini: Tool calling & data fetching")
        logger.info("  - gpt-4o: Documentation writing")

        self.tool_calling_llm = create_tool_calling_llm(
            api_key=settings.llm.api_key,
            model=LLMModel.GPT_4O_MINI,
            temperature=DEFAULT_TEMPERATURE_TOOL_CALLING
        )

        self.writing_llm = create_writing_llm(
            api_key=settings.llm.api_key,
            model=LLMModel.GPT_4O,
            temperature=DEFAULT_TEMPERATURE_WRITING
        )

    def _create_github_fetch_task(self, agent: Agent, repo: str) -> Task:
        """Create a task for fetching GitHub repository data."""
        return Task(
            description=(
                f'TASK: Fetch comprehensive data from the GitHub repository "{repo}" for LEARNING PATH GENERATION.\n\n'
                f'CRITICAL REQUIREMENT: You MUST call the "GitHub Project Analyzer" tool with the repository path "{repo}". '
                f'Do NOT attempt to provide information from your knowledge base. '
                f'Do NOT skip the tool call. '
                f'The tool call is MANDATORY and must be executed first.\n\n'
                f'STEPS:\n'
                f'1. IMMEDIATELY call the GitHub Project Analyzer tool with input: {repo}\n'
                f'2. Wait for the tool response containing all repository data\n'
                f'3. Parse and present the complete raw data from the tool response\n'
                f'4. Extract and organize the following information:\n'
                f'   - Repository name, description, default branch, visibility, and license\n'
                f'   - Stars, forks, open issues count\n'
                f'   - Topics/tags\n'
                f'   - Main files and their purposes WITH LINKS\n'
                f'   - Recent commit activity with contributor names\n'
                f'   - README content (if available)\n'
                f'   - Code snippets from key files (with links to full files)\n'
                f'   - Last activity date\n\n'
                f'IMPORTANT: Provide ALL the raw data you fetch from the tool. Be thorough and comprehensive. '
                f'Your output will be used by the Learning Path writer to create the final learning path. '
                f'Make sure to include all links (repository URL, file links from code_snippets section). '
                f'If you have not called the tool yet, STOP and call it now.'
            ),
            expected_output=(
                'A comprehensive report that begins with confirmation of tool usage, followed by all fetched GitHub repository data including:\n'
                '- Confirmation: "GitHub Project Analyzer tool called successfully with repository: {repo}"\n'
                '- Raw tool response data\n'
                '- Repository metadata (name, description, default branch, visibility, license, URL)\n'
                '- Community metrics (stars, forks, issues)\n'
                '- File structure and key files\n'
                '- Code snippets with links to full files (from code_snippets field in tool response)\n'
                '- Recent activity and commits with contributor names\n'
                '- Any additional relevant information from the repository\n'
            ),
            agent=agent
        )

    def _create_drive_search_task(self, agent: Agent, repo: str) -> Task:
        """Create a task for searching Google Drive for reference documentation."""
        return Task(
            description=(
                f'TASK: Search Google Drive for reference documentation related to the GitHub repository "{repo}".\n\n'
                f'CRITICAL REQUIREMENT: You MUST call the "Google Drive Document Analyzer" tool (NOT the GitHub tool). '
                f'Do NOT confuse GitHub repository files with Google Drive documents. '
                f'Do NOT report GitHub file structure as if it came from Google Drive. '
                f'Do NOT fabricate or make up document content. '
                f'Do NOT skip the tool call. '
                f'The tool call is MANDATORY and must be executed first.\n\n'
                f'IMPORTANT DISTINCTION:\n'
                f'- GitHub tool returns: directories (common, src, config), Python files (.py), config files (.json)\n'
                f'- Google Drive tool returns: Google Docs, Google Sheets with names like "DGE Serving Checklist" and URIs like "gdrive:///xxxxx"\n'
                f'- If you see directories or .py files, you are looking at the WRONG tool output!\n\n'
                f'STEPS:\n'
                f'1. IMMEDIATELY call the Google Drive Document Analyzer tool (NOT GitHub) with input: {repo}\n'
                f'2. Wait for the tool response - it should contain Google Docs/Sheets with both "uri" and "url" fields\n'
                f'3. Verify the response contains Google Drive documents (NOT GitHub files)\n'
                f'4. If the tool returns Google Drive documents, extract:\n'
                f'   - Document names and clickable URLs (use the "url" field, NOT "uri")\n'
                f'   - The "url" field contains full links like https://docs.google.com/document/d/...\n'
                f'   - Key definitions, concepts, or important points from each document\n'
                f'   - Why each document is relevant (what users will learn from it)\n'
                f'5. If the tool returns no documents or an error, report exactly: "No relevant documentation found in Google Drive"\n\n'
                f'VERIFICATION: Before reporting results, confirm:\n'
                f'- Did I call the Google Drive Document Analyzer tool? (NOT GitHub)\n'
                f'- Do the results have URIs starting with "gdrive:///"? (NOT GitHub URLs)\n'
                f'- Are these Google Docs/Sheets? (NOT Python files or directories)\n'
                f'If NO to any question above, you are reporting the WRONG tool output!'
            ),
            expected_output=(
                'A summary that proves Google Drive tool was called correctly:\n'
                '- Confirmation: "Called Google Drive Document Analyzer tool with query: {query}"\n'
                '- Number of Google Drive documents found (NOT GitHub files)\n'
                '- For EACH Google Drive document:\n'
                '  - Name and clickable URL (use the "url" field from tool response, NOT "uri")\n'
                '  - The "url" field contains the full Google Docs/Sheets link (https://docs.google.com/...)\n'
                '  - Key definitions or important points extracted from the document\n'
                '  - Why this document is useful for learning about the project\n'
                '- Key insights from the Google Drive document content (NOT from GitHub repository)\n'
                '- If tool returned no results: "No relevant documentation found in Google Drive"\n'
                '- NEVER report GitHub file structure (common, src, config, .py files) as Google Drive results'
            ),
            agent=agent
        )

    def _create_learning_path_writing_task(self, agent: Agent, repo: str) -> Task:
        """Create a task for writing a learning path based on fetched data."""
        return Task(
            description=(
                f'TASK: Generate a "Learning Path" in Markdown format for the GitHub repository "{repo}" '
                f'that guides users to the right resources for onboarding.\n\n'
                f'CRITICAL: This is a LEARNING PATH, not comprehensive documentation. Your goal is to tell users '
                f'WHICH documents and resources to read, not to explain everything in detail.\n\n'
                f'STEPS:\n'
                f'1. Review all data provided by previous agents (GitHub, Google Drive)\n'
                f'2. Extract ALL valid links from the agents\' outputs\n'
                f'3. Create a Learning Path with the structure below:\n\n'
                f'## Required Learning Path Structure:\n\n'
                f'# Learning Path: [Project Name]\n\n'
                f'## Overview\n'
                f'Synthesize a brief overview from ALL sources (GitHub description, Drive docs):\n'
                f'- What this project does (combine insights from all sources, not just GitHub description)\n'
                f'- Key purpose and use cases\n'
                f'- Technologies used\n'
                f'- Project metadata: [Project URL](link), License, Default Branch\n\n'
                f'## Recent Contributors\n'
                f'- List recent commit authors with their latest contributions\n'
                f'- Include commit titles and dates\n'
                f'- Keep this section as-is from GitHub data\n\n'
                f'## Repository Structure\n'
                f'List key directories and files with **clickable links**:\n'
                f'- [filename](https://github.com/[owner]/[repo]/blob/[branch]/[filepath]) - Brief purpose\n'
                f'- Focus on important entry points and configuration files\n\n'
                f'## Code Snippets (First Look)\n'
                f'Show code snippets from key files (if available from GitHub tool):\n'
                f'- Display the snippet with syntax highlighting\n'
                f'- Include link to full file: [View full file](link)\n'
                f'- Brief explanation of what this file does\n\n'
                f'## Reference Documentation\n'
                f'**From Google Drive** (if available):\n'
                f'- [Document Name](url) - Use the "url" field from Drive tool results (NOT "uri")\n'
                f'- The "url" field contains clickable Google Docs/Sheets links (https://docs.google.com/...)\n'
                f'- Extract important definitions, concepts, or guidelines mentioned in the Drive search results\n'
                f'- Tell users WHY they should read each document\n\n'
                f'## Getting Started\n'
                f'Guide users to the right resources:\n'
                f'- "Start by reading [README](link)"\n'
                f'- "Check configuration in [config file](link)"\n'
                f'- "Review setup instructions in [Drive doc](link)" (if available)\n'
                f'- Installation steps (brief, with links to detailed docs)\n\n'
                f'IMPORTANT FORMATTING RULES:\n'
                f'- Every resource MUST have a valid clickable link\n'
                f'- Use format: [Text](url) for all links\n'
                f'- Code snippets should use ```language syntax\n'
                f'- Extract key points from Drive documents, don\'t just list them\n'
                f'- Synthesize the overview from multiple sources\n'
                f'- DO NOT wrap in JSON or code blocks\n'
                f'- Return ONLY raw markdown starting with # header\n'
            ),
            expected_output=(
                'A complete Learning Path in Markdown with:\n'
                '- Project overview synthesized from GitHub and Drive sources\n'
                '- Recent contributors section (unchanged from GitHub)\n'
                '- Repository structure with clickable file links\n'
                '- Code snippets with links to full files\n'
                '- Reference documentation with key points/definitions extracted\n'
                '- For Google Drive links: Use the "url" field (https://docs.google.com/...), NOT "uri" (gdrive:///...)\n'
                '- Getting Started guide with links to resources\n'
                '- All links properly formatted as [text](url)\n'
                '- Focus on GUIDING users to resources, not explaining everything\n'
            ),
            agent=agent
        )

    # Backward compatibility alias
    _create_documentation_writing_task = _create_learning_path_writing_task

    def generate_documentation(self, repo: str) -> Dict:
        """
        Generate documentation for a given GitHub repository using collaborating agents.

        Args:
            repo: GitHub repository in format 'owner/repo'

        Returns:
            Dictionary containing the generated documentation

        Raises:
            ValueError: If repository format is invalid
        """
        if not validate_github_repo(repo):
            raise ValueError(f"Invalid repository format: {repo}. Expected format: owner/repo")

        # Calculate agent count
        agent_count = 2  # GitHub + Writer (minimum)
        if self.enable_google_drive:
            agent_count += 1

        logger.info("=" * 60)
        logger.info(f"Starting {agent_count}-agent collaboration for repository: {repo}")
        logger.info(f"  Agent 1 (gpt-4o-mini): Fetching GitHub data with code snippets...")
        if self.enable_google_drive:
            logger.info(f"  Agent 2 (gpt-4o-mini): Searching Google Drive for reference docs...")
        logger.info(f"  Agent {agent_count} (gpt-4o): Writing learning path...")
        logger.info("=" * 60)

        # Create agents and tasks
        agents: List[Agent] = []
        tasks: List[Task] = []

        # GitHub analyzer agent and task (required)
        github_analyzer = create_github_analyzer_agent(self.tool_calling_llm, self.github_tool)
        agents.append(github_analyzer)
        tasks.append(self._create_github_fetch_task(github_analyzer, repo))

        # Google Drive analyzer agent and task (optional)
        if self.enable_google_drive and self.drive_tool:
            drive_analyzer = create_drive_analyzer_agent(self.tool_calling_llm, self.drive_tool)
            agents.append(drive_analyzer)
            tasks.append(self._create_drive_search_task(drive_analyzer, repo))

        # Documentation writer agent and task (required)
        doc_writer = create_documentation_writer_agent(self.writing_llm)
        agents.append(doc_writer)
        tasks.append(self._create_documentation_writing_task(doc_writer, repo))

        # Create crew with sequential process
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        logger.info("Executing crew...")
        result = crew.kickoff()

        # Convert result to string and extract markdown
        result_str = str(result)
        markdown_content = extract_markdown_from_response(result_str)

        logger.info("Successfully generated Learning Path in Markdown format")
        return {
            "repository": repo,
            "documentation": markdown_content,
            "format": "markdown"
        }

    def save_documentation(self, documentation: Dict, output_file: Optional[str] = None) -> str:
        """
        Save learning path to a Markdown file.

        Args:
            documentation: Learning path dictionary with 'documentation' and 'format' keys
            output_file: Output file path (optional, auto-generated if not provided)

        Returns:
            Path to the saved file
        """
        if not output_file:
            repo = documentation.get("repository", "unknown")
            safe_repo = sanitize_filename(repo.replace('/', '_'))
            doc_format = documentation.get("format", "markdown")
            extension = ".md" if doc_format == "markdown" else ".txt"
            output_file = f"learning_path_{safe_repo}{extension}"

        # Get the documentation content
        content = documentation.get("documentation", "")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Learning path saved to {output_file}")
        return output_file

    def answer_code_question(self, repo: str, question: str, directory: str = "src") -> Dict:
        """
        Answer a specific question about the repository code.

        Args:
            repo: GitHub repository in format 'owner/repo'
            question: Question about the codebase
            directory: Directory to search (default: src)

        Returns:
            Dictionary containing the answer and relevant code references

        Raises:
            ValueError: If repository format is invalid
        """
        if not validate_github_repo(repo):
            raise ValueError(f"Invalid repository format: {repo}. Expected format: owner/repo")

        logger.info("=" * 60)
        logger.info(f"Code Q&A for repository: {repo}")
        logger.info(f"Question: {question}")
        logger.info(f"Searching in: {directory}/")
        logger.info("=" * 60)

        # Initialize Code Q&A tool
        code_qa_tool = GitHubCodeQATool()

        # Create Code Q&A agent with tool calling LLM
        code_qa_agent = create_code_qa_agent(self.tool_calling_llm, code_qa_tool)

        # Create task for answering the question
        qa_task = Task(
            description=(
                f'TASK: Answer the following question about the GitHub repository "{repo}":\n\n'
                f'Question: {question}\n\n'
                f'CRITICAL REQUIREMENT: You MUST call the "GitHub Code Q&A" tool with:\n'
                f'- repo: {repo}\n'
                f'- question: {question}\n'
                f'- directory: {directory}\n\n'
                f'STEPS:\n'
                f'1. IMMEDIATELY call the GitHub Code Q&A tool with the parameters above\n'
                f'2. Wait for the tool response containing code files\n'
                f'3. Analyze ONLY the code provided by the tool\n'
                f'4. Answer the question based on actual code content\n'
                f'5. Quote specific code snippets (with file references)\n'
                f'6. Provide file links for all referenced code\n\n'
                f'IMPORTANT:\n'
                f'- DO NOT make assumptions about code that is not in the tool response\n'
                f'- DO NOT use your training data to answer\n'
                f'- ONLY analyze code files returned by the tool\n'
                f'- If no relevant code is found, say so explicitly\n'
                f'- Always include file links (use the "link" field from tool response)\n'
            ),
            expected_output=(
                'A comprehensive answer that includes:\n'
                '- Direct answer to the question\n'
                '- Specific code examples (quoted from tool response)\n'
                '- File references with clickable links for each code snippet\n'
                '- Summary of findings\n'
                '- If no relevant code found: "No relevant code found in the specified directory"\n'
            ),
            agent=code_qa_agent
        )

        # Create single-agent crew
        crew = Crew(
            agents=[code_qa_agent],
            tasks=[qa_task],
            process=Process.sequential,
            verbose=True
        )

        logger.info("Executing Code Q&A agent...")
        result = crew.kickoff()

        # Convert result to string
        result_str = str(result)

        logger.info("Successfully answered code question")
        return {
            "repository": repo,
            "question": question,
            "directory": directory,
            "answer": result_str
        }
