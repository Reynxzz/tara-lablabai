"""Core crew orchestration for documentation generation"""
import json
from typing import Dict, Optional, List
from crewai import Agent, Task, Crew, Process

from src.config.settings import get_settings
from src.config.constants import LLMModel, DEFAULT_TEMPERATURE_TOOL_CALLING, DEFAULT_TEMPERATURE_WRITING
from src.tools import GitLabMCPTool, GoogleDriveMCPTool, RAGMilvusTool
from src.llm import create_tool_calling_llm, create_writing_llm
from src.agents import (
    create_gitlab_analyzer_agent,
    create_drive_analyzer_agent,
    create_rag_analyzer_agent,
    create_documentation_writer_agent
)
from src.utils.logger import setup_logger
from src.utils.validators import validate_gitlab_project, sanitize_filename

logger = setup_logger(__name__)


class DocumentationCrew:
    """
    CrewAI setup for generating documentation from GitLab projects using collaborating agents.

    Architecture:
    - Agent 1 (GitLab Data Analyzer): Uses gpt-oss LLM with GitLab tool to fetch project data
    - Agent 2 (Google Drive Analyzer): [Optional] Uses gpt-oss LLM to search Google Drive for reference docs
    - Agent 3 (RAG Analyzer): [Optional] Uses gpt-oss LLM to search internal knowledge base
    - Agent 4 (Documentation Writer): Uses sahabat-4bit LLM to write structured JSON documentation

    This design leverages gpt-oss for tool calling capabilities and sahabat-4bit for
    efficient documentation generation.
    """

    def __init__(
        self,
        enable_google_drive: bool = False,
        enable_rag: bool = False
    ):
        """
        Initialize DocumentationCrew with optional integrations.

        Args:
            enable_google_drive: Whether to enable Google Drive search for reference documentation
            enable_rag: Whether to enable internal knowledge base search
        """
        logger.info("Initializing DocumentationCrew")

        settings = get_settings()
        self.enable_google_drive = enable_google_drive
        self.enable_rag = enable_rag

        # Initialize tools
        logger.info("Initializing tools...")
        self.gitlab_tool = GitLabMCPTool()

        self.drive_tool = None
        if enable_google_drive:
            if settings.google_drive.is_configured():
                self.drive_tool = GoogleDriveMCPTool()
                if self.drive_tool.is_available():
                    logger.info("✅ Google Drive integration enabled")
                else:
                    logger.warning("⚠️  Google Drive integration disabled (MCP server not reachable)")
                    self.enable_google_drive = False
            else:
                logger.warning("⚠️  Google Drive integration disabled (no GOOGLE_DRIVE_TOKEN)")
                self.enable_google_drive = False

        self.rag_tool = None
        if enable_rag:
            if settings.rag.is_configured():
                self.rag_tool = RAGMilvusTool()
                if self.rag_tool.is_available():
                    logger.info("✅ RAG Milvus integration enabled")
                else:
                    logger.warning("⚠️  RAG Milvus integration disabled (database not available)")
                    self.enable_rag = False
            else:
                logger.warning("⚠️  RAG Milvus integration disabled (database not found)")
                self.enable_rag = False

        # Create LLM instances
        logger.info("Initializing dual-LLM architecture:")
        logger.info("  - gpt-oss (GPT OSS 120B): Tool calling & data fetching")
        logger.info("  - sahabat-4bit (Sahabat AI 70B 4-bit): Documentation writing")

        self.gpt_oss_llm = create_tool_calling_llm(
            endpoint=settings.llm.endpoint,
            model=LLMModel.GPT_OSS,
            temperature=DEFAULT_TEMPERATURE_TOOL_CALLING
        )

        self.sahabat_llm = create_writing_llm(
            endpoint=settings.llm.endpoint,
            model=LLMModel.SAHABAT_4BIT,
            temperature=DEFAULT_TEMPERATURE_WRITING
        )

    def _create_gitlab_fetch_task(self, agent: Agent, project: str) -> Task:
        """Create a task for fetching GitLab project data."""
        return Task(
            description=(
                f'TASK: Fetch comprehensive data from the GitLab project "{project}".\n\n'
                f'CRITICAL REQUIREMENT: You MUST call the "GitLab Project Analyzer" tool with the project path "{project}". '
                f'Do NOT attempt to provide information from your knowledge base. '
                f'Do NOT skip the tool call. '
                f'The tool call is MANDATORY and must be executed first.\n\n'
                f'STEPS:\n'
                f'1. IMMEDIATELY call the GitLab Project Analyzer tool with input: {project}\n'
                f'2. Wait for the tool response containing all project data\n'
                f'3. Parse and present the complete raw data from the tool response\n'
                f'4. Extract and organize the following information:\n'
                f'   - Project name, description, default branch, visibility, and license\n'
                f'   - Stars, forks, open issues count\n'
                f'   - Topics/tags\n'
                f'   - Main files and their purposes\n'
                f'   - Recent commit activity\n'
                f'   - README content (if available)\n'
                f'   - Last activity date\n\n'
                f'IMPORTANT: Provide ALL the raw data you fetch from the tool. Be thorough and comprehensive. '
                f'Your output will be used by the documentation writer to create the final documentation. '
                f'If you have not called the tool yet, STOP and call it now.'
            ),
            expected_output=(
                'A comprehensive report that begins with confirmation of tool usage, followed by all fetched GitLab project data including:\n'
                '- Confirmation: "Tool called successfully with project: {project}"\n'
                '- Raw tool response data\n'
                '- Project metadata (name, description, default branch, visibility, license, URL)\n'
                '- Community metrics (stars, forks, issues)\n'
                '- File structure and key files\n'
                '- Recent activity and commits\n'
                '- Any additional relevant information from the project'
            ),
            agent=agent
        )

    def _create_drive_search_task(self, agent: Agent, project: str) -> Task:
        """Create a task for searching Google Drive for reference documentation."""
        return Task(
            description=(
                f'TASK: Search Google Drive for reference documentation related to the GitLab project "{project}".\n\n'
                f'CRITICAL REQUIREMENT: You MUST call the "Google Drive Document Analyzer" tool (NOT the GitLab tool). '
                f'Do NOT confuse GitLab project files with Google Drive documents. '
                f'Do NOT report GitLab file structure as if it came from Google Drive. '
                f'Do NOT fabricate or make up document content. '
                f'Do NOT skip the tool call. '
                f'The tool call is MANDATORY and must be executed first.\n\n'
                f'IMPORTANT DISTINCTION:\n'
                f'- GitLab tool returns: directories (common, src, config), Python files (.py), config files (.json)\n'
                f'- Google Drive tool returns: Google Docs, Google Sheets with names like "DGE Serving Checklist" and URIs like "gdrive:///xxxxx"\n'
                f'- If you see directories or .py files, you are looking at the WRONG tool output!\n\n'
                f'STEPS:\n'
                f'1. IMMEDIATELY call the Google Drive Document Analyzer tool (NOT GitLab) with input: {project}\n'
                f'2. Wait for the tool response - it should contain Google Docs/Sheets with URIs starting with "gdrive:///"\n'
                f'3. Verify the response contains Google Drive documents (NOT GitLab files)\n'
                f'4. If the tool returns Google Drive documents, summarize the ACTUAL document content\n'
                f'5. If the tool returns no documents or an error, report exactly: "No relevant documentation found in Google Drive"\n\n'
                f'VERIFICATION: Before reporting results, confirm:\n'
                f'- Did I call the Google Drive Document Analyzer tool? (NOT GitLab)\n'
                f'- Do the results have URIs starting with "gdrive:///"? (NOT GitLab URLs)\n'
                f'- Are these Google Docs/Sheets? (NOT Python files or directories)\n'
                f'If NO to any question above, you are reporting the WRONG tool output!'
            ),
            expected_output=(
                'A summary that proves Google Drive tool was called correctly:\n'
                '- Confirmation: "Called Google Drive Document Analyzer tool with query: {query}"\n'
                '- Number of Google Drive documents found (NOT GitLab files)\n'
                '- For EACH Google Drive document: Name, URI (must start with "gdrive:///"), and brief content summary\n'
                '- Key insights from the Google Drive document content (NOT from GitLab repository)\n'
                '- If tool returned no results: "No relevant documentation found in Google Drive"\n'
                '- NEVER report GitLab file structure (common, src, config, .py files) as Google Drive results'
            ),
            agent=agent
        )

    def _create_rag_search_task(self, agent: Agent, project: str) -> Task:
        """Create a task for searching internal knowledge base."""
        return Task(
            description=(
                f'TASK: Search the internal knowledge base for information relevant to the GitLab project "{project}".\n\n'
                f'CRITICAL REQUIREMENT: You MUST call the "Internal Knowledge Base Search" tool. '
                f'Do NOT fabricate or make up knowledge base content. '
                f'Do NOT skip the tool call. '
                f'Do NOT provide information from memory or training data. '
                f'The tool call is MANDATORY and must be executed first.\n\n'
                f'STEPS:\n'
                f'1. IMMEDIATELY call the Internal Knowledge Base Search tool with input: {project}\n'
                f'2. Wait for the tool response containing search results\n'
                f'3. If the tool returns results, summarize ONLY the actual content from the tool response\n'
                f'4. If the tool returns no results or an error, report exactly: "No relevant information found in internal knowledge base"\n'
                f'5. Try additional search queries ONLY if the first search returns results but they are not relevant:\n'
                f'   - Technical terms or concepts from the project\n'
                f'   - User segments or data features the project might use\n\n'
                f'IMPORTANT: You must ONLY report findings that came from the tool response. '
                f'If you have not called the tool yet, STOP and call it now. '
                f'Never fabricate, infer, or make up knowledge base content. '
                f'Your output will be verified against tool usage logs.'
            ),
            expected_output=(
                'A summary that begins with tool usage confirmation, followed by search results:\n'
                '- Confirmation: "Tool called successfully with query: {query}"\n'
                '- Number of results found (from tool response)\n'
                '- Collection searched (from tool response)\n'
                '- Key insights based ONLY on tool response content\n'
                '- How the project relates to internal systems (based ONLY on tool results)\n'
                '- If tool returned no results: "No relevant information found in internal knowledge base"\n'
                '- NEVER include fabricated or assumed knowledge base content'
            ),
            agent=agent
        )

    def _create_documentation_writing_task(self, agent: Agent, project: str) -> Task:
        """Create a task for writing documentation based on fetched data."""
        return Task(
            description=(
                f'TASK: Generate comprehensive documentation for the GitLab project "{project}" '
                f'based on the data provided by previous agents.\n\n'
                f'STEPS:\n'
                f'1. Review all project data provided by the previous agents\n'
                f'2. Create a comprehensive JSON document with these sections:\n'
                f'   - overview: {{name, description, purpose, default_branch, visibility, license}}\n'
                f'   - features: [list of key features]\n'
                f'   - tech_stack: {{topics, dependencies}}\n'
                f'   - structure: {{main_files: [files with descriptions]}}\n'
                f'   - activity: {{stars, forks, open_issues, last_activity}}\n'
                f'   - getting_started: {{installation, usage, project_url}}\n\n'
                f'IMPORTANT: Return ONLY the JSON object, no markdown code blocks, no extra text. '
                f'Start your response directly with {{.'
            ),
            expected_output=(
                'A valid, complete JSON object (not wrapped in markdown code blocks) following this structure:\n'
                '{\n'
                '  "overview": {"name": "...", "description": "...", "purpose": "...", "default_branch": "...", "visibility": "...", "license": "..."},\n'
                '  "features": ["feature1", "feature2"],\n'
                '  "tech_stack": {"topics": [], "dependencies": "..."},\n'
                '  "structure": {"main_files": [{"name": "...", "purpose": "..."}]},\n'
                '  "activity": {"stars": 0, "forks": 0, "open_issues": 0, "last_activity": "..."},\n'
                '  "getting_started": {"installation": "...", "usage": "...", "project_url": "..."}\n'
                '}'
            ),
            agent=agent
        )

    def generate_documentation(self, project: str) -> Dict:
        """
        Generate documentation for a given GitLab project using collaborating agents.

        Args:
            project: GitLab project in format 'namespace/project'

        Returns:
            Dictionary containing the generated documentation

        Raises:
            ValueError: If project format is invalid
        """
        if not validate_gitlab_project(project):
            raise ValueError(f"Invalid project format: {project}. Expected format: namespace/project")

        # Calculate agent count
        agent_count = 2  # GitLab + Writer (minimum)
        if self.enable_google_drive:
            agent_count += 1
        if self.enable_rag:
            agent_count += 1

        logger.info("=" * 60)
        logger.info(f"Starting {agent_count}-agent collaboration for project: {project}")
        logger.info(f"  Agent 1 (gpt-oss): Fetching GitLab data...")
        if self.enable_google_drive:
            logger.info(f"  Agent 2 (gpt-oss): Searching Google Drive for references...")
        if self.enable_rag:
            logger.info(f"  Agent {3 if self.enable_google_drive else 2} (gpt-oss): Searching internal knowledge base...")
        logger.info(f"  Agent {agent_count} (sahabat-4bit): Writing documentation...")
        logger.info("=" * 60)

        # Create agents and tasks
        agents: List[Agent] = []
        tasks: List[Task] = []

        # GitLab analyzer agent and task (required)
        gitlab_analyzer = create_gitlab_analyzer_agent(self.gpt_oss_llm, self.gitlab_tool)
        agents.append(gitlab_analyzer)
        tasks.append(self._create_gitlab_fetch_task(gitlab_analyzer, project))

        # Google Drive analyzer agent and task (optional)
        if self.enable_google_drive and self.drive_tool:
            drive_analyzer = create_drive_analyzer_agent(self.gpt_oss_llm, self.drive_tool)
            agents.append(drive_analyzer)
            tasks.append(self._create_drive_search_task(drive_analyzer, project))

        # RAG analyzer agent and task (optional)
        if self.enable_rag and self.rag_tool:
            rag_analyzer = create_rag_analyzer_agent(self.gpt_oss_llm, self.rag_tool)
            agents.append(rag_analyzer)
            tasks.append(self._create_rag_search_task(rag_analyzer, project))

        # Documentation writer agent and task (required)
        doc_writer = create_documentation_writer_agent(self.sahabat_llm)
        agents.append(doc_writer)
        tasks.append(self._create_documentation_writing_task(doc_writer, project))

        # Create crew with sequential process
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        logger.info("Executing crew...")
        result = crew.kickoff()

        # Parse result as JSON
        try:
            result_str = str(result)

            # Strip markdown code blocks if present
            if result_str.strip().startswith('```'):
                start = result_str.find('\n')
                end = result_str.rfind('```')
                if start != -1 and end != -1:
                    result_str = result_str[start+1:end].strip()

            doc_json = json.loads(result_str)
            logger.info("Successfully parsed documentation as JSON")
            return doc_json

        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse result as JSON: {e}")
            return {
                "project": project,
                "documentation": str(result),
                "format": "text",
                "note": "Documentation could not be parsed as JSON, returning as text"
            }

    def save_documentation(self, documentation: Dict, output_file: Optional[str] = None) -> str:
        """
        Save documentation to a JSON file.

        Args:
            documentation: Documentation dictionary
            output_file: Output file path (optional, auto-generated if not provided)

        Returns:
            Path to the saved file
        """
        if not output_file:
            project = documentation.get("project", "unknown")
            safe_project = sanitize_filename(project.replace('/', '_'))
            output_file = f"documentation_{safe_project}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documentation, f, indent=2, ensure_ascii=False)

        logger.info(f"Documentation saved to {output_file}")
        return output_file
