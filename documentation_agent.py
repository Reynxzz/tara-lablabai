"""Documentation Generation Agent for GitLab using CrewAI"""
import os
import json
from crewai import Agent, Task, Crew, Process
from github_mcp_tool import GitLabMCPTool
from google_drive_mcp_tool import GoogleDriveMCPTool
from goto_custom_llm import GoToCustomLLM


class DocumentationCrew:
    """
    CrewAI setup for generating documentation from GitLab projects using two collaborating agents.

    Architecture:
    - Agent 1 (GitLab Data Analyzer): Uses gpt-oss LLM with GitLab tool to fetch project data
    - Agent 2 (Google Drive Analyzer): [Optional] Uses gpt-oss LLM to search Google Drive for reference docs
    - Agent 3 (Documentation Writer): Uses sahabat-4bit LLM to write structured JSON documentation

    This design leverages gpt-oss for tool calling capabilities and sahabat-4bit for
    efficient documentation generation.
    """

    def __init__(self, enable_google_drive: bool = False):
        """
        Initialize DocumentationCrew with optional Google Drive integration.

        Args:
            enable_google_drive: Whether to enable Google Drive search for reference documentation
        """
        self.gitlab_tool = GitLabMCPTool()
        self.enable_google_drive = enable_google_drive

        # Initialize Google Drive tool if enabled
        self.drive_tool = None
        if enable_google_drive:
            drive_token = os.getenv("GOOGLE_DRIVE_TOKEN")
            if drive_token:
                self.drive_tool = GoogleDriveMCPTool(access_token=drive_token)
                if self.drive_tool.is_available():
                    print("✅ Google Drive integration enabled")
                else:
                    print("⚠️  Google Drive integration disabled (MCP server not reachable)")
                    self.enable_google_drive = False
            else:
                print("⚠️  Google Drive integration disabled (no GOOGLE_DRIVE_TOKEN)")
                self.enable_google_drive = False

        # Create two LLM instances for collaboration
        print("Initializing dual-LLM architecture:")
        print("  - gpt-oss (GPT OSS 120B): Tool calling & data fetching")
        print("  - sahabat-4bit (Sahabat AI 70B 4-bit): Documentation writing")

        # gpt-oss for tool calling (GitLab data fetching)
        # Lower temperature for more reliable tool calling
        self.gpt_oss_llm = GoToCustomLLM(
            model="openai/gpt-oss-120b",
            endpoint="https://litellm-staging.gopay.sh",
            temperature=0.3,  # Low temperature for deterministic tool calling
            supports_tools=True  # Enable tool calling support
        )

        # sahabat-4bit for documentation writing
        self.sahabat_llm = GoToCustomLLM(
            model="GoToCompany/Llama-Sahabat-AI-v2-70B-IT-awq-4bit",
            endpoint="https://litellm-staging.gopay.sh",
            temperature=0.6
        )

    def create_gitlab_analyzer_agent(self) -> Agent:
        """Create an agent specialized in fetching GitLab data using tools (uses gpt-oss)"""
        return Agent(
            role='GitLab Data Analyzer',
            goal='Always use the GitLab Project Analyzer tool to fetch real-time project data. Never rely on pre-existing knowledge.',
            backstory=(
                'You are a meticulous data analyst who specializes in extracting information from GitLab projects. '
                'Your PRIMARY and ONLY method of gathering data is through the GitLab Project Analyzer tool. '
                'You NEVER provide information from memory or training data - you ALWAYS call the tool first. '
                'Your job is to ALWAYS use the GitLab Project Analyzer tool to fetch ALL relevant data about a project '
                'including its structure, files, commits, stars, forks, and other metadata. You provide thorough, '
                'detailed information that will be used by the documentation team. '
                'Tool usage is mandatory and non-negotiable for every project analysis task.'
            ),
            tools=[self.gitlab_tool],
            llm=self.gpt_oss_llm,  # Use gpt-oss for tool calling
            verbose=True,
            allow_delegation=False
        )

    def create_drive_analyzer_agent(self) -> Agent:
        """Create an agent specialized in searching Google Drive for reference documentation (uses gpt-oss)"""
        return Agent(
            role='Google Drive Reference Analyzer',
            goal='Search Google Drive for relevant reference documentation, specifications, and related materials that can enhance project documentation.',
            backstory=(
                'You are a research specialist who excels at finding relevant reference materials '
                'in Google Drive. You use the Google Drive Document Analyzer tool to search for '
                'documentation, specifications, design docs, and related materials that provide '
                'additional context about projects. You provide concise summaries of relevant findings '
                'that help the documentation team create more comprehensive documentation.'
            ),
            tools=[self.drive_tool] if self.drive_tool else [],
            llm=self.gpt_oss_llm,  # Use gpt-oss for tool calling
            verbose=True,
            allow_delegation=False
        )

    def create_documentation_writer_agent(self) -> Agent:
        """Create an agent specialized in writing documentation (uses sahabat-4bit)"""
        return Agent(
            role='Technical Documentation Writer',
            goal='Generate comprehensive, well-structured documentation in JSON format based on GitLab project data and optional Google Drive references',
            backstory=(
                'You are an expert technical writer with deep knowledge of software architecture '
                'and documentation best practices. You take raw data about projects and transform it into '
                'clear, comprehensive documentation that helps developers understand projects quickly. '
                'You synthesize information from multiple sources including GitLab data and Google Drive '
                'reference materials to create thorough documentation. '
                'You always output documentation in valid JSON format with proper structure.'
            ),
            tools=[],  # No tools - only writes documentation based on previous agent's output
            llm=self.sahabat_llm,  # Use sahabat-4bit for documentation writing
            verbose=True,
            allow_delegation=False
        )

    def create_gitlab_fetch_task(self, agent: Agent, project: str) -> Task:
        """Create a task for fetching GitLab project data"""
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

    def create_drive_search_task(self, agent: Agent, project: str) -> Task:
        """Create a task for searching Google Drive for reference documentation"""
        return Task(
            description=(
                f'TASK: Search Google Drive for reference documentation related to the GitLab project "{project}".\n\n'
                f'STEPS:\n'
                f'1. Use the Google Drive Document Analyzer tool to search for relevant documents\n'
                f'2. Search queries to try:\n'
                f'   - Project name: "{project}"\n'
                f'   - Related keywords from the project description\n'
                f'   - Technical specifications or design docs\n'
                f'3. Review the search results and identify the most relevant documents\n'
                f'4. Summarize key findings from the reference materials\n\n'
                f'IMPORTANT: Provide a concise summary of relevant findings. '
                f'If no relevant documents are found, state that clearly. '
                f'Your findings will supplement the GitLab data in the final documentation.'
            ),
            expected_output=(
                'A summary of Google Drive search results including:\n'
                '- Number of relevant documents found\n'
                '- Brief description of each relevant document\n'
                '- Key insights or information from the documents\n'
                '- If no documents found: "No relevant documentation found in Google Drive"'
            ),
            agent=agent
        )

    def create_documentation_writing_task(self, agent: Agent, project: str) -> Task:
        """Create a task for writing documentation based on fetched data"""
        return Task(
            description=(
                f'TASK: Generate comprehensive documentation for the GitLab project "{project}" '
                f'based on the data provided by previous agents (GitLab data and optional Google Drive references).\n\n'
                f'STEPS:\n'
                f'1. Review the project data provided by the previous agents\n'
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

    def generate_documentation(self, project: str) -> dict:
        """
        Generate documentation for a given GitLab project using collaborating agents.

        Agent 1 (gpt-oss): Fetches GitLab data using tools
        Agent 2 (gpt-oss): [Optional] Searches Google Drive for reference docs
        Agent 3 (sahabat-4bit): Writes documentation based on fetched data

        Args:
            project: GitLab project in format 'namespace/project'

        Returns:
            Dictionary containing the generated documentation
        """
        agent_count = 2 if not self.enable_google_drive else 3
        print(f"\n{'='*60}")
        print(f"Starting {agent_count}-agent collaboration:")
        print(f"  Agent 1 (gpt-oss): Fetching GitLab data...")
        if self.enable_google_drive:
            print(f"  Agent 2 (gpt-oss): Searching Google Drive for references...")
            print(f"  Agent 3 (sahabat-4bit): Writing documentation...")
        else:
            print(f"  Agent 2 (sahabat-4bit): Writing documentation...")
        print(f"{'='*60}\n")

        # Create agents
        agents = []
        tasks = []

        # GitLab analyzer agent and task
        gitlab_analyzer = self.create_gitlab_analyzer_agent()
        agents.append(gitlab_analyzer)
        fetch_task = self.create_gitlab_fetch_task(gitlab_analyzer, project)
        tasks.append(fetch_task)

        # Google Drive analyzer agent and task (optional)
        if self.enable_google_drive:
            drive_analyzer = self.create_drive_analyzer_agent()
            agents.append(drive_analyzer)
            drive_task = self.create_drive_search_task(drive_analyzer, project)
            tasks.append(drive_task)

        # Documentation writer agent and task
        doc_writer = self.create_documentation_writer_agent()
        agents.append(doc_writer)
        write_task = self.create_documentation_writing_task(doc_writer, project)
        tasks.append(write_task)

        # Create crew with sequential process
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        # Execute the crew
        result = crew.kickoff()

        # Try to parse the result as JSON
        try:
            # Convert result to string
            result_str = str(result)

            # Strip markdown code blocks if present
            # Handles: ```json\n{...}\n``` or ```\n{...}\n```
            if result_str.strip().startswith('```'):
                # Find the first newline after ```
                start = result_str.find('\n')
                # Find the last ```
                end = result_str.rfind('```')
                if start != -1 and end != -1:
                    result_str = result_str[start+1:end].strip()

            # Parse as JSON
            doc_json = json.loads(result_str)
            return doc_json
        except json.JSONDecodeError as e:
            # If parsing fails, wrap the result in a JSON structure
            print(f"Warning: Could not parse result as JSON: {e}")
            return {
                "project": project,
                "documentation": str(result),
                "format": "text",
                "note": "Documentation could not be parsed as JSON, returning as text"
            }

    def save_documentation(self, documentation: dict, output_file: str = "documentation.json"):
        """Save documentation to a JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documentation, f, indent=2, ensure_ascii=False)
        print(f"\nDocumentation saved to {output_file}")


def main():
    """Main function to run the documentation generation"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python documentation_agent.py <namespace/project> [--with-drive]")
        print("Example: python documentation_agent.py gopay-ds/Growth/my-project")
        print("Example with Google Drive: python documentation_agent.py gopay-ds/Growth/my-project --with-drive")
        sys.exit(1)

    project = sys.argv[1]
    enable_drive = "--with-drive" in sys.argv

    print(f"Generating documentation for GitLab project: {project}")
    if enable_drive:
        print("Google Drive integration: ENABLED")
    print("=" * 60)

    # Create documentation crew
    doc_crew = DocumentationCrew(enable_google_drive=enable_drive)

    # Generate documentation
    documentation = doc_crew.generate_documentation(project)

    # Save to file
    output_file = f"documentation_{project.replace('/', '_')}.json"
    doc_crew.save_documentation(documentation, output_file)

    print("\n" + "=" * 60)
    print("Documentation generation complete!")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
