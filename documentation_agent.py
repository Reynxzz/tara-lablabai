"""Documentation Generation Agent using CrewAI"""
import json
from crewai import Agent, Task, Crew, Process
from github_mcp_tool import GitHubMCPTool
from goto_custom_llm import GoToCustomLLM


class DocumentationCrew:
    """
    CrewAI setup for generating documentation from GitHub repositories using two collaborating agents.

    Architecture:
    - Agent 1 (GitHub Data Analyzer): Uses gpt-oss LLM with GitHub tool to fetch repository data
    - Agent 2 (Documentation Writer): Uses sahabat-4bit LLM to write structured JSON documentation

    This design leverages gpt-oss for tool calling capabilities and sahabat-4bit for
    efficient documentation generation.
    """

    def __init__(self):
        self.github_tool = GitHubMCPTool()

        # Create two LLM instances for collaboration
        print("Initializing dual-LLM architecture:")
        print("  - gpt-oss (GPT OSS 120B): Tool calling & data fetching")
        print("  - sahabat-4bit (Sahabat AI 70B 4-bit): Documentation writing")

        # gpt-oss for tool calling (GitHub data fetching)
        self.gpt_oss_llm = GoToCustomLLM(
            model="openai/gpt-oss-120b",
            endpoint="https://litellm-staging.gopay.sh",
            temperature=1.0
        )

        # sahabat-4bit for documentation writing
        self.sahabat_llm = GoToCustomLLM(
            model="GoToCompany/Llama-Sahabat-AI-v2-70B-IT-awq-4bit",
            endpoint="https://litellm-staging.gopay.sh",
            temperature=0.6
        )

    def create_github_analyzer_agent(self) -> Agent:
        """Create an agent specialized in fetching GitHub data using tools (uses gpt-oss)"""
        return Agent(
            role='GitHub Data Analyzer',
            goal='Fetch and analyze GitHub repository data using the GitHub tool, providing comprehensive raw data for documentation',
            backstory=(
                'You are a meticulous data analyst who specializes in extracting information from GitHub repositories. '
                'Your job is to use the GitHub Repository Analyzer tool to fetch all relevant data about a repository '
                'including its structure, files, commits, stars, forks, and other metadata. You provide thorough, '
                'detailed information that will be used by the documentation team.'
            ),
            tools=[self.github_tool],
            llm=self.gpt_oss_llm,  # Use gpt-oss for tool calling
            verbose=True,
            allow_delegation=False
        )

    def create_documentation_writer_agent(self) -> Agent:
        """Create an agent specialized in writing documentation (uses sahabat-4bit)"""
        return Agent(
            role='Technical Documentation Writer',
            goal='Generate comprehensive, well-structured documentation in JSON format based on GitHub repository data',
            backstory=(
                'You are an expert technical writer with deep knowledge of software architecture '
                'and documentation best practices. You take raw data about repositories and transform it into '
                'clear, comprehensive documentation that helps developers understand projects quickly. '
                'You always output documentation in valid JSON format with proper structure.'
            ),
            tools=[],  # No tools - only writes documentation based on previous agent's output
            llm=self.sahabat_llm,  # Use sahabat-4bit for documentation writing
            verbose=True,
            allow_delegation=False
        )

    def create_github_fetch_task(self, agent: Agent, repository: str) -> Task:
        """Create a task for fetching GitHub repository data"""
        return Task(
            description=(
                f'TASK: Fetch comprehensive data from the GitHub repository "{repository}".\n\n'
                f'STEPS:\n'
                f'1. Use the GitHub Repository Analyzer tool to fetch all repository data\n'
                f'2. Extract and organize the following information:\n'
                f'   - Repository name, description, primary language, and license\n'
                f'   - Stars, forks, open issues count\n'
                f'   - Topics/tags\n'
                f'   - Main files and their purposes\n'
                f'   - Recent commit activity\n'
                f'   - README content (if available)\n'
                f'   - Last updated date\n\n'
                f'IMPORTANT: Provide ALL the raw data you fetch. Be thorough and comprehensive. '
                f'Your output will be used by the documentation writer to create the final documentation.'
            ),
            expected_output=(
                'A comprehensive report containing all fetched GitHub repository data including:\n'
                '- Repository metadata (name, description, language, license, URL)\n'
                '- Community metrics (stars, forks, issues)\n'
                '- File structure and key files\n'
                '- Recent activity and commits\n'
                '- Any additional relevant information from the repository'
            ),
            agent=agent
        )

    def create_documentation_writing_task(self, agent: Agent, repository: str) -> Task:
        """Create a task for writing documentation based on fetched data"""
        return Task(
            description=(
                f'TASK: Generate comprehensive documentation for the GitHub repository "{repository}" '
                f'based on the data provided by the GitHub Data Analyzer.\n\n'
                f'STEPS:\n'
                f'1. Review the repository data provided by the previous agent\n'
                f'2. Create a comprehensive JSON document with these sections:\n'
                f'   - overview: {{name, description, purpose, language, license}}\n'
                f'   - features: [list of key features]\n'
                f'   - tech_stack: {{language, topics, dependencies}}\n'
                f'   - structure: {{main_files: [files with descriptions]}}\n'
                f'   - activity: {{stars, forks, open_issues, last_updated}}\n'
                f'   - getting_started: {{installation, usage, repository_url}}\n\n'
                f'IMPORTANT: Return ONLY the JSON object, no markdown code blocks, no extra text. '
                f'Start your response directly with {{.'
            ),
            expected_output=(
                'A valid, complete JSON object (not wrapped in markdown code blocks) following this structure:\n'
                '{{\n'
                '  "overview": {{"name": "...", "description": "...", "purpose": "...", "language": "...", "license": "..."}},\n'
                '  "features": ["feature1", "feature2"],\n'
                '  "tech_stack": {{"language": "...", "topics": [], "dependencies": "..."}},\n'
                '  "structure": {{"main_files": [{{"name": "...", "purpose": "..."}}]}},\n'
                '  "activity": {{"stars": 0, "forks": 0, "open_issues": 0, "last_updated": "..."}},\n'
                '  "getting_started": {{"installation": "...", "usage": "...", "repository_url": "..."}}\n'
                '}}'
            ),
            agent=agent
        )

    def generate_documentation(self, repository: str) -> dict:
        """
        Generate documentation for a given GitHub repository using two collaborating agents.

        Agent 1 (gpt-oss): Fetches GitHub data using tools
        Agent 2 (sahabat-4bit): Writes documentation based on fetched data

        Args:
            repository: GitHub repository in format 'owner/repo'

        Returns:
            Dictionary containing the generated documentation
        """
        print(f"\n{'='*60}")
        print(f"Starting two-agent collaboration:")
        print(f"  Agent 1 (gpt-oss): Fetching GitHub data...")
        print(f"  Agent 2 (sahabat-4bit): Writing documentation...")
        print(f"{'='*60}\n")

        # Create agents
        github_analyzer = self.create_github_analyzer_agent()
        doc_writer = self.create_documentation_writer_agent()

        # Create tasks
        fetch_task = self.create_github_fetch_task(github_analyzer, repository)
        write_task = self.create_documentation_writing_task(doc_writer, repository)

        # Create crew with sequential process (fetch data first, then write docs)
        crew = Crew(
            agents=[github_analyzer, doc_writer],
            tasks=[fetch_task, write_task],
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
                "repository": repository,
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
        print("Usage: python documentation_agent.py <owner/repo>")
        print("Example: python documentation_agent.py openai/gpt-3")
        sys.exit(1)

    repository = sys.argv[1]

    print(f"Generating documentation for repository: {repository}")
    print("=" * 60)

    # Create documentation crew
    doc_crew = DocumentationCrew()

    # Generate documentation
    documentation = doc_crew.generate_documentation(repository)

    # Save to file
    output_file = f"documentation_{repository.replace('/', '_')}.json"
    doc_crew.save_documentation(documentation, output_file)

    print("\n" + "=" * 60)
    print("Documentation generation complete!")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
