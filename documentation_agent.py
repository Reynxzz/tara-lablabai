"""Documentation Generation Agent using CrewAI"""
import os
import json
from crewai import Agent, Task, Crew, Process, LLM
from github_mcp_tool import GitHubMCPTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DocumentationCrew:
    """CrewAI setup for generating documentation from GitHub repositories"""

    def __init__(self):
        self.github_tool = GitHubMCPTool()

        # Configure Gemini LLM using LiteLLM format
        self.llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )

    def create_documentation_agent(self) -> Agent:
        """Create an agent specialized in generating documentation"""
        return Agent(
            role='Technical Documentation Specialist',
            goal='Generate comprehensive, well-structured documentation in JSON format based on GitHub repository analysis',
            backstory=(
                'You are an expert technical writer with deep knowledge of software architecture '
                'and documentation best practices. You excel at analyzing codebases and creating '
                'clear, comprehensive documentation that helps developers understand projects quickly. '
                'You always output documentation in valid JSON format with proper structure.'
            ),
            tools=[self.github_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_documentation_task(self, agent: Agent, repository: str) -> Task:
        """Create a task for generating documentation"""
        return Task(
            description=(
                f'Analyze the GitHub repository "{repository}" and generate comprehensive documentation. '
                f'Your documentation must be in valid JSON format and include:\n'
                f'1. Project overview (name, description, purpose)\n'
                f'2. Key features and capabilities\n'
                f'3. Technology stack and dependencies\n'
                f'4. Project structure and main components\n'
                f'5. Recent development activity\n'
                f'6. Getting started information\n'
                f'7. Key files and their purposes\n\n'
                f'Use the GitHub Repository Analyzer tool to gather information about the repository. '
                f'Structure your output as a well-organized JSON document with clear sections.'
            ),
            expected_output=(
                'A valid JSON document containing comprehensive documentation with sections for '
                'overview, features, tech stack, structure, activity, and getting started guide. '
                'The JSON should be properly formatted and include all relevant information extracted '
                'from the repository analysis.'
            ),
            agent=agent
        )

    def generate_documentation(self, repository: str) -> dict:
        """
        Generate documentation for a given GitHub repository

        Args:
            repository: GitHub repository in format 'owner/repo'

        Returns:
            Dictionary containing the generated documentation
        """
        # Create agent and task
        agent = self.create_documentation_agent()
        task = self.create_documentation_task(agent, repository)

        # Create crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        # Execute the crew
        result = crew.kickoff()

        # Try to parse the result as JSON
        try:
            # The result might be a string or already parsed
            if isinstance(result, str):
                doc_json = json.loads(result)
            else:
                doc_json = json.loads(str(result))
            return doc_json
        except json.JSONDecodeError:
            # If parsing fails, wrap the result in a JSON structure
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
