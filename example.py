"""Example script for generating documentation"""
from documentation_agent import DocumentationCrew
import json


def generate_docs_for_repo(repository: str):
    """Generate documentation for a specific repository"""
    print(f"\n{'='*60}")
    print(f"Generating documentation for: {repository}")
    print(f"{'='*60}\n")

    # Create documentation crew
    doc_crew = DocumentationCrew()

    # Generate documentation
    documentation = doc_crew.generate_documentation(repository)

    # Display generated documentation
    print("\n" + "="*60)
    print("GENERATED DOCUMENTATION:")
    print("="*60)
    print(json.dumps(documentation, indent=2))
    print("="*60)

    # Save to file
    output_file = f"docs_{repository.replace('/', '_')}.json"
    doc_crew.save_documentation(documentation, output_file)

    return documentation


if __name__ == "__main__":
    # Example: Generate documentation for a popular repository
    # You can change this to any public GitHub repository

    example_repos = [
        "crewai/crewai",  # CrewAI itself
        # Add more repositories as needed
    ]

    for repo in example_repos:
        try:
            generate_docs_for_repo(repo)
        except Exception as e:
            print(f"Error generating docs for {repo}: {e}")
