#!/usr/bin/env python3
"""Main entry point for running the documentation generation agent"""
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from src.core import DocumentationCrew
from src.utils.logger import setup_logger
from src.utils.validators import validate_gitlab_project

logger = setup_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive documentation for GitLab projects using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate documentation for a GitLab project
  python scripts/run_documentation_agent.py gopay-ds/Growth/my-project

  # Include Google Drive search for reference documentation
  python scripts/run_documentation_agent.py gopay-ds/Growth/my-project --with-drive

  # Include internal knowledge base search
  python scripts/run_documentation_agent.py gopay-ds/Growth/my-project --with-rag

  # Enable all integrations
  python scripts/run_documentation_agent.py gopay-ds/Growth/my-project --with-drive --with-rag

  # Specify custom output file
  python scripts/run_documentation_agent.py gopay-ds/Growth/my-project --output my-docs.json
        """
    )

    parser.add_argument(
        'project',
        type=str,
        help='GitLab project in format "namespace/project" (e.g., "gopay-ds/Growth/my-project")'
    )

    parser.add_argument(
        '--with-drive',
        action='store_true',
        help='Enable Google Drive search for reference documentation (requires GOOGLE_DRIVE_TOKEN)'
    )

    parser.add_argument(
        '--with-rag',
        action='store_true',
        help='Enable internal knowledge base search using RAG (requires Milvus database)'
    )

    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output file path (default: auto-generated from project name)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main function to run the documentation generation."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    # Validate project format
    if not validate_gitlab_project(args.project):
        logger.error(f"Invalid project format: {args.project}")
        logger.error("Expected format: namespace/project (e.g., 'gopay-ds/Growth/my-project')")
        sys.exit(1)

    # Print configuration
    print("=" * 80)
    print(f"Documentation Generation for GitLab Project")
    print("=" * 80)
    print(f"Project: {args.project}")
    print(f"Google Drive integration: {'ENABLED' if args.with_drive else 'DISABLED'}")
    print(f"RAG integration: {'ENABLED' if args.with_rag else 'DISABLED'}")
    print("=" * 80)
    print()

    try:
        # Create documentation crew
        logger.info("Initializing documentation crew...")
        doc_crew = DocumentationCrew(
            enable_google_drive=args.with_drive,
            enable_rag=args.with_rag
        )

        # Generate documentation
        logger.info(f"Generating documentation for project: {args.project}")
        documentation = doc_crew.generate_documentation(args.project)

        # Save to file
        output_file = doc_crew.save_documentation(documentation, args.output)

        # Print success message
        print()
        print("=" * 80)
        print("✅ Documentation generation complete!")
        print("=" * 80)
        print(f"Output saved to: {output_file}")
        print()

        # Print brief summary
        if isinstance(documentation, dict) and "overview" in documentation:
            print("📄 Documentation Summary:")
            overview = documentation.get("overview", {})
            print(f"  Name: {overview.get('name', 'N/A')}")
            print(f"  Description: {overview.get('description', 'N/A')[:100]}...")
            activity = documentation.get("activity", {})
            print(f"  Activity: {activity.get('stars', 0)} stars, {activity.get('forks', 0)} forks")

        return 0

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1

    except KeyboardInterrupt:
        logger.info("\nDocumentation generation interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Error generating documentation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
