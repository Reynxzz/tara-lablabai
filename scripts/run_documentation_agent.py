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
from src.utils.validators import validate_github_repo

logger = setup_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive documentation for GitHub repositories using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate documentation for a GitHub repository
  python scripts/run_documentation_agent.py owner/repo

  # Include Google Drive search for reference documentation
  python scripts/run_documentation_agent.py owner/repo --with-drive

  # Specify custom output file
  python scripts/run_documentation_agent.py owner/repo --output my-docs.md
        """
    )

    parser.add_argument(
        'repo',
        type=str,
        help='GitHub repository in format "owner/repo" (e.g., "facebook/react")'
    )

    parser.add_argument(
        '--with-drive',
        action='store_true',
        help='Enable Google Drive search for reference documentation (requires GOOGLE_DRIVE_TOKEN)'
    )

    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output file path (default: auto-generated from repository name)'
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

    # Validate repository format
    if not validate_github_repo(args.repo):
        logger.error(f"Invalid repository format: {args.repo}")
        logger.error("Expected format: owner/repo (e.g., 'facebook/react')")
        sys.exit(1)

    # Print configuration
    print("=" * 80)
    print(f"Documentation Generation for GitHub Repository")
    print("=" * 80)
    print(f"Repository: {args.repo}")
    print(f"Google Drive integration: {'ENABLED' if args.with_drive else 'DISABLED'}")
    print("=" * 80)
    print()

    try:
        # Create documentation crew
        logger.info("Initializing documentation crew...")
        doc_crew = DocumentationCrew(
            enable_google_drive=args.with_drive
        )

        # Generate documentation
        logger.info(f"Generating documentation for repository: {args.repo}")
        documentation = doc_crew.generate_documentation(args.repo)

        # Save to file
        output_file = doc_crew.save_documentation(documentation, args.output)

        # Print success message
        print()
        print("=" * 80)
        print("Documentation generation complete!")
        print("=" * 80)
        print(f"Output saved to: {output_file}")
        print()

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
