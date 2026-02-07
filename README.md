# NoBuddy - Your Onboarding Buddy

**"Nobody knows? NoBuddy knows!"**

An AI-powered Learning Path Generator that helps developers onboard to new projects by creating guided learning paths from GitHub repositories and Google Drive documents. Built with CrewAI multi-agent collaboration and dual-LLM architecture using OpenAI GPT models.

## Features

### Learning Path Generation
- Creates curated learning paths with valid, clickable links to resources
- Synthesizes information from multiple sources (GitHub, Google Drive)
- Provides overview, code snippets, reference documentation, and getting started guides
- Focuses on **guiding** users to resources rather than explaining everything inline

### Code Q&A
- Interactive code deep-dive agent that answers specific questions about the codebase
- Fetches and analyzes code files from specified directories
- Returns answers with code examples and file references
- Example: "What feature processing does this project do?"

### Multi-Source Intelligence
- **GitHub Integration**: Fetches project metadata, file structure, commits, README, and code snippets
- **Google Drive Integration**: Searches for reference documentation with clickable links

### Professional Web Interface
- Streamlit-based UI with clean green theme
- User authentication with GitHub and Drive tokens
- Repository selection from user's accessible repositories
- Searchable repository dropdown

### Dual-LLM Architecture
- **GPT-4o-mini**: For tool calling and data fetching (lower cost)
- **GPT-4o**: For learning path writing (better quality)

## Prerequisites

- Python 3.10 or higher
- GitHub Personal Access Token (with `repo` scope for private repos, or `public_repo` for public only)
- OpenAI API Key
- Google Drive Token (optional, for Drive integration)

## Installation

1. Clone or navigate to this directory:
```bash
cd nobuddy
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```

Then edit `.env` and add your credentials:
```
# Required
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx

# Optional (for GitHub Enterprise)
GITHUB_API_URL=https://api.github.com

# Optional (for Google Drive integration)
GOOGLE_DRIVE_TOKEN=your_google_drive_token
MCP_DRIVE_URL=drive.taraai.tech
```

### Getting a GitHub Personal Access Token

1. Go to GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name and select scopes: `repo` (for private repos) or `public_repo` (for public only)
4. Click "Generate token"
5. Copy the token

### Getting an OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key

## Usage

### Web Interface (Recommended)

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. **Login**:
   - Enter your GitHub Personal Access Token
   - (Optional) Enter your Google Drive Token
   - Click "Login and Load Repositories"

4. **Generate Learning Path**:
   - Select a repository from the dropdown
   - Enable optional integrations (Drive)
   - Click "Generate Learning Path"
   - View and download the generated learning path

5. **Ask Code Questions** (after selecting a repository):
   - Enter your question (e.g., "What feature processing does this project do?")
   - Specify directory to search (default: `src`)
   - Click "Ask Question"
   - View the answer with code examples and file links

### Command Line Interface

Generate a learning path for a GitHub repository:

```bash
python scripts/run_documentation_agent.py owner/repo
```

With Google Drive integration:
```bash
python scripts/run_documentation_agent.py owner/repo --with-drive
```

Example:
```bash
python scripts/run_documentation_agent.py facebook/react
```

This will:
1. Fetch repository data from GitHub (metadata, files, commits, code snippets)
2. Search Google Drive for reference documentation (if enabled)
3. Generate a comprehensive learning path in Markdown
4. Save the output to `learning_path_facebook_react.md`

### Programmatic Usage

```python
from src.core.crew import DocumentationCrew

# Create the crew with optional integrations
doc_crew = DocumentationCrew(
    enable_google_drive=True  # Optional
)

# Generate learning path
result = doc_crew.generate_documentation("owner/repo")

# Save to file
output_path = doc_crew.save_documentation(result)
print(f"Learning path saved to: {output_path}")

# Answer a code question
qa_result = doc_crew.answer_code_question(
    repo="owner/repo",
    question="What feature processing does this project do?",
    directory="src"
)
print(qa_result["answer"])
```

## Output Format

The generated learning path is in **Markdown format** and includes:

```markdown
# Learning Path: Project Name

## Overview
- What this project does (synthesized from GitHub and Drive)
- Key purpose and use cases
- Technologies used
- Project metadata: [Project URL](link), License, Default Branch

## Recent Contributors
- List of recent commit authors with their latest contributions
- Include commit titles and dates

## Repository Structure
List key directories and files with clickable links:
- [main.py](https://github.com/owner/repo/blob/main/src/main.py) - Application entry point
- [config.py](https://github.com/owner/repo/blob/main/src/config.py) - Configuration management

## Code Snippets (First Look)
```python
# From src/main.py
def main():
    # Application logic here
    pass
```
[View full file](https://github.com/owner/repo/blob/main/src/main.py)

## Reference Documentation

**From Google Drive:**
- [Setup Guide](https://docs.google.com/document/d/xxx) - Installation and configuration
- [Architecture Overview](https://docs.google.com/document/d/yyy) - System design

## Getting Started
- Start by reading [README](link)
- Check configuration in [config file](link)
- Review setup instructions in [Drive doc](link)
- Installation steps with links to detailed docs
```

## Project Structure

```
nobuddy/
├── app.py                          # Streamlit web interface
├── scripts/
│   └── run_documentation_agent.py  # CLI script for learning path generation
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── factory.py              # Agent factory functions
│   ├── config/
│   │   ├── __init__.py
│   │   ├── constants.py            # Constants and enums
│   │   └── settings.py             # Configuration management
│   ├── core/
│   │   ├── __init__.py
│   │   └── crew.py                 # DocumentationCrew orchestration
│   ├── llm/
│   │   ├── __init__.py
│   │   └── custom_llm.py           # OpenAI LLM integration
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── github_tool.py          # GitHub Project Analyzer
│   │   ├── github_code_qa_tool.py  # GitHub Code Q&A
│   │   └── google_drive_tool.py    # Google Drive integration
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # Logging utilities
│       └── validators.py           # Input validation
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
└── README.md                       # This file
```

## How It Works

### Multi-Agent Architecture

NoBuddy uses a **multi-agent collaboration system** with CrewAI:

#### Learning Path Generation Flow

1. **GitHub Data Analyzer Agent** (uses GPT-4o-mini):
   - Calls GitHub Project Analyzer tool
   - Fetches: repository metadata, file structure, commits, README, code snippets
   - Extracts contributor information
   - Returns comprehensive repository data with clickable links

2. **Google Drive Analyzer Agent** (uses GPT-4o-mini) - *Optional*:
   - Calls Google Drive Document Analyzer tool
   - Searches for reference documentation related to the project
   - Extracts key definitions and important points from documents
   - Converts Drive URIs to clickable Google Docs/Sheets links

3. **Learning Path Writer Agent** (uses GPT-4o):
   - Synthesizes information from all previous agents
   - Creates a curated learning path in Markdown format
   - Focuses on guiding users to resources (not explaining everything)
   - Includes valid, clickable links to all resources
   - Outputs structured learning path with code snippets and references

#### Code Q&A Flow

1. **Code Q&A Agent** (uses GPT-4o-mini):
   - Calls GitHub Code Q&A tool
   - Fetches up to 10 Python files from specified directory (e.g., `src/`)
   - Analyzes code to answer the specific question
   - Returns answer with code examples and file links

### Tools

1. **GitHub Project Analyzer** (`src/tools/github_tool.py`):
   - Fetches repository metadata, file structure, commits, README
   - Extracts code snippets from key files (main.py, app.py, etc.)
   - Generates clickable links to files on GitHub

2. **GitHub Code Q&A** (`src/tools/github_code_qa_tool.py`):
   - Deep-dives into repository code for Q&A
   - Recursively fetches Python files from directories
   - Returns full code content (up to 1000 lines per file)

3. **Google Drive Document Analyzer** (`src/tools/google_drive_tool.py`):
   - Searches Google Drive via MCP server
   - Converts Drive URIs to clickable URLs (https://docs.google.com/...)
   - Returns document content with metadata

### Execution Process

**Sequential workflow:**
```
GitHub Agent -> Drive Agent (optional) -> Learning Path Writer
```

Each agent:
- Has strict instructions to ONLY use its designated tool
- Cannot fabricate or assume information
- Must include proof (links, source fields) in responses
- Passes data to the next agent in the pipeline

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | Yes | GitHub Personal Access Token |
| `OPENAI_API_KEY` | Yes | OpenAI API Key |
| `GITHUB_API_URL` | No | GitHub API URL (default: https://api.github.com) |
| `GOOGLE_DRIVE_TOKEN` | No | Google Drive token for Drive integration |
| `MCP_DRIVE_URL` | No | MCP server URL for Drive (default: drive.taraai.tech) |

## Troubleshooting

### GitHub API Authentication Errors

If you encounter `401 Unauthorized` errors:
- Ensure your `GITHUB_TOKEN` has the correct scopes: `repo` or `public_repo`
- Verify the token hasn't expired
- Check that you have access to the repository you're trying to analyze
- Test your token with: `curl -H "Authorization: Bearer your_token" https://api.github.com/user`

### OpenAI API Errors

If you encounter OpenAI API errors:
- Ensure your `OPENAI_API_KEY` is valid
- Check your API usage limits and billing
- Verify the key has access to GPT-4o and GPT-4o-mini models

### Agent Tool Confusion

If agents are calling the wrong tools:
- Check logs for which tool was actually called
- Ensure tool descriptions are distinct (GitHub Project Analyzer vs GitHub Code Q&A)
- Verify agent backstories have clear instructions about which tool to use

### Google Drive Integration Issues

If Drive search fails:
- Ensure the MCP server is running at `MCP_DRIVE_URL`
- Verify your Drive token is valid
- Check network connectivity to the MCP server
- Review MCP server logs for errors

### Code Q&A Returns "No Files Found"

If Code Q&A can't find files:
- Verify the directory exists in the repository (default: `src/`)
- Check if there are Python files (`.py`) in that directory
- Try a different directory (e.g., `lib/`, `app/`)
- Ensure the GitHub token has repository read access

## License

MIT License - feel free to use and modify as needed.

## Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [GitHub REST API Documentation](https://docs.github.com/en/rest)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
