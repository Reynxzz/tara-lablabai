# NoBuddy - Your Onboarding Buddy

**"Nobody knows? NoBuddy knows!"**

An AI-powered Learning Path Generator that helps developers onboard to new projects by creating guided learning paths from GitLab repositories, Google Drive documents, and internal knowledge bases. Built with CrewAI multi-agent collaboration and dual-LLM architecture.

## Features

### Learning Path Generation
- Creates curated learning paths with valid, clickable links to resources
- Synthesizes information from multiple sources (GitLab, Google Drive, Knowledge Base)
- Provides overview, code snippets, reference documentation, and getting started guides
- Focuses on **guiding** users to resources rather than explaining everything inline

### Code Q&A
- Interactive code deep-dive agent that answers specific questions about the codebase
- Fetches and analyzes code files from specified directories
- Returns answers with code examples and file references
- Example: "What feature processing does this project do?"

### Multi-Source Intelligence
- **GitLab Integration**: Fetches project metadata, file structure, commits, README, and code snippets
- **Google Drive Integration**: Searches for reference documentation with clickable links
- **Internal Knowledge Base**: Smart keyword extraction from repository names (RAG with Milvus)

### Professional Web Interface
- Streamlit-based UI with clean green theme
- User authentication with GitLab and Drive tokens
- Project selection from user's accessible repositories
- Searchable project dropdown

### Dual-LLM Architecture
- **GPT OSS 120B**: For tool calling and data fetching
- **Sahabat AI 70B 4-bit**: For learning path writing

## Prerequisites

- Python 3.10 or higher
- GitLab Personal Access Token (with `read_api` and `read_repository` scopes)
- Google Drive Token (optional, for Drive integration)
- Access to GoTo LiteLLM endpoint (gpt-oss and sahabat-4bit models)
- Milvus database with knowledge base (optional, for RAG integration)

## Installation

1. Clone or navigate to this directory:
```bash
cd /Users/luthfi.reynaldi/Documents/goto-hacks-2025
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
GITLAB_TOKEN=your_gitlab_personal_access_token
GITLAB_URL=https://source.golabs.io
LLM_ENDPOINT=https://litellm-staging.gopay.sh

# Optional (for Google Drive integration)
GOOGLE_DRIVE_TOKEN=your_google_drive_token
GOOGLE_DRIVE_MCP_URL=http://localhost:3000

# Optional (for RAG integration)
RAG_DB_PATH=./milvus_demo_batch_bmth_v3_3.db
RAG_EMBEDDING_MODEL=GoToCompany/embeddinggemma-300m-gotoai-v1
RAG_EMBEDDING_ENDPOINT=https://litellm-staging.gopay.sh/embeddings
```

### Getting a GitLab Personal Access Token

1. Go to your GitLab instance: `https://source.golabs.io/-/user_settings/personal_access_tokens`
2. Click "Add new token"
3. Give it a name and select scopes: `read_api`, `read_repository`
4. Set expiration date as needed
5. Click "Create personal access token"
6. Copy the token

## Usage

### Web Interface (Recommended)

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. **Login**:
   - Enter your GitLab Personal Access Token
   - (Optional) Enter your Google Drive Token
   - Click "Login and Load Projects"

4. **Generate Learning Path**:
   - Select a project from the dropdown
   - Enable optional integrations (Drive, Knowledge Base)
   - Click "Generate Learning Path"
   - View and download the generated learning path

5. **Ask Code Questions** (after selecting a project):
   - Enter your question (e.g., "What feature processing does this project do?")
   - Specify directory to search (default: `src`)
   - Click "Ask Question"
   - View the answer with code examples and file links

### Command Line Interface

Generate a learning path for a GitLab project:

```bash
python scripts/run_documentation_agent.py namespace/project
```

With Google Drive integration:
```bash
python scripts/run_documentation_agent.py namespace/project --with-drive
```

With all integrations:
```bash
python scripts/run_documentation_agent.py namespace/project --with-drive --with-rag
```

Example:
```bash
python scripts/run_documentation_agent.py gopay-ds/Growth/gopay-dge-ride-model_pipeline-staging --with-drive
```

This will:
1. Fetch project data from GitLab (metadata, files, commits, code snippets)
2. Search Google Drive for reference documentation (if enabled)
3. Search internal knowledge base for relevant context (if enabled)
4. Generate a comprehensive learning path in Markdown
5. Save the output to `learning_path_gopay-ds_Growth_gopay-dge-ride-model_pipeline-staging.md`

### Programmatic Usage

```python
from src.core.crew import DocumentationCrew

# Create the crew with optional integrations
doc_crew = DocumentationCrew(
    enable_google_drive=True,  # Optional
    enable_rag=True            # Optional
)

# Generate learning path
result = doc_crew.generate_documentation("namespace/project")

# Save to file
output_path = doc_crew.save_documentation(result)
print(f"Learning path saved to: {output_path}")

# Answer a code question
qa_result = doc_crew.answer_code_question(
    project="namespace/project",
    question="What feature processing does this project do?",
    directory="src"
)
print(qa_result["answer"])
```

## Output Format

The generated learning path is in **Markdown format** and includes:

```markdown
# 🎯 Learning Path: Project Name

## 📋 Overview
- What this project does (synthesized from GitLab, Drive, and Knowledge Base)
- Key purpose and use cases
- Technologies used
- Project metadata: [Project URL](link), License, Default Branch

## 👥 Recent Contributors
- List of recent commit authors with their latest contributions
- Include commit titles and dates

## 📁 Repository Structure
List key directories and files with clickable links:
- [main.py](https://source.golabs.io/project/-/blob/main/src/main.py) - Application entry point
- [config.py](https://source.golabs.io/project/-/blob/main/src/config.py) - Configuration management

## 💻 Code Snippets (First Look)
```python
# From src/main.py
def main():
    # Application logic here
    pass
```
[View full file](https://source.golabs.io/project/-/blob/main/src/main.py)

## 📚 Reference Documentation

**From Google Drive:**
- [DGE Serving Checklist](https://docs.google.com/document/d/xxx) - Deployment guidelines and best practices
- [Feature Engineering Guide](https://docs.google.com/document/d/yyy) - How to process and transform features

**From Internal Knowledge Base:**
- Relevant context from internal systems (e.g., dge, genie, user_income)
- How this project relates to company infrastructure

## 🚀 Getting Started
- Start by reading [README](link)
- Check configuration in [config file](link)
- Review setup instructions in [Drive doc](link)
- Installation steps with links to detailed docs
```

### Code Q&A Output

When asking code questions, the output includes:

```markdown
**Question:** What feature processing does this project do?

**Answer:**
Based on the code analysis, this project performs the following feature processing:

1. **Data Cleaning** (src/processing/clean.py:45)
   - Removes null values and duplicates
   - Handles missing data with mean imputation

[View code](https://source.golabs.io/project/-/blob/main/src/processing/clean.py)

2. **Feature Engineering** (src/features/engineering.py:120)
   - Creates interaction features
   - Applies log transformations

[View code](https://source.golabs.io/project/-/blob/main/src/features/engineering.py)

**Summary:** The project uses a two-stage pipeline for feature processing...
```

## Project Structure

```
goto-hacks-2025/
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
│   │   └── goto_custom_llm.py      # Custom LLM integration
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── gitlab_tool.py          # GitLab Project Analyzer
│   │   ├── gitlab_code_qa_tool.py  # GitLab Code Q&A
│   │   ├── google_drive_tool.py    # Google Drive integration
│   │   └── rag_tool.py             # Internal Knowledge Base Search
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # Logging utilities
│       └── validators.py           # Input validation
├── tests/
│   └── test_rag_tool.py            # RAG tool tests
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

1. **GitLab Data Analyzer Agent** (uses GPT OSS 120B):
   - Calls GitLab Project Analyzer tool
   - Fetches: project metadata, file structure, commits, README, code snippets
   - Extracts contributor information
   - Returns comprehensive project data with clickable links

2. **Google Drive Analyzer Agent** (uses GPT OSS 120B) - *Optional*:
   - Calls Google Drive Document Analyzer tool
   - Searches for reference documentation related to the project
   - Extracts key definitions and important points from documents
   - Converts Drive URIs to clickable Google Docs/Sheets links

3. **Internal Knowledge Base Analyzer Agent** (uses GPT OSS 120B) - *Optional*:
   - Calls Internal Knowledge Base Search tool
   - Extracts project keywords from repository name (e.g., "genie" from "gopay-genie-model_pipeline")
   - Searches Milvus vector database with smart keyword matching
   - Returns relevant context with source attribution (dge, genie, user_income, etc.)

4. **Learning Path Writer Agent** (uses Sahabat AI 70B 4-bit):
   - Synthesizes information from all previous agents
   - Creates a curated learning path in Markdown format
   - Focuses on guiding users to resources (not explaining everything)
   - Includes valid, clickable links to all resources
   - Outputs structured learning path with code snippets and references

#### Code Q&A Flow

1. **Code Q&A Agent** (uses GPT OSS 120B):
   - Calls GitLab Code Q&A tool
   - Fetches up to 10 Python files from specified directory (e.g., `src/`)
   - Analyzes code to answer the specific question
   - Returns answer with code examples and file links

### Tools

1. **GitLab Project Analyzer** (`src/tools/gitlab_tool.py`):
   - Fetches project metadata, file structure, commits, README
   - Extracts code snippets from key files (main.py, app.py, etc.)
   - Generates clickable links to files in GitLab

2. **GitLab Code Q&A** (`src/tools/gitlab_code_qa_tool.py`):
   - Deep-dives into repository code for Q&A
   - Recursively fetches Python files from directories
   - Returns full code content (up to 1000 lines per file)

3. **Google Drive Document Analyzer** (`src/tools/google_drive_tool.py`):
   - Searches Google Drive via MCP server
   - Converts Drive URIs to clickable URLs (https://docs.google.com/...)
   - Returns document content with metadata

4. **Internal Knowledge Base Search** (`src/tools/rag_tool.py`):
   - Semantic search using Milvus vector database
   - Uses EmbeddingGemma-300M for embeddings
   - Searches `combined_item` collection with source field attribution

### Execution Process

**Sequential workflow:**
```
GitLab Agent → Drive Agent (optional) → RAG Agent (optional) → Learning Path Writer
```

Each agent:
- Has strict instructions to ONLY use its designated tool
- Cannot fabricate or assume information
- Must include proof (links, source fields) in responses
- Passes data to the next agent in the pipeline

## Customization

### Modify the LLM Models

Edit `src/llm/goto_custom_llm.py` or update constants in `src/config/constants.py`:

```python
# In src/config/constants.py
class LLMModel(str, Enum):
    GPT_OSS = "openai/gpt-oss-120b"
    SAHABAT_4BIT = "GoToCompany/Llama-Sahabat-AI-v2-70B-IT-awq-4bit"
```

Adjust temperature in `src/core/crew.py`:

```python
# For tool calling (data fetching)
DEFAULT_TEMPERATURE_TOOL_CALLING = 0.3

# For writing (learning path generation)
DEFAULT_TEMPERATURE_WRITING = 0.6
```

### Modify Agent Behavior

Edit agent factories in `src/agents/factory.py`:

```python
def create_gitlab_analyzer_agent(llm, gitlab_tool):
    return Agent(
        role='Your Custom Role',
        goal='Your custom goal',
        backstory='Your custom backstory with strict rules',
        tools=[gitlab_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
```

### Modify Learning Path Structure

Edit the learning path task in `src/core/crew.py` → `_create_learning_path_writing_task()`:

```python
# Change the required sections
f'## Your Custom Section\n'
f'- Custom content here\n\n'
```

### Add More Data Sources

1. Create a new tool in `src/tools/your_tool.py`:
```python
class YourCustomTool(BaseTool):
    name: str = "Your Tool Name"
    description: str = "What it does"

    def _run(self, query: str) -> str:
        # Your implementation
        return json.dumps(result)
```

2. Create an agent in `src/agents/factory.py`:
```python
def create_your_analyzer_agent(llm, your_tool):
    return Agent(...)
```

3. Add to crew in `src/core/crew.py`:
```python
your_analyzer = create_your_analyzer_agent(self.gpt_oss_llm, your_tool)
agents.append(your_analyzer)
tasks.append(self._create_your_task(your_analyzer, project))
```

### Customize UI Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#4CAF50"      # Green buttons
backgroundColor = "#FFFFFF"    # White background
secondaryBackgroundColor = "#F1F8F4"  # Light green panels
textColor = "#2D5F2E"         # Dark green text
```

## Troubleshooting

### GitLab API Authentication Errors

If you encounter `403 Forbidden` errors:
- Ensure your `GITLAB_TOKEN` has the correct scopes: `read_api` and `read_repository`
- Verify the token hasn't expired
- Check that you have access to the project you're trying to analyze
- Test your token with: `curl -H "PRIVATE-TOKEN: your_token" https://source.golabs.io/api/v4/user`

### Agent Tool Confusion

If agents are calling the wrong tools:
- Check logs for which tool was actually called
- Ensure tool descriptions are distinct (GitLab Project Analyzer vs GitLab Code Q&A)
- Verify agent backstories have clear instructions about which tool to use

### RAG Tool Returns No Results

If the knowledge base search returns nothing:
- Verify the Milvus database path is correct (`RAG_DB_PATH`)
- Check if the project keywords exist in the knowledge base (genie, pills, dge, user_income)
- Try different search queries manually to test the database
- Verify the embedding endpoint is reachable

### Google Drive Integration Issues

If Drive search fails:
- Ensure the MCP server is running at `GOOGLE_DRIVE_MCP_URL`
- Verify your Drive token is valid
- Check network connectivity to the MCP server
- Review MCP server logs for errors

### Code Q&A Returns "No Files Found"

If Code Q&A can't find files:
- Verify the directory exists in the repository (default: `src/`)
- Check if there are Python files (`.py`) in that directory
- Try a different directory (e.g., `common/`, `lib/`)
- Ensure the GitLab token has repository read access

### Streamlit Authentication Loop

If you keep getting logged out:
- Clear browser cache and cookies
- Check if tokens are valid
- Verify session state is preserved
- Try restarting the Streamlit server

## Key Features Explained

### Smart RAG Keyword Extraction

The system intelligently extracts project names from repository paths:
- `gopay-genie-model_pipeline-production` → searches for "genie"
- `gopay-dge-ride-model_pipeline-staging` → searches for "dge" and "ride"
- `gopay-pills-service` → searches for "pills"

Known project keywords: **genie**, **pills**, **push notification (pn)**, **user income**, **dge**, **ride**

### Multi-Source Overview Synthesis

The Learning Path Writer synthesizes information from ALL sources:
- GitLab description + README
- Google Drive document summaries
- Internal knowledge base context

This creates a richer overview than just using the GitLab description.

### Clickable Links Everywhere

All resources include valid, clickable links:
- **GitLab files**: `https://source.golabs.io/project/-/blob/main/file.py`
- **Google Drive**: `https://docs.google.com/document/d/xxx`
- **Code snippets**: Links to full files in GitLab

## Development

### Running Tests

```bash
# Test RAG tool
python -m pytest tests/test_rag_tool.py -v

# Test with actual query
python tests/test_rag_tool.py
```

### Environment Setup

For development, you may want to use a `.env.local` file:
```bash
cp .env .env.local
# Edit .env.local with test credentials
```

### Logging

Logs are written to console with color-coded levels:
- `INFO`: General operation flow
- `DEBUG`: Detailed tool calls and responses
- `WARNING`: Non-critical issues
- `ERROR`: Critical failures

Enable debug logging:
```python
# In src/utils/logger.py
logger.setLevel(logging.DEBUG)
```

## Built For

**GoTo Hackathon 2025** by **Bring Me The Hackathon** team

## License

MIT License - feel free to use and modify as needed.

## Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [GitLab REST API Documentation](https://docs.gitlab.com/ee/api/rest/)
- [GitLab Projects API](https://docs.gitlab.com/ee/api/projects.html)
- [GitLab Repository API](https://docs.gitlab.com/ee/api/repositories.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Milvus Documentation](https://milvus.io/docs)
