# GitHub Documentation Generator with CrewAI & MCP

An AI-powered documentation generator that analyzes GitHub repositories and creates comprehensive documentation in JSON format using CrewAI agents and the Model Context Protocol (MCP).

## Features

- Analyzes GitHub repositories using MCP for standardized data access
- Generates structured documentation in JSON format
- Extracts repository metadata, file structure, and recent activity
- Uses CrewAI for intelligent documentation generation
- Customizable output format and content

## Prerequisites

- Python 3.10 or higher
- Node.js (for MCP GitHub server)
- GitHub Personal Access Token
- Google AI Studio API Key (for Gemini)

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
GITHUB_TOKEN=your_github_personal_access_token
GOOGLE_API_KEY=your_google_api_key
```

### Getting a GitHub Token

1. Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name and select scopes: `repo`, `read:user`
4. Copy the token and add it to your `.env` file

### Getting a Google AI Studio API Key

1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key" or "Get API Key"
3. Select a Google Cloud project or create a new one
4. Copy the API key and add it to your `.env` file

## Usage

### Basic Usage

Generate documentation for any public GitHub repository:

```bash
python documentation_agent.py owner/repo
```

Example:
```bash
python documentation_agent.py facebook/react
```

This will:
1. Analyze the repository using GitHub API
2. Generate comprehensive documentation using CrewAI
3. Save the output to `documentation_owner_repo.json`

### Using the Example Script

Run the example script to see a demonstration:

```bash
python example.py
```

### Programmatic Usage

You can also use the documentation generator in your own Python code:

```python
from documentation_agent import DocumentationCrew

# Create the documentation crew
doc_crew = DocumentationCrew()

# Generate documentation for a repository
documentation = doc_crew.generate_documentation("owner/repo")

# Save to file
doc_crew.save_documentation(documentation, "output.json")
```

## Output Format

The generated documentation is in JSON format and includes:

```json
{
  "repository": "owner/repo",
  "overview": {
    "name": "Repository Name",
    "description": "Description",
    "purpose": "Main purpose"
  },
  "features": ["Feature 1", "Feature 2"],
  "tech_stack": {
    "language": "Primary language",
    "dependencies": []
  },
  "structure": {
    "main_components": []
  },
  "activity": {
    "recent_commits": []
  },
  "getting_started": {
    "installation": "",
    "usage": ""
  }
}
```

## Project Structure

```
goto-hacks-2025/
├── documentation_agent.py   # Main CrewAI agent implementation
├── github_mcp_tool.py       # GitHub MCP tool wrapper
├── example.py               # Example usage script
├── mcp_config.json          # MCP server configuration
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## How It Works

1. **GitHub MCP Tool** (`github_mcp_tool.py`):
   - Wraps GitHub API access using MCP principles
   - Fetches repository info, file structure, commits, and README
   - Provides structured data to the CrewAI agent

2. **Documentation Agent** (`documentation_agent.py`):
   - CrewAI agent specialized in technical documentation powered by Google Gemini 1.5 Pro
   - Analyzes repository data from the GitHub tool
   - Generates comprehensive, structured JSON documentation

3. **Crew Execution**:
   - Agent receives the task to document a repository
   - Uses the GitHub tool to gather repository information
   - Processes and structures the information into JSON
   - Returns formatted documentation

## Customization

### Modify the LLM Model

Edit `documentation_agent.py` to use a different Gemini model:

```python
self.llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro", "gemini-pro", etc.
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,  # Adjust temperature for creativity (0.0-1.0)
    convert_system_message_to_human=True
)
```

Available models:
- `gemini-1.5-pro` - Most capable model (default)
- `gemini-1.5-flash` - Faster and more cost-effective
- `gemini-pro` - Previous generation model

### Modify the Agent

Edit `documentation_agent.py` to customize the agent's behavior:

```python
def create_documentation_agent(self) -> Agent:
    return Agent(
        role='Your Custom Role',
        goal='Your custom goal',
        backstory='Your custom backstory',
        # ... customize as needed
    )
```

### Modify the Output Format

Edit the task description in `create_documentation_task()` to change what information is included in the documentation.

### Add More Tools

Create additional tools in the style of `GitHubMCPTool` and add them to the agent's `tools` list.

## Troubleshooting

### GitHub API Rate Limiting

If you encounter rate limiting:
- Ensure your `GITHUB_TOKEN` is set in `.env`
- Wait for the rate limit to reset (usually 1 hour)
- Use a GitHub token with higher rate limits

### Google Gemini API Errors

If you encounter Gemini API errors:
- Check that `GOOGLE_API_KEY` is correctly set in `.env`
- Ensure your API key is valid and active in Google AI Studio
- Check that you haven't exceeded your API quota
- Visit https://aistudio.google.com/ to verify your API key status

### MCP Configuration Issues

The MCP configuration is primarily for reference. The current implementation uses direct GitHub API access through the `requests` library for simplicity and reliability.

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [GitHub API Documentation](https://docs.github.com/en/rest)
