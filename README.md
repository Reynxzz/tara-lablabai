# TARA - Your Onboarding Buddy

AI-powered Learning Path Generator that helps developers onboard to new projects by creating guided learning paths from GitHub repositories.

## Features

- **Learning Path Generation**: Creates curated learning paths with clickable links to resources
- **Code Q&A**: Ask specific questions about the codebase and get answers with code examples
- **Google Drive Integration**: Optionally search for reference documentation

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
export OPENAI_API_KEY=sk-xxxxxxxxxxxx
```

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501, login with your GitHub token, select a repository, and generate a learning path.

## CLI Usage

```bash
python scripts/run_documentation_agent.py owner/repo
```

With Google Drive:
```bash
python scripts/run_documentation_agent.py owner/repo --with-drive
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | Yes | GitHub Personal Access Token |
| `OPENAI_API_KEY` | Yes | OpenAI API Key |
| `GOOGLE_DRIVE_TOKEN` | No | For Drive integration |

## How It Works

TARA uses a multi-agent system powered by GPT-4o:

1. **GitHub Data Fetcher** - Fetches repository data using the GitHub API
2. **Learning Path Writer** - Formats the data into a readable learning path

## License

MIT
