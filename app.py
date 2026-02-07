"""Streamlit web interface for TARA - Learning Path Generator"""
import streamlit as st
import sys
import os
import requests
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.crew import DocumentationCrew, extract_markdown_from_response
from src.utils.logger import setup_logger
from src.utils.validators import validate_github_repo
from src.config.settings import get_settings

logger = setup_logger(__name__)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'github_token' not in st.session_state:
    st.session_state.github_token = ""
if 'drive_token' not in st.session_state:
    st.session_state.drive_token = ""
if 'user_repos' not in st.session_state:
    st.session_state.user_repos = []
if 'github_api_url' not in st.session_state:
    st.session_state.github_api_url = os.getenv("GITHUB_API_URL", "https://api.github.com")


def fetch_user_repos(github_token: str, github_api_url: str) -> List[Dict[str, Any]]:
    """
    Fetch all repositories accessible by the user from GitHub API.

    Args:
        github_token: GitHub personal access token
        github_api_url: GitHub API URL

    Returns:
        List of repository dictionaries with id, name, and full_name
    """
    try:
        headers = {
            'Authorization': f'Bearer {github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'X-GitHub-Api-Version': '2022-11-28'
        }
        repos = []
        page = 1
        per_page = 100

        while True:
            url = f'{github_api_url}/user/repos'
            response = requests.get(
                url,
                headers=headers,
                params={
                    'per_page': per_page,
                    'page': page,
                    'sort': 'pushed',
                    'direction': 'desc',
                    'affiliation': 'owner,collaborator,organization_member'
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Failed to fetch repos: HTTP {response.status_code} - {response.text[:200]}")
                return []

            try:
                page_repos = response.json()
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return []

            if not page_repos:
                break

            if not isinstance(page_repos, list):
                logger.error(f"Unexpected response format: {type(page_repos)}")
                return []

            for repo in page_repos:
                # Skip if repo is None or doesn't have required fields
                if not repo or not isinstance(repo, dict):
                    continue

                full_name = repo.get('full_name')
                if not full_name:
                    continue

                description = repo.get('description', '') or ''
                repos.append({
                    'id': repo.get('id'),
                    'name': repo.get('name'),
                    'full_name': full_name,
                    'description': description[:100] if description else '',
                    'pushed_at': repo.get('pushed_at', ''),
                    'private': repo.get('private', False)
                })

            # Check if there are more pages
            if len(page_repos) < per_page:
                break
            page += 1

        logger.info(f"Fetched {len(repos)} repositories for user")
        return repos

    except Exception as e:
        logger.error(f"Error fetching user repos: {e}")
        return []


def verify_github_token(github_token: str, github_api_url: str) -> bool:
    """
    Verify if the GitHub token is valid by making a test API call.

    Args:
        github_token: GitHub personal access token
        github_api_url: GitHub API URL

    Returns:
        True if token is valid, False otherwise
    """
    try:
        headers = {
            'Authorization': f'Bearer {github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'X-GitHub-Api-Version': '2022-11-28'
        }
        url = f'{github_api_url}/user'
        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error verifying GitHub token: {e}")
        return False


# Page configuration
st.set_page_config(
    page_title="TARA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with green theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2D5F2E;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5a6c57;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        border-radius: 4px;
        padding: 0.6rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    }
    .success-box {
        padding: 1rem;
        border-radius: 4px;
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        color: #2e7d32;
    }
    .info-box {
        padding: 1rem;
        border-radius: 4px;
        background-color: #f1f8f4;
        border-left: 4px solid #81c784;
        margin: 1rem 0;
        color: #2D5F2E;
    }
    .metric-card {
        background-color: #f1f8f4;
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #c8e6c9;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8fdf9;
    }
    /* Header separator */
    .header-line {
        height: 3px;
        background: linear-gradient(90deg, #4CAF50 0%, #81c784 100%);
        margin: 1rem 0 2rem 0;
    }
    /* Chat styling */
    .stChatMessage {
        background-color: #f8fdf9;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .stChatInput > div {
        border-color: #4CAF50 !important;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">TARA - Your Onboarding Buddy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered Learning Path Generator for easy project onboarding.</div>', unsafe_allow_html=True)
st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)

# Login page - shown if not authenticated
if not st.session_state.authenticated:
    st.markdown("---")
    st.markdown("### Authentication")
    st.markdown('<div class="info-box">Please enter your GitHub token and optionally your Google Drive token to get started.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### GitHub Authentication")
        github_token_input = st.text_input(
            "GitHub Personal Access Token",
            type="password",
            placeholder="ghp_xxxxxxxxxxxxxxxxxxxx",
            help="Enter your GitHub personal access token. This token will be used to fetch your accessible repositories."
        )

    with col2:
        st.markdown("#### Google Drive Authentication (Optional)")
        drive_token_input = st.text_input(
            "Google Drive Token",
            type="password",
            placeholder="Enter your Drive token (optional)",
            help="Optional: Enter your Google Drive token to enable Drive search integration."
        )

    if st.button("Login and Load Repositories", type="primary", use_container_width=True):
        if not github_token_input:
            st.error("GitHub token is required")
        else:
            with st.spinner("Verifying GitHub token and fetching your repositories..."):
                # Verify GitHub token
                if not verify_github_token(github_token_input, st.session_state.github_api_url):
                    st.error("Invalid GitHub token. Please check your token and try again.")
                else:
                    # Fetch user repositories
                    repos = fetch_user_repos(github_token_input, st.session_state.github_api_url)

                    if not repos:
                        st.warning("No repositories found or error fetching repositories. Please check your token permissions.")
                    else:
                        # Store tokens and repos in session state
                        st.session_state.github_token = github_token_input
                        st.session_state.drive_token = drive_token_input
                        st.session_state.user_repos = repos
                        st.session_state.authenticated = True
                        st.success(f"Successfully authenticated. Found {len(repos)} accessible repositories.")
                        st.rerun()

    st.markdown("---")
    st.markdown("""
    **How to get your GitHub Personal Access Token:**
    1. Go to GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
    2. Click "Generate new token (classic)"
    3. Give it a name and select scopes: `repo` (for private repos) or `public_repo` (for public only)
    4. Click "Generate token"
    5. Copy the token and paste it above
    """)

    # Stop rendering the rest of the page
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Configuration")

    # User info and logout
    st.markdown("### User Session")
    st.markdown('<div class="success-box" style="margin: 0; padding: 0.5rem;">Authenticated</div>', unsafe_allow_html=True)
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.github_token = ""
        st.session_state.drive_token = ""
        st.session_state.user_repos = []
        st.rerun()

    st.markdown("### Repository Settings")

    # Create a searchable list of repositories
    if st.session_state.user_repos:
        # Create options for selectbox
        repo_options = [""] + [r['full_name'] for r in st.session_state.user_repos]
        repo_names = ["-- Select a repository --"] + [
            f"{r['full_name']}" + (" (private)" if r['private'] else "") + (f" - {r['description'][:40]}..." if r['description'] else "")
            for r in st.session_state.user_repos
        ]

        # Searchable selectbox
        selected_index = st.selectbox(
            "Select GitHub Repository",
            range(len(repo_options)),
            format_func=lambda i: repo_names[i],
            help="Search and select a repository from your accessible repositories"
        )

        repo_input = repo_options[selected_index]

        # Show repo count
        st.caption(f"{len(st.session_state.user_repos)} repositories available")
    else:
        st.warning("No repositories loaded. Please logout and login again.")
        repo_input = ""

    st.markdown("### Integration Options")
    enable_drive = st.checkbox(
        "Enable Google Drive Search",
        value=False,
        help="Search Google Drive for reference documentation"
    )

    st.markdown("### Output Settings")
    output_file = st.text_input(
        "Output Filename (optional)",
        placeholder="documentation.md",
        help="Leave empty to auto-generate filename"
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Powered by GPT-4o**

    **Agents:**
    - GitHub Data Fetcher
    - Learning Path Writer
    """)

# File viewer section
with st.expander("View Existing Learning Paths", expanded=False):
    st.markdown("Load and view previously generated learning paths")

    # List existing markdown files
    existing_files = list(Path(".").glob("learning_path_*.md")) + list(Path(".").glob("documentation_*.md"))

    if existing_files:
        selected_file = st.selectbox(
            "Select a file to view:",
            options=existing_files,
            format_func=lambda x: x.name
        )

        if selected_file and st.button("Load File"):
            with open(selected_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract and clean markdown (in case it has JSON wrapping)
            cleaned_content = extract_markdown_from_response(content)

            st.markdown("---")
            st.markdown("### File Content")

            # Show in tabs
            view_tab1, view_tab2 = st.tabs(["Preview", "Download"])

            with view_tab1:
                st.markdown(cleaned_content, unsafe_allow_html=False)

            with view_tab2:
                st.download_button(
                    label="Download Markdown",
                    data=cleaned_content,
                    file_name=selected_file.name,
                    mime="text/markdown"
                )
    else:
        st.info("No learning path files found. Generate a learning path first.")

st.markdown("---")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Generate Learning Path")

    # Validation feedback
    if repo_input:
        if validate_github_repo(repo_input):
            st.markdown(f'<div class="success-box">Valid repository format: <strong>{repo_input}</strong></div>', unsafe_allow_html=True)
        else:
            st.error("Invalid repository format. Expected: owner/repo-name")

    # Generate button
    generate_button = st.button("Generate Learning Path", type="primary", use_container_width=True)

with col2:
    st.markdown("### Agent Status")
    agent_count = 2  # Minimum: GitHub + Writer
    if enable_drive:
        agent_count += 1

    st.markdown(f'<div class="metric-card"><h2 style="color: #4CAF50; margin: 0;">{agent_count}</h2><p style="margin: 0; color: #5a6c57;">Active Agents</p></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f"""
    **Enabled Agents:**
    - {'Y' if True else 'X'} GitHub Data Analyzer
    - {'Y' if enable_drive else 'X'} Google Drive Analyzer
    - {'Y' if True else 'X'} Learning Path Writer
    """)

# Learning path generation
if generate_button:
    if not repo_input:
        st.error("Please select a GitHub repository")
    elif not validate_github_repo(repo_input):
        st.error("Invalid repository format. Expected: owner/repo-name")
    else:
        try:
            # Progress container
            with st.container():
                st.markdown("---")
                st.markdown("### Generation Progress")

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Initialize crew with runtime tokens
                status_text.text("Initializing learning path crew...")
                progress_bar.progress(10)

                # Configure settings with runtime tokens
                # Set environment variables for OpenAI
                os.environ["GITHUB_TOKEN"] = st.session_state.github_token
                if st.session_state.drive_token:
                    os.environ["GOOGLE_DRIVE_TOKEN"] = st.session_state.drive_token

                get_settings(
                    github_token=st.session_state.github_token,
                    drive_token=st.session_state.drive_token if enable_drive else None,
                    force_reload=True
                )

                doc_crew = DocumentationCrew(
                    enable_google_drive=enable_drive
                )

                # Generate learning path
                status_text.text(f"Generating learning path for {repo_input}...")
                progress_bar.progress(30)

                with st.spinner(f"Agents are collaborating to create your learning path..."):
                    documentation = doc_crew.generate_documentation(repo_input)

                progress_bar.progress(80)
                status_text.text("Saving learning path...")

                # Save learning path
                output_path = doc_crew.save_documentation(
                    documentation,
                    output_file if output_file else None
                )

                progress_bar.progress(100)
                status_text.text("Learning path generated successfully")

                # Success message
                st.markdown(f'<div class="success-box"><strong>Learning path saved to:</strong> {output_path}</div>', unsafe_allow_html=True)

                # Display learning path
                st.markdown("---")
                st.markdown("### Generated Learning Path")

                # Tabs for different views
                tab1, tab2 = st.tabs(["Preview", "Download"])

                with tab1:
                    # Render markdown properly
                    markdown_content = documentation.get("documentation", "")
                    st.markdown(markdown_content, unsafe_allow_html=False)

                with tab2:
                    st.download_button(
                        label="Download Learning Path",
                        data=documentation.get("documentation", ""),
                        file_name=os.path.basename(output_path),
                        mime="text/markdown"
                    )

                    st.info(f"File also saved locally at: `{output_path}`")

        except ValueError as e:
            st.error(f"Validation Error: {str(e)}")
            logger.error(f"Validation error: {e}")

        except Exception as e:
            st.error(f"Error generating learning path: {str(e)}")
            logger.error(f"Error generating learning path: {e}", exc_info=True)

            with st.expander("Error Details"):
                st.code(str(e))

# Code Chat Section (only show if user has selected a repository)
if repo_input and validate_github_repo(repo_input):
    st.markdown("---")
    st.markdown("### Chat with the Code")
    st.markdown(f"Ask questions about **{repo_input}** - I'll analyze the code and help you understand it.")

    # Initialize chat history for this repo
    chat_key = f"chat_history_{repo_input.replace('/', '_')}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # Directory selector in a small expander
    with st.expander("Settings", expanded=False):
        code_directory = st.text_input(
            "Directory to search",
            value=".",
            help="Which directory to look for code (e.g., src, lib, . for root)"
        )
        if st.button("Clear Chat History"):
            st.session_state[chat_key] = []
            st.rerun()

    # Handle pending question from button click (before displaying history)
    pending_key = f"pending_question_{chat_key}"
    if pending_key in st.session_state and st.session_state[pending_key]:
        pending_q = st.session_state[pending_key]
        st.session_state[pending_key] = None

        # Add user message
        st.session_state[chat_key].append({"role": "user", "content": pending_q})

        # Get AI response with loading indicator
        with st.spinner("Analyzing code..."):
            try:
                os.environ["GITHUB_TOKEN"] = st.session_state.github_token
                get_settings(github_token=st.session_state.github_token, drive_token=None, force_reload=True)
                doc_crew = DocumentationCrew(enable_google_drive=False)
                qa_result = doc_crew.answer_code_question(
                    repo=repo_input,
                    question=pending_q,
                    directory=code_directory,
                    chat_history=st.session_state[chat_key][:-1]  # Exclude the question we just added
                )
                answer = qa_result.get("answer", "Sorry, I couldn't analyze the code.")
                st.session_state[chat_key].append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state[chat_key].append({"role": "assistant", "content": f"Error: {str(e)}"})

    # Display chat history
    for message in st.session_state[chat_key]:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])

    # Example questions as clickable buttons (only show if no history)
    if not st.session_state[chat_key]:
        st.markdown("**Quick questions:**")
        example_cols = st.columns(2)
        examples = [
            ("What does this project do?", "Explain the entry point"),
            ("How is the code structured?", "What dependencies are used?")
        ]
        for col_idx, col in enumerate(example_cols):
            with col:
                for row_idx in range(2):
                    example = examples[row_idx][col_idx]
                    if st.button(example, key=f"example_{col_idx}_{row_idx}", use_container_width=True):
                        st.session_state[pending_key] = example
                        st.rerun()

    # Chat input at the bottom
    if prompt := st.chat_input("Ask about the code..."):
        # Add user message to history
        st.session_state[chat_key].append({"role": "user", "content": prompt})

        # Get AI response
        with st.spinner("Analyzing code..."):
            try:
                os.environ["GITHUB_TOKEN"] = st.session_state.github_token
                get_settings(github_token=st.session_state.github_token, drive_token=None, force_reload=True)
                doc_crew = DocumentationCrew(enable_google_drive=False)
                qa_result = doc_crew.answer_code_question(
                    repo=repo_input,
                    question=prompt,
                    directory=code_directory,
                    chat_history=st.session_state[chat_key][:-1]  # Pass history excluding current question
                )
                answer = qa_result.get("answer", "Sorry, I couldn't analyze the code.")
                st.session_state[chat_key].append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state[chat_key].append({"role": "assistant", "content": f"Error: {str(e)}"})

        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #5a6c57; padding: 1rem; font-size: 0.9rem;">
    TARA - AI-powered Learning Path Generator
</div>
""", unsafe_allow_html=True)
