"""Streamlit web interface for NoBuddy - Documentation Generator"""
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
from src.utils.validators import validate_gitlab_project
from src.config.settings import get_settings

logger = setup_logger(__name__)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'gitlab_token' not in st.session_state:
    st.session_state.gitlab_token = ""
if 'drive_token' not in st.session_state:
    st.session_state.drive_token = ""
if 'user_projects' not in st.session_state:
    st.session_state.user_projects = []
if 'gitlab_url' not in st.session_state:
    st.session_state.gitlab_url = os.getenv("GITLAB_URL", "https://source.golabs.io")


def fetch_user_projects(gitlab_token: str, gitlab_url: str) -> List[Dict[str, Any]]:
    """
    Fetch all projects accessible by the user from GitLab API.

    Args:
        gitlab_token: GitLab personal access token
        gitlab_url: GitLab instance URL

    Returns:
        List of project dictionaries with id, name, and path_with_namespace
    """
    try:
        headers = {'PRIVATE-TOKEN': gitlab_token}
        projects = []
        page = 1
        per_page = 100

        while True:
            url = f'{gitlab_url}/api/v4/projects'
            response = requests.get(
                url,
                headers=headers,
                params={
                    'membership': True,  # Only projects user is a member of
                    'per_page': per_page,
                    'page': page,
                    'order_by': 'last_activity_at',
                    'sort': 'desc'
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Failed to fetch projects: HTTP {response.status_code} - {response.text[:200]}")
                return []

            try:
                page_projects = response.json()
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return []

            if not page_projects:
                break

            if not isinstance(page_projects, list):
                logger.error(f"Unexpected response format: {type(page_projects)}")
                return []

            for project in page_projects:
                # Skip if project is None or doesn't have required fields
                if not project or not isinstance(project, dict):
                    continue

                path_with_namespace = project.get('path_with_namespace')
                if not path_with_namespace:
                    continue

                description = project.get('description', '') or ''
                projects.append({
                    'id': project.get('id'),
                    'name': project.get('name'),
                    'path_with_namespace': path_with_namespace,
                    'description': description[:100] if description else '',  # Truncate description
                    'last_activity_at': project.get('last_activity_at', '')
                })

            # Check if there are more pages
            if len(page_projects) < per_page:
                break
            page += 1

        logger.info(f"Fetched {len(projects)} projects for user")
        return projects

    except Exception as e:
        logger.error(f"Error fetching user projects: {e}")
        return []


def verify_gitlab_token(gitlab_token: str, gitlab_url: str) -> bool:
    """
    Verify if the GitLab token is valid by making a test API call.

    Args:
        gitlab_token: GitLab personal access token
        gitlab_url: GitLab instance URL

    Returns:
        True if token is valid, False otherwise
    """
    try:
        headers = {'PRIVATE-TOKEN': gitlab_token}
        url = f'{gitlab_url}/api/v4/user'
        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error verifying GitLab token: {e}")
        return False


# Page configuration
st.set_page_config(
    page_title="NoBuddy",
    page_icon="🟢",
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
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">NoBuddy - Your Onboarding Buddy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your team’s knowledge, instantly searchable for easy project onboarding. Turns "nobody knows" into "NoBuddy knows"</div>', unsafe_allow_html=True)
st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)

# Login page - shown if not authenticated
if not st.session_state.authenticated:
    st.markdown("---")
    st.markdown("### Authentication")
    st.markdown('<div class="info-box">Please enter your GitLab token and optionally your Google Drive token to get started.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### GitLab Authentication")
        gitlab_token_input = st.text_input(
            "GitLab Personal Access Token",
            type="password",
            placeholder="glpat-xxxxxxxxxxxxxxxxxxxx",
            help="Enter your GitLab personal access token. This token will be used to fetch your accessible projects."
        )

    with col2:
        st.markdown("#### Google Drive Authentication (Optional)")
        drive_token_input = st.text_input(
            "Google Drive Token",
            type="password",
            placeholder="Enter your Drive token (optional)",
            help="Optional: Enter your Google Drive token to enable Drive search integration."
        )

    if st.button("Login and Load Projects", type="primary", use_container_width=True):
        if not gitlab_token_input:
            st.error("✗ GitLab token is required")
        else:
            with st.spinner("Verifying GitLab token and fetching your projects..."):
                # Verify GitLab token
                if not verify_gitlab_token(gitlab_token_input, st.session_state.gitlab_url):
                    st.error("✗ Invalid GitLab token. Please check your token and try again.")
                else:
                    # Fetch user projects
                    projects = fetch_user_projects(gitlab_token_input, st.session_state.gitlab_url)

                    if not projects:
                        st.warning("No projects found or error fetching projects. Please check your token permissions.")
                    else:
                        # Store tokens and projects in session state
                        st.session_state.gitlab_token = gitlab_token_input
                        st.session_state.drive_token = drive_token_input
                        st.session_state.user_projects = projects
                        st.session_state.authenticated = True
                        st.success(f"✓ Successfully authenticated. Found {len(projects)} accessible projects.")
                        st.rerun()

    st.markdown("---")
    st.markdown("""
    **How to get your GitLab Personal Access Token:**
    1. Go to your GitLab instance → User Settings → Access Tokens
    2. Create a new token with `read_api` and `read_repository` scopes
    3. Copy the token and paste it above
    """)

    # Stop rendering the rest of the page
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Configuration")

    # User info and logout
    st.markdown("### User Session")
    st.markdown('<div class="success-box" style="margin: 0; padding: 0.5rem;">✓ Authenticated</div>', unsafe_allow_html=True)
    if st.button("Logout ⏻", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.gitlab_token = ""
        st.session_state.drive_token = ""
        st.session_state.user_projects = []
        st.rerun()

    st.markdown("### Project Settings")

    # Create a searchable list of projects
    if st.session_state.user_projects:
        # Create options for selectbox
        project_options = [""] + [p['path_with_namespace'] for p in st.session_state.user_projects]
        project_names = ["-- Select a project --"] + [
            f"{p['path_with_namespace']}" + (f" - {p['description'][:50]}..." if p['description'] else "")
            for p in st.session_state.user_projects
        ]

        # Searchable selectbox
        selected_index = st.selectbox(
            "Select GitLab Project",
            range(len(project_options)),
            format_func=lambda i: project_names[i],
            help="Search and select a project from your accessible projects"
        )

        project_input = project_options[selected_index]

        # Show project count
        st.caption(f"{len(st.session_state.user_projects)} projects available")
    else:
        st.warning("No projects loaded. Please logout and login again.")
        project_input = ""

    st.markdown("### Integration Options")
    enable_drive = st.checkbox(
        "Enable Google Drive Search",
        value=False,
        help="Search Google Drive for reference documentation"
    )

    enable_rag = st.checkbox(
        "Enable Internal Knowledge Base",
        value=False,
        help="Search internal Milvus knowledge base for relevant information"
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
    This tool uses a **dual-LLM architecture**:
    - GPT OSS 120B: For tool calling and data fetching
    - Sahabat AI 70B: For documentation writing

    **Agents:**
    - GitLab Data Analyzer
    - Google Drive Analyzer (optional)
    - Internal KB Analyzer (optional)
    - Documentation Writer
    """)

# File viewer section
with st.expander("View Existing Documentation", expanded=False):
    st.markdown("Load and view previously generated documentation files")

    # List existing markdown files
    existing_files = list(Path(".").glob("documentation_*.md"))

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
        st.info("No documentation files found. Generate some documentation first.")

st.markdown("---")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Generate Documentation")

    # Validation feedback
    if project_input:
        if validate_gitlab_project(project_input):
            st.markdown(f'<div class="success-box">✓ Valid project format: <strong>{project_input}</strong></div>', unsafe_allow_html=True)
        else:
            st.error("✗ Invalid project format. Expected: namespace/project-name")

    # Generate button
    generate_button = st.button("Generate Documentation", type="primary", use_container_width=True)

with col2:
    st.markdown("### Agent Status")
    agent_count = 2  # Minimum: GitLab + Writer
    if enable_drive:
        agent_count += 1
    if enable_rag:
        agent_count += 1

    st.markdown(f'<div class="metric-card"><h2 style="color: #4CAF50; margin: 0;">{agent_count}</h2><p style="margin: 0; color: #5a6c57;">Active Agents</p></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f"""
    **Enabled Agents:**
    - {'✓' if True else '✗'} GitLab Data Analyzer
    - {'✓' if enable_drive else '✗'} Google Drive Analyzer
    - {'✓' if enable_rag else '✗'} Internal KB Analyzer
    - {'✓' if True else '✗'} Documentation Writer
    """)

# Documentation generation
if generate_button:
    if not project_input:
        st.error("✗ Please enter a GitLab project path")
    elif not validate_gitlab_project(project_input):
        st.error("✗ Invalid project format. Expected: namespace/project-name")
    else:
        try:
            # Progress container
            with st.container():
                st.markdown("---")
                st.markdown("### Generation Progress")

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Initialize crew with runtime tokens
                status_text.text("Initializing documentation crew...")
                progress_bar.progress(10)

                # Configure settings with runtime tokens
                get_settings(
                    gitlab_token=st.session_state.gitlab_token,
                    drive_token=st.session_state.drive_token if enable_drive else None,
                    force_reload=True
                )

                doc_crew = DocumentationCrew(
                    enable_google_drive=enable_drive,
                    enable_rag=enable_rag
                )

                # Generate documentation
                status_text.text(f"Generating documentation for {project_input}...")
                progress_bar.progress(30)

                with st.spinner(f"Agents are collaborating to generate documentation..."):
                    documentation = doc_crew.generate_documentation(project_input)

                progress_bar.progress(80)
                status_text.text("Saving documentation...")

                # Save documentation
                output_path = doc_crew.save_documentation(
                    documentation,
                    output_file if output_file else None
                )

                progress_bar.progress(100)
                status_text.text("✓ Documentation generated successfully")

                # Success message
                st.markdown(f'<div class="success-box">✓ <strong>Documentation saved to:</strong> {output_path}</div>', unsafe_allow_html=True)

                # Display documentation
                st.markdown("---")
                st.markdown("### Generated Documentation")

                # Tabs for different views
                tab1, tab2 = st.tabs(["Preview", "Download"])

                with tab1:
                    # Render markdown properly
                    markdown_content = documentation.get("documentation", "")
                    st.markdown(markdown_content, unsafe_allow_html=False)

                with tab2:
                    st.download_button(
                        label="Download Markdown",
                        data=documentation.get("documentation", ""),
                        file_name=os.path.basename(output_path),
                        mime="text/markdown"
                    )

                    st.info(f"File also saved locally at: `{output_path}`")

        except ValueError as e:
            st.error(f"✗ Validation Error: {str(e)}")
            logger.error(f"Validation error: {e}")

        except Exception as e:
            st.error(f"✗ Error generating documentation: {str(e)}")
            logger.error(f"Error generating documentation: {e}", exc_info=True)

            with st.expander("Error Details"):
                st.code(str(e))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #5a6c57; padding: 1rem; font-size: 0.9rem;">
    Built for GoTo Hackathon 2025 by Bring Me The Hackathon.
</div>
""", unsafe_allow_html=True)
