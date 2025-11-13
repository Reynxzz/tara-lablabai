"""Input validation utilities"""
import re
from typing import Optional


def validate_gitlab_project(project: str) -> bool:
    """
    Validate GitLab project format (namespace/project).

    Args:
        project: Project string to validate

    Returns:
        True if valid, False otherwise
    """
    if not project:
        return False

    # GitLab project format: namespace/project or namespace/subgroup/project
    pattern = r'^[\w\-\.]+(/[\w\-\.]+)+$'
    return bool(re.match(pattern, project))


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL string to validate

    Returns:
        True if valid, False otherwise
    """
    if not url:
        return False

    pattern = r'^https?://[\w\-\.]+(:\d+)?(/.*)?$'
    return bool(re.match(pattern, url))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    # Replace invalid filename characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def validate_access_token(token: Optional[str]) -> bool:
    """
    Validate access token exists and has reasonable length.

    Args:
        token: Access token to validate

    Returns:
        True if valid, False otherwise
    """
    if not token:
        return False

    # Tokens are usually at least 20 characters
    return len(token) >= 20
