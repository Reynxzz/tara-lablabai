"""Input validation utilities"""
import re
from typing import Optional


def validate_github_repo(repo: str) -> bool:
    """
    Validate GitHub repository format (owner/repo).

    Args:
        repo: Repository string to validate

    Returns:
        True if valid, False otherwise
    """
    if not repo:
        return False

    # GitHub repo format: owner/repo
    # Owner and repo can contain alphanumeric, hyphens, and underscores
    # Owner cannot start with hyphen
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9_-]*/[a-zA-Z0-9][a-zA-Z0-9_.-]*$'
    return bool(re.match(pattern, repo))


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
