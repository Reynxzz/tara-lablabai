"""Constants for the documentation generation system"""
from enum import Enum


class LLMModel(str, Enum):
    """Available LLM models"""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


class AgentRole(str, Enum):
    """Agent role definitions"""
    GITHUB_ANALYZER = "GitHub Data Analyzer"
    DRIVE_ANALYZER = "Google Drive Reference Analyzer"
    LEARNING_PATH_WRITER = "Learning Path Writer"
    CODE_QA_AGENT = "Code Q&A Agent"


class ToolName(str, Enum):
    """Tool name definitions"""
    GITHUB_TOOL = "GitHub Project Analyzer"
    DRIVE_TOOL = "Google Drive Document Analyzer"


# Default values
DEFAULT_TEMPERATURE_TOOL_CALLING = 0.3
DEFAULT_TEMPERATURE_WRITING = 0.6
DEFAULT_DRIVE_TOP_K = 3
DEFAULT_REQUEST_TIMEOUT = 300
