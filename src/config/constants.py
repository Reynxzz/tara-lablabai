"""Constants for the documentation generation system"""
from enum import Enum
from typing import List


class LLMModel(str, Enum):
    """Available LLM models"""
    GPT_OSS = "openai/gpt-oss-120b"
    SAHABAT_4BIT = "GoToCompany/Llama-Sahabat-AI-v2-70B-IT-awq-4bit"
    EMBEDDING_GEMMA = "google/embeddinggemma-300m"


class AgentRole(str, Enum):
    """Agent role definitions"""
    GITLAB_ANALYZER = "GitLab Data Analyzer"
    DRIVE_ANALYZER = "Google Drive Reference Analyzer"
    RAG_ANALYZER = "Internal Knowledge Base Analyzer"
    LEARNING_PATH_WRITER = "Learning Path Writer"
    CODE_QA_AGENT = "Code Q&A Agent"


class ToolName(str, Enum):
    """Tool name definitions"""
    GITLAB_TOOL = "GitLab Project Analyzer"
    DRIVE_TOOL = "Google Drive Document Analyzer"
    RAG_TOOL = "Internal Knowledge Base Search"


# RAG Milvus collection name (single combined collection)
RAG_COLLECTION_NAME = "combined_item"

# Default values
DEFAULT_TEMPERATURE_TOOL_CALLING = 0.3
DEFAULT_TEMPERATURE_WRITING = 0.6
DEFAULT_RAG_TOP_K = 5
DEFAULT_DRIVE_TOP_K = 3
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_EMBEDDING_TIMEOUT = 30
