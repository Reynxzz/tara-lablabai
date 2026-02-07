"""Configuration settings loaded from environment variables"""
import os
from typing import Optional
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables
load_dotenv()


@dataclass
class GitHubConfig:
    """GitHub configuration"""
    token: Optional[str]
    api_url: str

    @classmethod
    def from_env(cls, runtime_token: Optional[str] = None) -> "GitHubConfig":
        token = runtime_token or os.getenv("GITHUB_TOKEN")
        api_url = os.getenv("GITHUB_API_URL", "https://api.github.com")

        if not token:
            raise ValueError("GITHUB_TOKEN is required (either from environment or runtime)")

        return cls(token=token, api_url=api_url)


@dataclass
class GoogleDriveConfig:
    """Google Drive configuration"""
    token: Optional[str]
    mcp_url: str

    @classmethod
    def from_env(cls, runtime_token: Optional[str] = None) -> "GoogleDriveConfig":
        return cls(
            token=runtime_token or os.getenv("GOOGLE_DRIVE_TOKEN"),
            mcp_url=os.getenv("MCP_DRIVE_URL", "drive.taraai.tech")
        )

    def is_configured(self) -> bool:
        """Check if Google Drive is properly configured"""
        return self.token is not None and len(self.token) > 0


@dataclass
class LLMConfig:
    """LLM configuration for OpenAI"""
    api_key: str
    timeout: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        return cls(
            api_key=api_key,
            timeout=int(os.getenv("LLM_TIMEOUT", "300"))
        )


@dataclass
class Settings:
    """Global application settings"""
    github: GitHubConfig
    google_drive: GoogleDriveConfig
    llm: LLMConfig

    @classmethod
    def load(cls, github_token: Optional[str] = None, drive_token: Optional[str] = None) -> "Settings":
        """Load all settings from environment or runtime parameters"""
        return cls(
            github=GitHubConfig.from_env(runtime_token=github_token),
            google_drive=GoogleDriveConfig.from_env(runtime_token=drive_token),
            llm=LLMConfig.from_env()
        )


# Global settings instance
settings: Optional[Settings] = None


def get_settings(github_token: Optional[str] = None, drive_token: Optional[str] = None, force_reload: bool = False) -> Settings:
    """Get or create global settings instance"""
    global settings
    if settings is None or force_reload or github_token or drive_token:
        settings = Settings.load(github_token=github_token, drive_token=drive_token)
    return settings
