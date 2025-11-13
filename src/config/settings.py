"""Configuration settings loaded from environment variables"""
import os
from typing import Optional
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables
load_dotenv()


@dataclass
class GitLabConfig:
    """GitLab configuration"""
    token: Optional[str]
    url: str

    @classmethod
    def from_env(cls, runtime_token: Optional[str] = None) -> "GitLabConfig":
        token = runtime_token or os.getenv("GITLAB_TOKEN")
        url = os.getenv("GITLAB_URL", "https://source.golabs.io")

        if not token:
            raise ValueError("GITLAB_TOKEN is required (either from environment or runtime)")

        return cls(token=token, url=url)


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
    """LLM endpoint configuration"""
    endpoint: str
    timeout: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            endpoint=os.getenv("LLM_ENDPOINT", "https://litellm-staging.gopay.sh"),
            timeout=int(os.getenv("LLM_TIMEOUT", "300"))
        )


@dataclass
class RAGConfig:
    """RAG Milvus configuration"""
    db_path: str
    embedding_endpoint: str
    embedding_model: str
    top_k: int

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            db_path=os.getenv("MILVUS_DB_PATH", "./milvus_demo_batch_bmth_v1.db"),
            embedding_endpoint=os.getenv("EMBEDDING_ENDPOINT", "https://litellm-staging.gopay.sh/embeddings"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m"),
            top_k=int(os.getenv("RAG_TOP_K", "5"))
        )

    def is_configured(self) -> bool:
        """Check if RAG database is available"""
        return os.path.exists(self.db_path)


@dataclass
class Settings:
    """Global application settings"""
    gitlab: GitLabConfig
    google_drive: GoogleDriveConfig
    llm: LLMConfig
    rag: RAGConfig

    @classmethod
    def load(cls, gitlab_token: Optional[str] = None, drive_token: Optional[str] = None) -> "Settings":
        """Load all settings from environment or runtime parameters"""
        return cls(
            gitlab=GitLabConfig.from_env(runtime_token=gitlab_token),
            google_drive=GoogleDriveConfig.from_env(runtime_token=drive_token),
            llm=LLMConfig.from_env(),
            rag=RAGConfig.from_env()
        )


# Global settings instance
settings: Optional[Settings] = None


def get_settings(gitlab_token: Optional[str] = None, drive_token: Optional[str] = None, force_reload: bool = False) -> Settings:
    """Get or create global settings instance"""
    global settings
    if settings is None or force_reload or gitlab_token or drive_token:
        settings = Settings.load(gitlab_token=gitlab_token, drive_token=drive_token)
    return settings
