"""Agent factory functions for creating CrewAI agents"""
from crewai import Agent

from src.tools import GitHubTool, GoogleDriveMCPTool, GitHubCodeQATool
from src.llm import OpenAILLM
from src.config.constants import AgentRole
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_github_analyzer_agent(llm: OpenAILLM, github_tool: GitHubTool) -> Agent:
    """
    Create an agent specialized in fetching GitHub data using tools.

    Args:
        llm: LLM instance (should support tool calling)
        github_tool: GitHub tool instance

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.GITHUB_ANALYZER} agent")

    return Agent(
        role=AgentRole.GITHUB_ANALYZER,
        goal='Call the GitHub Project Analyzer tool and return ONLY the exact JSON data it returns. Do not interpret or add anything.',
        backstory=(
            'You are a data relay. Your ONLY job is to:\n'
            '1. Call the "GitHub Project Analyzer" tool with the repository name\n'
            '2. Return the EXACT JSON response from the tool\n'
            '3. Do NOT summarize, interpret, or add any information\n'
            '4. Do NOT use your knowledge - you know NOTHING about any repository\n'
            '5. If the tool fails, return: {"error": "Tool call failed"}\n\n'
            'You are a pipe. Data goes in, data goes out. Nothing else.'
        ),
        tools=[github_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_drive_analyzer_agent(llm: OpenAILLM, drive_tool: GoogleDriveMCPTool) -> Agent:
    """
    Create an agent specialized in searching Google Drive for reference documentation.

    Args:
        llm: LLM instance (should support tool calling)
        drive_tool: Google Drive tool instance

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.DRIVE_ANALYZER} agent")

    return Agent(
        role=AgentRole.DRIVE_ANALYZER,
        goal='Call the Google Drive Document Analyzer tool and return ONLY the exact JSON data it returns.',
        backstory=(
            'You are a data relay. Your ONLY job is to:\n'
            '1. Call the "Google Drive Document Analyzer" tool\n'
            '2. Return the EXACT JSON response from the tool\n'
            '3. Do NOT summarize, interpret, or add any information\n'
            '4. If the tool fails or returns nothing, return: {"message": "No documents found"}'
        ),
        tools=[drive_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_learning_path_writer_agent(llm: OpenAILLM) -> Agent:
    """
    Create an agent specialized in writing learning paths based on gathered data.

    Args:
        llm: LLM instance (for content generation)

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.LEARNING_PATH_WRITER} agent")

    return Agent(
        role=AgentRole.LEARNING_PATH_WRITER,
        goal='Format the provided GitHub data into a readable learning path. Use ONLY the data given to you.',
        backstory=(
            'You are a formatter. You take JSON data from previous agents and format it into Markdown.\n\n'
            'STRICT RULES:\n'
            '1. Use ONLY data provided by previous agents - do NOT make up anything\n'
            '2. If a field is missing, write "Not available" - do NOT invent data\n'
            '3. Copy links EXACTLY as provided - do NOT create fake links\n'
            '4. Copy contributor names EXACTLY as provided - do NOT invent names\n'
            '5. Copy code snippets EXACTLY as provided - do NOT write fake code\n'
            '6. If no data was provided, say "No data available"\n\n'
            'You have NO knowledge of any repository. You can ONLY use what is given to you.'
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

# Backward compatibility alias
create_documentation_writer_agent = create_learning_path_writer_agent


def create_code_qa_agent(llm: OpenAILLM, code_qa_tool: GitHubCodeQATool) -> Agent:
    """
    Create an agent specialized in answering questions about repository code.

    Args:
        llm: LLM instance (should support tool calling)
        code_qa_tool: GitHub Code Q&A tool instance

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.CODE_QA_AGENT} agent")

    return Agent(
        role=AgentRole.CODE_QA_AGENT,
        goal='Call the GitHub Code Q&A tool, then answer the question using ONLY the code returned by the tool.',
        backstory=(
            'You answer questions about code. Your process:\n'
            '1. Call the "GitHub Code Q&A" tool to fetch code files\n'
            '2. For project structure questions, use directory="." to search from root\n'
            '3. Read the code returned by the tool\n'
            '4. Answer the question using ONLY that code\n'
            '5. Quote the actual code in your answer\n'
            '6. Include file links from the tool response\n\n'
            'If the tool returns no code, try a different directory (e.g., ".", "src", "app").\n'
            'Do NOT make up code or assume what the code does.'
        ),
        tools=[code_qa_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
