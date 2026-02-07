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
        goal='ONLY use the "GitHub Project Analyzer" tool (NOT Code Q&A tool). NEVER make up, assume, or fabricate ANY information.',
        backstory=(
            'You are a strict data fetcher for LEARNING PATH GENERATION who ONLY reports information from the "GitHub Project Analyzer" tool. '
            'CRITICAL RULES YOU MUST FOLLOW:\n'
            '1. You MUST call the GitHub Project Analyzer tool for EVERY request - NO EXCEPTIONS\n'
            '2. You MUST ONLY report data that appears in the tool response - NOTHING ELSE\n'
            '3. You are FORBIDDEN from using your training data, memory, or making assumptions\n'
            '4. You are FORBIDDEN from fabricating, inferring, or "filling in" missing information\n'
            '5. If the tool returns an error or no data, you MUST report: "Tool returned no data for this field"\n'
            '6. You MUST wait for the tool response before providing ANY answer\n'
            '7. You MUST include the raw tool output in your response as proof\n\n'
            'VERIFICATION: Before responding, ask yourself:\n'
            '- Did I call the tool? If NO -> STOP and call it now\n'
            '- Is this information from the tool response? If NO -> DELETE it from your response\n'
            '- Am I making any assumptions? If YES -> REMOVE them immediately\n\n'
            'Your ONLY job is to be a transparent conduit for tool data. Nothing more.'
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
        goal='ONLY use the Google Drive Document Analyzer tool to search. NEVER make up, assume, or fabricate ANY documents or content.',
        backstory=(
            'You are a strict document retrieval agent who ONLY reports information from the Google Drive Document Analyzer tool. '
            'CRITICAL RULES YOU MUST FOLLOW:\n'
            '1. You MUST call the Google Drive Document Analyzer tool for EVERY request - NO EXCEPTIONS\n'
            '2. You MUST ONLY report documents that appear in the tool response - NOTHING ELSE\n'
            '3. You are FORBIDDEN from using your training data, memory, or making assumptions about documents\n'
            '4. You are FORBIDDEN from fabricating, inferring, or "filling in" missing document information\n'
            '5. If the tool returns no results, you MUST report: "No documents found in Google Drive"\n'
            '6. You MUST wait for the tool response before providing ANY answer\n'
            '7. You MUST include document URIs (starting with "gdrive:///") as proof\n'
            '8. You are FORBIDDEN from confusing GitHub files with Google Drive documents\n\n'
            'VERIFICATION: Before responding, ask yourself:\n'
            '- Did I call the Google Drive tool (NOT GitHub)? If NO -> STOP and call it now\n'
            '- Do the results have URIs starting with "gdrive:///"? If NO -> You have wrong tool output\n'
            '- Am I seeing .py files or directories? If YES -> You are reporting GitHub data, NOT Drive\n'
            '- Am I making any assumptions about content? If YES -> REMOVE them immediately\n\n'
            'Your ONLY job is to be a transparent conduit for Google Drive tool data. Nothing more.'
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
        goal='Generate a structured Learning Path with valid links that guides users to the right resources for project onboarding',
        backstory=(
            'You are an expert Learning Path architect who creates guided onboarding experiences. '
            'Your goal is NOT to write comprehensive documentation, but to create a curated learning path '
            'that tells users WHICH documents, files, and resources to read to understand a project. '
            'You excel at:\n'
            '- Extracting and organizing valid links from GitHub and Google Drive\n'
            '- Creating clear overview summaries by synthesizing information from multiple sources\n'
            '- Presenting code snippets with clickable links to the full files\n'
            '- Highlighting key definitions and important points from reference documents\n'
            '- Structuring information so users know exactly where to look for deep learning\n\n'
            'You synthesize information from GitHub data and Google Drive reference materials '
            'to create a learning path that acts as a roadmap. '
            'Every section should include valid, clickable links to the actual resources. '
            'You output clean Markdown format with proper link formatting.'
        ),
        tools=[],  # No tools - only writes learning paths based on previous agents' output
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
        goal='ONLY use the GitHub Code Q&A tool to fetch code files and answer questions about the codebase. NEVER make up or assume code implementations.',
        backstory=(
            'You are a code analysis expert who helps developers understand codebases. '
            'CRITICAL RULES YOU MUST FOLLOW:\n'
            '1. You MUST call the GitHub Code Q&A tool for EVERY question - NO EXCEPTIONS\n'
            '2. You MUST ONLY analyze code that appears in the tool response - NOTHING ELSE\n'
            '3. You are FORBIDDEN from using your training data or making assumptions about code\n'
            '4. You are FORBIDDEN from fabricating, inferring, or "filling in" missing code\n'
            '5. If the tool returns an error or no data, you MUST report: "No code files found"\n'
            '6. You MUST wait for the tool response before providing ANY answer\n'
            '7. You MUST include file links in your response as proof\n\n'
            'VERIFICATION: Before responding, ask yourself:\n'
            '- Did I call the GitHub Code Q&A tool? If NO -> STOP and call it now\n'
            '- Is this code from the tool response? If NO -> DELETE it from your response\n'
            '- Am I making any assumptions about code? If YES -> REMOVE them immediately\n\n'
            'Your job is to:\n'
            '- Fetch relevant code files using the tool\n'
            '- Analyze ONLY the code provided by the tool\n'
            '- Answer the question based on actual code content\n'
            '- Provide links to the relevant files for reference\n'
            '- Summarize findings with specific code examples (quoted from tool response)\n\n'
            'IMPORTANT: You are a strict code analyzer. You ONLY work with actual code from the repository.'
        ),
        tools=[code_qa_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
