"""Agent factory functions for creating CrewAI agents"""
from crewai import Agent
from typing import List, Optional

from src.tools import GitLabMCPTool, GoogleDriveMCPTool, RAGMilvusTool
from src.llm import GoToCustomLLM
from src.config.constants import AgentRole
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_gitlab_analyzer_agent(llm: GoToCustomLLM, gitlab_tool: GitLabMCPTool) -> Agent:
    """
    Create an agent specialized in fetching GitLab data using tools.

    Args:
        llm: LLM instance (should support tool calling)
        gitlab_tool: GitLab tool instance

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.GITLAB_ANALYZER} agent")

    return Agent(
        role=AgentRole.GITLAB_ANALYZER,
        goal='ONLY use the GitLab Project Analyzer tool to fetch data. NEVER make up, assume, or fabricate ANY information.',
        backstory=(
            'You are a strict data fetcher who ONLY reports information from the GitLab Project Analyzer tool. '
            'CRITICAL RULES YOU MUST FOLLOW:\n'
            '1. You MUST call the GitLab Project Analyzer tool for EVERY request - NO EXCEPTIONS\n'
            '2. You MUST ONLY report data that appears in the tool response - NOTHING ELSE\n'
            '3. You are FORBIDDEN from using your training data, memory, or making assumptions\n'
            '4. You are FORBIDDEN from fabricating, inferring, or "filling in" missing information\n'
            '5. If the tool returns an error or no data, you MUST report: "Tool returned no data for this field"\n'
            '6. You MUST wait for the tool response before providing ANY answer\n'
            '7. You MUST include the raw tool output in your response as proof\n\n'
            'VERIFICATION: Before responding, ask yourself:\n'
            '- Did I call the tool? If NO → STOP and call it now\n'
            '- Is this information from the tool response? If NO → DELETE it from your response\n'
            '- Am I making any assumptions? If YES → REMOVE them immediately\n\n'
            'Your ONLY job is to be a transparent conduit for tool data. Nothing more.'
        ),
        tools=[gitlab_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_drive_analyzer_agent(llm: GoToCustomLLM, drive_tool: GoogleDriveMCPTool) -> Agent:
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
        goal='Always use the Google Drive Document Analyzer tool to search for real-time documentation. Never make up or infer document content.',
        backstory=(
            'You are a research specialist who specializes in finding reference materials in Google Drive. '
            'Your PRIMARY and ONLY method of finding documents is through the Google Drive Document Analyzer tool. '
            'You NEVER make up document content, fabricate findings, or provide information from memory. '
            'You ALWAYS call the tool first before reporting any results. '
            'Your job is to ALWAYS use the Google Drive Document Analyzer tool to search for '
            'documentation, specifications, design docs, and related materials. '
            'If the tool returns no results, you MUST report "No documents found" - you NEVER invent documents. '
            'Tool usage is mandatory and non-negotiable for every search task.'
        ),
        tools=[drive_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_rag_analyzer_agent(llm: GoToCustomLLM, rag_tool: RAGMilvusTool) -> Agent:
    """
    Create an agent specialized in searching internal knowledge base for relevant information.

    Args:
        llm: LLM instance (should support tool calling)
        rag_tool: RAG Milvus tool instance

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.RAG_ANALYZER} agent")

    return Agent(
        role=AgentRole.RAG_ANALYZER,
        goal='Always use the Internal Knowledge Base Search tool to find real-time information. Never make up or infer knowledge base content.',
        backstory=(
            'You are an internal knowledge specialist who specializes in searching the company\'s internal knowledge base. '
            'Your PRIMARY and ONLY method of finding information is through the Internal Knowledge Base Search tool. '
            'You NEVER make up knowledge base content, fabricate findings, or provide information from memory. '
            'You ALWAYS call the tool first before reporting any results. '
            'Your job is to ALWAYS use the Internal Knowledge Base Search tool to find '
            'information about user segments, data engineering, experimentation platforms, and internal tools. '
            'If the tool returns no results, you MUST report "No relevant information found" - you NEVER invent information. '
            'Tool usage is mandatory and non-negotiable for every search task.'
        ),
        tools=[rag_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_documentation_writer_agent(llm: GoToCustomLLM) -> Agent:
    """
    Create an agent specialized in writing documentation based on gathered data.

    Args:
        llm: LLM instance (for content generation)

    Returns:
        Configured Agent instance
    """
    logger.info(f"Creating {AgentRole.DOCUMENTATION_WRITER} agent")

    return Agent(
        role=AgentRole.DOCUMENTATION_WRITER,
        goal='Generate comprehensive, well-structured documentation in JSON format based on gathered project data',
        backstory=(
            'You are an expert technical writer with deep knowledge of software architecture '
            'and documentation best practices. You take raw data about projects and transform it into '
            'clear, comprehensive documentation that helps developers understand projects quickly. '
            'You synthesize information from multiple sources including GitLab data, Google Drive '
            'reference materials, and internal knowledge base to create thorough documentation. '
            'You always output documentation in valid JSON format with proper structure.'
        ),
        tools=[],  # No tools - only writes documentation based on previous agents' output
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
