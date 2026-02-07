"""Custom LLM implementation using OpenAI API"""
from typing import Any, List, Optional, Union, Dict
from openai import OpenAI
from crewai.llm import BaseLLM

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAILLM(BaseLLM):
    """
    Custom LLM implementation using OpenAI API.

    Uses the OpenAI Python SDK to make API calls to GPT models.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        timeout: int = 300,
        supports_tools: bool = False
    ):
        """
        Initialize the OpenAI LLM.

        Args:
            model: The model name to use (e.g., gpt-4o, gpt-4o-mini)
            api_key: OpenAI API key
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            supports_tools: Whether this model supports function/tool calling
        """
        super().__init__(model=model, temperature=temperature)

        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._supports_tools = supports_tools

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key, timeout=timeout)

        logger.info(
            f"Initialized OpenAILLM: model={model}, supports_tools={supports_tools}"
        )

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        callbacks: Optional[Any] = None,
        **kwargs
    ) -> str:
        """
        Make a call to the OpenAI API.

        Args:
            messages: Either a string or list of message dicts with role/content
            callbacks: Optional callbacks (not used)
            **kwargs: Additional parameters

        Returns:
            The generated text response

        Raises:
            RuntimeError: If the API call fails
        """
        # Convert string to messages format if needed
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Prepare the parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add optional parameters
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]

        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]

        try:
            logger.debug(f"Calling OpenAI API with model: {self.model}")
            response = self.client.chat.completions.create(**params)

            content = response.choices[0].message.content

            logger.debug(f"OpenAI response received: {len(content)} characters")
            return content

        except Exception as e:
            error_msg = f"Error calling OpenAI API: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def supports_function_calling(self) -> bool:
        """Indicate whether this LLM supports function/tool calling."""
        return self._supports_tools

    def supports_stop_words(self) -> bool:
        """Indicate whether this LLM supports stop sequences."""
        return True

    def get_context_window_size(self) -> int:
        """Return the context window size for this model."""
        # GPT-4o and GPT-4o-mini have 128k context window
        if "gpt-4o" in self.model:
            return 128000
        return 8192


def create_tool_calling_llm(api_key: str, model: str, temperature: float = 0.3) -> OpenAILLM:
    """
    Factory function to create an LLM instance configured for tool calling.

    Args:
        api_key: OpenAI API key
        model: Model name
        temperature: Temperature (default: 0.3 for deterministic tool calling)

    Returns:
        Configured OpenAILLM instance
    """
    return OpenAILLM(
        model=model,
        api_key=api_key,
        temperature=temperature,
        supports_tools=True
    )


def create_writing_llm(api_key: str, model: str, temperature: float = 0.6) -> OpenAILLM:
    """
    Factory function to create an LLM instance configured for content writing.

    Args:
        api_key: OpenAI API key
        model: Model name
        temperature: Temperature (default: 0.6 for creative writing)

    Returns:
        Configured OpenAILLM instance
    """
    return OpenAILLM(
        model=model,
        api_key=api_key,
        temperature=temperature,
        supports_tools=False
    )
