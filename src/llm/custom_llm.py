"""Custom LLM implementation for GoToCompany's LiteLLM endpoint"""
import requests
from typing import Any, List, Optional, Union, Dict
from crewai.llm import BaseLLM

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GoToCustomLLM(BaseLLM):
    """
    Custom LLM implementation for GoToCompany's LiteLLM proxy endpoint.

    This bypasses LiteLLM's provider detection by making direct HTTP calls
    to the endpoint without authentication headers.
    """

    def __init__(
        self,
        model: str,
        endpoint: str,
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        timeout: int = 300,
        supports_tools: bool = False
    ):
        """
        Initialize the GoToCompany custom LLM.

        Args:
            model: The model name to use
            endpoint: The base URL of the LiteLLM proxy
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            supports_tools: Whether this model supports function/tool calling
        """
        super().__init__(model=model, temperature=temperature)

        self.endpoint = endpoint.rstrip('/')
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._supports_tools = supports_tools

        logger.info(
            f"Initialized GoToCustomLLM: model={model}, "
            f"endpoint={endpoint}, supports_tools={supports_tools}"
        )

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        callbacks: Optional[Any] = None,
        **kwargs
    ) -> str:
        """
        Make a call to the custom LLM endpoint.

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

        # Prepare the payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False
        }

        # Add optional parameters
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        # Make the request WITHOUT authorization header
        headers = {"Content-Type": "application/json"}

        try:
            logger.debug(f"Calling LLM endpoint: {self.endpoint}/chat/completions")
            response = requests.post(
                f"{self.endpoint}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            logger.debug(f"LLM response received: {len(content)} characters")
            return content

        except requests.exceptions.Timeout as e:
            error_msg = f"Request to {self.endpoint} timed out after {self.timeout}s"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling GoToCompany LLM endpoint: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except (KeyError, IndexError) as e:
            error_msg = f"Unexpected response format from endpoint: {str(e)}"
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
        return 8192


def create_tool_calling_llm(endpoint: str, model: str, temperature: float = 0.3) -> GoToCustomLLM:
    """
    Factory function to create an LLM instance configured for tool calling.

    Args:
        endpoint: LLM endpoint URL
        model: Model name
        temperature: Temperature (default: 0.3 for deterministic tool calling)

    Returns:
        Configured GoToCustomLLM instance
    """
    return GoToCustomLLM(
        model=model,
        endpoint=endpoint,
        temperature=temperature,
        supports_tools=True
    )


def create_writing_llm(endpoint: str, model: str, temperature: float = 0.6) -> GoToCustomLLM:
    """
    Factory function to create an LLM instance configured for content writing.

    Args:
        endpoint: LLM endpoint URL
        model: Model name
        temperature: Temperature (default: 0.6 for creative writing)

    Returns:
        Configured GoToCustomLLM instance
    """
    return GoToCustomLLM(
        model=model,
        endpoint=endpoint,
        temperature=temperature,
        supports_tools=False
    )
