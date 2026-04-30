"""
LLM Factory - Centralized LLM instantiation for multi-provider support.

Supports both OpenAI and Google Gemini providers through a unified interface.
"""

from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel


def get_llm_instance(
    provider: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    verbose: bool = False,
    **kwargs: Any
) -> BaseChatModel:
    """
    Create an LLM instance based on the specified provider.
    
    Args:
        provider: LLM provider name ("openai" or "gemini")
        model: Model name (e.g., "gpt-4o-mini" or "gemini-2.0-flash-exp")
        api_key: API key for the provider
        temperature: LLM temperature (0.0 to 2.0)
        verbose: Enable verbose logging
        **kwargs: Additional provider-specific parameters
    
    Returns:
        BaseChatModel instance (ChatOpenAI or ChatGoogleGenerativeAI)
    
    Raises:
        ValueError: If provider is not supported
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=temperature,
            verbose=verbose,
            **kwargs
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            verbose=verbose,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: 'openai', 'gemini'"
        )
