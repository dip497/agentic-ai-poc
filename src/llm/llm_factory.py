"""
LLM Factory for supporting multiple LLM providers.

This module provides a factory for creating LLM instances that work with:
- OpenAI (ChatOpenAI)
- Google Gemini (ChatGoogleGenerativeAI)
- OpenRouter (ChatOpenAI with custom base_url)
- Anthropic Claude (ChatAnthropic)
- Local models via Ollama
"""

import os
from typing import Optional, Dict, Any, Union
import logging
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances from different providers."""
    
    @staticmethod
    def create_llm(
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the provider.
        
        Args:
            provider: LLM provider ("openai", "gemini", "openrouter", "anthropic", "ollama")
            model: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens
            **kwargs: Additional provider-specific arguments
            
        Returns:
            BaseChatModel instance
        """
        provider = provider.lower()
        
        if provider == "openai":
            return LLMFactory._create_openai_llm(model, temperature, max_tokens, **kwargs)
        elif provider == "gemini":
            return LLMFactory._create_gemini_llm(model, temperature, max_tokens, **kwargs)
        elif provider == "openrouter":
            return LLMFactory._create_openrouter_llm(model, temperature, max_tokens, **kwargs)
        elif provider == "anthropic":
            return LLMFactory._create_anthropic_llm(model, temperature, max_tokens, **kwargs)
        elif provider == "ollama":
            return LLMFactory._create_ollama_llm(model, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def _create_openai_llm(model: str, temperature: float, max_tokens: int, **kwargs) -> BaseChatModel:
        """Create OpenAI LLM instance."""
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                **{k: v for k, v in kwargs.items() if k != "api_key"}
            )
        except ImportError:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
    
    @staticmethod
    def _create_gemini_llm(model: str, temperature: float, max_tokens: int, **kwargs) -> BaseChatModel:
        """Create Google Gemini LLM instance."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
            
            # Map common model names to Gemini model names
            model_mapping = {
                "gemini-pro": "gemini-pro",
                "gemini-1.5-pro": "gemini-1.5-pro",
                "gemini-1.5-flash": "gemini-1.5-flash",
                "gemini-1.0-pro": "gemini-1.0-pro"
            }
            
            gemini_model = model_mapping.get(model, model)
            
            return ChatGoogleGenerativeAI(
                model=gemini_model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=api_key,
                **{k: v for k, v in kwargs.items() if k != "api_key"}
            )
        except ImportError:
            raise ImportError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
    
    @staticmethod
    def _create_openrouter_llm(model: str, temperature: float, max_tokens: int, **kwargs) -> BaseChatModel:
        """Create OpenRouter LLM instance."""
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
            
            # OpenRouter uses OpenAI-compatible API
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": kwargs.get("app_name", "moveworks-ai-system"),
                    "X-Title": kwargs.get("app_name", "Moveworks AI System")
                },
                **{k: v for k, v in kwargs.items() if k not in ["api_key", "app_name"]}
            )
        except ImportError:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
    
    @staticmethod
    def _create_anthropic_llm(model: str, temperature: float, max_tokens: int, **kwargs) -> BaseChatModel:
        """Create Anthropic Claude LLM instance."""
        try:
            from langchain_anthropic import ChatAnthropic
            
            api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_api_key=api_key,
                **{k: v for k, v in kwargs.items() if k != "api_key"}
            )
        except ImportError:
            raise ImportError("langchain-anthropic not installed. Run: pip install langchain-anthropic")
    
    @staticmethod
    def _create_ollama_llm(model: str, temperature: float, max_tokens: int, **kwargs) -> BaseChatModel:
        """Create Ollama LLM instance for local models."""
        try:
            from langchain_community.llms import Ollama
            from langchain_core.language_models import BaseChatModel
            
            base_url = kwargs.get("base_url", "http://localhost:11434")
            
            # Note: Ollama doesn't have a direct ChatModel, so we'll use a wrapper
            # In practice, you might want to use langchain_community.chat_models.ChatOllama
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=model,
                    temperature=temperature,
                    base_url=base_url,
                    **{k: v for k, v in kwargs.items() if k != "base_url"}
                )
            except ImportError:
                # Fallback to regular Ollama
                return Ollama(
                    model=model,
                    temperature=temperature,
                    base_url=base_url,
                    **{k: v for k, v in kwargs.items() if k != "base_url"}
                )
        except ImportError:
            raise ImportError("langchain-community not installed. Run: pip install langchain-community")
    
    @staticmethod
    def get_available_providers() -> Dict[str, Dict[str, Any]]:
        """Get information about available providers."""
        providers = {
            "openai": {
                "name": "OpenAI",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "env_var": "OPENAI_API_KEY",
                "install": "pip install langchain-openai"
            },
            "gemini": {
                "name": "Google Gemini",
                "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
                "env_var": "GOOGLE_API_KEY",
                "install": "pip install langchain-google-genai"
            },
            "openrouter": {
                "name": "OpenRouter",
                "models": [
                    "anthropic/claude-3.5-sonnet",
                    "google/gemini-pro-1.5",
                    "meta-llama/llama-3.1-8b-instruct",
                    "mistralai/mixtral-8x7b-instruct"
                ],
                "env_var": "OPENROUTER_API_KEY",
                "install": "pip install langchain-openai"
            },
            "anthropic": {
                "name": "Anthropic Claude",
                "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
                "env_var": "ANTHROPIC_API_KEY",
                "install": "pip install langchain-anthropic"
            },
            "ollama": {
                "name": "Ollama (Local)",
                "models": ["llama3.1", "mistral", "codellama", "phi3"],
                "env_var": "None (local)",
                "install": "pip install langchain-community"
            }
        }
        return providers
    
    @staticmethod
    def check_provider_availability(provider: str) -> Dict[str, Any]:
        """Check if a provider is available and configured."""
        providers = LLMFactory.get_available_providers()
        
        if provider not in providers:
            return {"available": False, "error": f"Unknown provider: {provider}"}
        
        provider_info = providers[provider]
        
        # Check if required package is installed
        try:
            if provider == "openai":
                import langchain_openai
            elif provider == "gemini":
                import langchain_google_genai
            elif provider == "anthropic":
                import langchain_anthropic
            elif provider == "ollama":
                import langchain_community
        except ImportError:
            return {
                "available": False,
                "error": f"Required package not installed. Run: {provider_info['install']}"
            }
        
        # Check API key (except for Ollama)
        if provider != "ollama":
            env_var = provider_info["env_var"]
            if provider == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY")
            elif provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")
            elif provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            else:
                api_key = os.getenv(env_var)
            
            if not api_key:
                return {
                    "available": False,
                    "error": f"API key not found. Set {env_var} environment variable."
                }
        
        return {"available": True, "provider_info": provider_info}


def create_llm_from_config(config: Dict[str, Any]) -> BaseChatModel:
    """
    Create LLM from configuration dictionary.
    
    Args:
        config: Configuration with provider, model, etc.
        
    Returns:
        BaseChatModel instance
    """
    provider = config.get("provider", "openai")
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 2000)
    
    # Extract provider-specific config
    provider_config = config.get("provider_config", {})
    
    return LLMFactory.create_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **provider_config
    )


# Example configurations for different providers
EXAMPLE_CONFIGS = {
    "openai": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 2000
    },
    "gemini": {
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 2000
    },
    "openrouter_claude": {
        "provider": "openrouter",
        "model": "anthropic/claude-3.5-sonnet",
        "temperature": 0.1,
        "max_tokens": 2000,
        "provider_config": {
            "app_name": "Moveworks AI System"
        }
    },
    "openrouter_llama": {
        "provider": "openrouter",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.1,
        "max_tokens": 2000
    },
    "ollama_local": {
        "provider": "ollama",
        "model": "llama3.1",
        "temperature": 0.1,
        "max_tokens": 2000,
        "provider_config": {
            "base_url": "http://localhost:11434"
        }
    }
}
