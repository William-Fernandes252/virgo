"""Dependency injection container for Virgo CLI."""

from dependency_injector import containers, providers
from langchain_tavily import TavilySearch

from virgo.core.actions import GenerateArticleAction
from virgo.core.agent import VirgoAgent
from virgo.core.agent.graph import create_graph
from virgo.core.agent.llms import (
    LanguageModelProvider,
    OllamaLanguageModelProvider,
    OpenAILanguageModelProvider,
)
from virgo.core.agent.tools import TavilyResearcher
from virgo.core.settings import VirgoSettings


class Container(containers.DeclarativeContainer):
    """DI container for Virgo application.

    This container manages the dependencies for the Virgo CLI,
    providing configured instances of agents and actions.

    Example usage:
        ```python
        from virgo.cli import container

        # Override for testing
        with container.generate_action.override(mock_action):
            result = runner.invoke(app, ["generate", "question"])
        ```
    """

    wiring_config = containers.WiringConfiguration(
        modules=["virgo.cli.commands"],
    )

    config = providers.Configuration(strict=True, pydantic_settings=[VirgoSettings()])
    """The configuration provider for Virgo settings."""

    _language_model_provider = providers.Selector[LanguageModelProvider](
        config.genai_provider,
        openai=providers.Singleton(OpenAILanguageModelProvider),
        ollama=providers.Singleton(OllamaLanguageModelProvider),
    )

    _chat_model = providers.Callable(
        lambda provider, model_name: provider.get_chat_model(model_name),
        provider=_language_model_provider,
        model_name=config.model_name,
    )

    _tavily_tool = providers.Singleton(
        TavilySearch,
        max_results=5,
    )

    _researcher = providers.Callable(
        TavilyResearcher,
        tool=_tavily_tool,
    )

    _graph = providers.Singleton(
        create_graph,
        llm=_chat_model,
        researcher=_researcher,
    )

    _agent = providers.Singleton(
        VirgoAgent,
        graph=_graph,
    )
    """The Virgo agent singleton provider."""

    generate_action = providers.Factory(
        GenerateArticleAction,
        generator=_agent,
    )
    """The action provider for generating articles."""


__all__ = [
    "Container",
]
