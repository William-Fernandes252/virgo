"""Tools for the Virgo assistant.

Currently includes a web search tool using Tavily.
"""

from langchain_tavily import TavilySearch

from virgo.core.agent.schemas import Reflection


class TavilyResearcher:
    """Callable that performs research using the Tavily search tool."""

    def __init__(self, tool: TavilySearch) -> None:
        self._tool = tool

    def __call__(
        self,
        reflection: Reflection,
        value: str,
        references: list[str] | None = None,
    ) -> list[str]:
        """Run the search queries found in the reflection.

        Args:
            reflection (Reflection): The reflection object containing search queries.
            value (str): The answer text (unused but required by schema).
            references (list[str], optional): References for revised answers (unused).

        Returns:
            list[str]: The list of search results.
        """
        return self._tool.batch(
            [{"query": query} for query in reflection.search_queries]
        )
