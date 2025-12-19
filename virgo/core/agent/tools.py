"""Tools for the Virgo assistant.

Currently includes a web search tool using Tavily.
"""

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from virgo.core.agent.schemas import Answer, Reflection, Revised

_tavily_tool = TavilySearch(max_results=5)
"""The Tavily search tool for retrieving relevant documents from the web."""


def _run_queries(
    reflection: Reflection, value: str, references: list[str] | None = None
) -> list[str]:
    """Run the search queries found in the reflection.

    Args:
        reflection (Reflection): The reflection object containing search queries.
        value (str): The answer text (unused but required by schema).
        references (list[str], optional): References for revised answers (unused).

    Returns:
        list[str]: The list of search results.
    """
    # Extract queries from the nested reflection object
    return _tavily_tool.batch([{"query": query} for query in reflection.search_queries])


execute_tools = ToolNode(
    [
        StructuredTool.from_function(
            _run_queries,
            name=Answer.__name__,
            description=Answer.__doc__,
            args_schema=Answer,
        ),
        StructuredTool.from_function(
            _run_queries,
            name=Revised.__name__,
            description=Revised.__doc__,
            args_schema=Revised,
        ),
    ]
)
"""The tool node for executing the search queries."""
