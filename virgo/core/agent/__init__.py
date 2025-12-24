"""Agent module containing LangGraph implementation for Virgo."""

from langchain_core.messages import HumanMessage

from virgo.core.agent.graph import VirgoGraph
from virgo.core.agent.schemas import MarkdownArticle


class VirgoAgent:
    """The Virgo agent that wraps the LangGraph implementation."""

    _graph: VirgoGraph

    def __init__(self, graph: VirgoGraph) -> None:
        """Initialize the Virgo agent.

        Args:
            graph: The computational graph for the agent.

        """
        self._graph = graph

    def generate(self, question: str) -> MarkdownArticle | None:
        """Generate an article based on the input question.

        Args:
            question: The question to generate an article for.

        Returns:
            MarkdownArticle if generation succeeded, None otherwise.
        """
        message = HumanMessage(content=question)
        result = self._graph.invoke({"messages": [message]})  # type: ignore[arg-type]
        return result.get("formatted_article")


__all__ = [
    "VirgoAgent",
]
