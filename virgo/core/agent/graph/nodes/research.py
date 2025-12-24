"""Researcher node definition.

This module defines the researcher node in the Virgo agent's computational graph,
a tool calling node that is responsible for conducting research based on reflections and queries.
"""

from typing import Protocol

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from virgo.core.agent.schemas import Answer, Reflection, Revised


class Researcher(Protocol):
    """Callable that performs research based on reflections and queries."""

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
        ...


def create_node(researcher: Researcher) -> ToolNode:
    """Create the researcher node.

    Args:
        researcher: The researcher callable to run the search queries.

    Returns:
        StateNode[AnswerState]: The researcher state node.
    """
    return ToolNode(
        [
            StructuredTool.from_function(
                researcher,
                name=Answer.__name__,
                description=Answer.__doc__,
                args_schema=Answer,
            ),
            StructuredTool.from_function(
                researcher,
                name=Revised.__name__,
                description=Revised.__doc__,
                args_schema=Revised,
            ),
        ]
    )
