"""The graph definitions for Virgo."""

import operator
import os
from enum import Enum
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, StateGraph

from virgo.chains import first_responder, revisor
from virgo.tools import execute_tools

VIRGO_MAX_ITERATIONS = int(os.getenv("VIRGO_MAX_ITERATIONS", 5))
"""The maximum number of iterations for the Virgo graph."""


class _VirgoNodes(Enum):
    """The nodes in the Virgo graph."""

    DRAFT = "draft"
    EXECUTE_TOOLS = "execute_tools"
    REVISE = "revise"


class Answer(TypedDict):
    """The answer graph that produces detailed answers to questions."""

    messages: Annotated[list[BaseMessage], operator.add]


def _first_responder_node(state: Answer) -> Answer:
    """The first responder node that generates detailed answers to questions.

    Args:
        state (Answer): The current state of the graph.

    Returns:
        Answer: The updated state of the graph with the first response.
    """
    return Answer(messages=[first_responder.invoke({"messages": state["messages"]})])


def _revisor_node(state: Answer) -> Answer:
    """The revisor node that revises previous answers based on reflections and new information.

    Args:
        state (Answer): The current state of the graph.
    Returns:
        Answer: The updated state of the graph with the revised answer.
    """
    return Answer(messages=[revisor.invoke({"messages": state["messages"]})])


# Define the state graph for Virgo
builder = StateGraph(state_schema=Answer)

# Add the nodes to the graph
builder.add_node(_VirgoNodes.DRAFT.value, _first_responder_node)
builder.add_node(_VirgoNodes.EXECUTE_TOOLS.value, execute_tools)
builder.add_node(_VirgoNodes.REVISE.value, _revisor_node)

# Define the edges between the nodes
builder.add_edge(_VirgoNodes.DRAFT.value, _VirgoNodes.EXECUTE_TOOLS.value)
builder.add_edge(_VirgoNodes.EXECUTE_TOOLS.value, _VirgoNodes.REVISE.value)

# Define the entry point of the graph
builder.set_entry_point(_VirgoNodes.DRAFT.value)


# Define the event loop for the graph
def _event_loop(state: Answer) -> str:
    """The event loop that runs the graph until the final answer is produced, or the maximum number of iterations is reached.

    Args:
        state (Answer): The current state of the graph.
    Returns:
        str: The final answer produced by the graph.
    """
    count_tool_invocations = sum(
        isinstance(msg, ToolMessage) for msg in state["messages"]
    )
    if count_tool_invocations >= VIRGO_MAX_ITERATIONS:
        return END
    return _VirgoNodes.EXECUTE_TOOLS.value


builder.add_conditional_edges(
    _VirgoNodes.REVISE.value,
    _event_loop,
)


__all__ = [
    "builder",
]
