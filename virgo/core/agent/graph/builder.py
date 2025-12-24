"""Virgo agent graph definition module.

This module defines the structure and nodes of the Virgo agent's computational graph.
"""

import os
import sys
from typing import Final, TypedDict

from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import StateNode
from langgraph.prebuilt import ToolNode

from virgo.core.agent.graph.state import AnswerState

VIRGO_MAX_ITERATIONS: Final[int] = int(os.getenv("VIRGO_MAX_ITERATIONS", 5))

# Define interned node names for efficiency
DRAFT: Final[str] = sys.intern("draft")
RESEARCH: Final[str] = sys.intern("research")
REVISE: Final[str] = sys.intern("revise")
FORMAT: Final[str] = sys.intern("format")


type _VirgoGraphBuilder = StateGraph[AnswerState, None, AnswerState, AnswerState]
"""The type alias for the Virgo state graph.

In the current implementation, it starts with an list of messages (HumanMessage with the question),
and ends with an AnswerState containing the final formatted Markdown article.
"""


class VirgoNodes(TypedDict):
    """The nodes in the Virgo graph."""

    DRAFT: StateNode[AnswerState]
    RESEARCH: ToolNode
    REVISE: StateNode[AnswerState]
    FORMAT: StateNode[AnswerState]


def create_graph_builder(nodes: VirgoNodes) -> _VirgoGraphBuilder:
    """Create the state graph builder for Virgo.

    Args:
        nodes: The nodes to be added to the graph.

    Returns:
        VirgoStateGraph: The state graph builder for Virgo.
    """
    builder = StateGraph[AnswerState, None, AnswerState, AnswerState](
        state_schema=AnswerState
    )

    # Nodes
    builder.add_node(DRAFT, nodes["DRAFT"])
    builder.add_node(RESEARCH, nodes["RESEARCH"])
    builder.add_node(REVISE, nodes["REVISE"])
    builder.add_node(FORMAT, nodes["FORMAT"])

    # Edges
    builder.add_edge(DRAFT, RESEARCH)
    builder.add_edge(RESEARCH, REVISE)

    # Entry point
    builder.set_entry_point(DRAFT)

    builder.add_conditional_edges(
        REVISE,
        _event_loop,
    )

    # Add edge from FORMAT to END
    builder.add_edge(FORMAT, END)

    return builder


# Define the event loop for the graph
def _event_loop(state: AnswerState) -> str:
    """The event loop that runs the graph until the final answer is produced, or the maximum number of iterations is reached.

    Args:
        state (AnswerState): The current state of the graph.
    Returns:
        str: The next node to execute.
    """
    count_tool_invocations = sum(
        isinstance(msg, ToolMessage) for msg in state["messages"]
    )
    if count_tool_invocations >= VIRGO_MAX_ITERATIONS:
        return FORMAT
    return RESEARCH
