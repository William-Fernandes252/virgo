"""The graph definitions for Virgo."""

import operator
import os
from enum import Enum
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langgraph.graph import END, StateGraph

from virgo.chains import first_responder, markdown_formatter, revisor
from virgo.schemas import MarkdownArticle
from virgo.tools import execute_tools

VIRGO_MAX_ITERATIONS = int(os.getenv("VIRGO_MAX_ITERATIONS", 5))
"""The maximum number of iterations for the Virgo graph."""


class _VirgoNodes(Enum):
    """The nodes in the Virgo graph."""

    DRAFT = "draft"
    EXECUTE_TOOLS = "execute_tools"
    REVISE = "revise"
    FORMAT = "format"


class Answer(TypedDict):
    """The answer graph that produces detailed answers to questions."""

    messages: Annotated[list[BaseMessage], operator.add]
    formatted_article: MarkdownArticle | None


def _first_responder_node(state: Answer) -> Answer:
    """The first responder node that generates detailed answers to questions.

    Args:
        state (Answer): The current state of the graph.

    Returns:
        Answer: The updated state of the graph with the first response.
    """
    return Answer(
        messages=[first_responder.invoke({"messages": state["messages"]})],
        formatted_article=None,
    )


def _revisor_node(state: Answer) -> Answer:
    """The revisor node that revises previous answers based on reflections and new information.

    Args:
        state (Answer): The current state of the graph.
    Returns:
        Answer: The updated state of the graph with the revised answer.
    """
    return Answer(
        messages=[revisor.invoke({"messages": state["messages"]})],
        formatted_article=None,
    )


def _format_node(state: Answer) -> Answer:
    """The format node that converts the final answer to a well-formatted Markdown article.

    Args:
        state (Answer): The current state of the graph.
    Returns:
        Answer: The updated state with the formatted Markdown article.
    """
    # Extract the last AI message with tool calls (the final revised answer)
    last_revised_message: AIMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_revised_message = msg
            break

    if not last_revised_message or not last_revised_message.tool_calls:
        return Answer(messages=[], formatted_article=None)

    # Extract the answer content and references from the tool call
    tool_call = last_revised_message.tool_calls[0]
    article_content = tool_call["args"].get("value", "")
    references = tool_call["args"].get("references", [])

    # Format using the markdown formatter chain
    formatter_chain = markdown_formatter | PydanticToolsParser(tools=[MarkdownArticle])
    result = formatter_chain.invoke(
        {
            "article": article_content,
            "references": "\n".join(references) if references else "None",
        }
    )

    formatted_article = result[0] if result else None
    return Answer(messages=[], formatted_article=formatted_article)


# Define the event loop for the graph
def _event_loop(state: Answer) -> str:
    """The event loop that runs the graph until the final answer is produced, or the maximum number of iterations is reached.

    Args:
        state (Answer): The current state of the graph.
    Returns:
        str: The next node to execute.
    """
    count_tool_invocations = sum(
        isinstance(msg, ToolMessage) for msg in state["messages"]
    )
    if count_tool_invocations >= VIRGO_MAX_ITERATIONS:
        return _VirgoNodes.FORMAT.value
    return _VirgoNodes.EXECUTE_TOOLS.value


def create_graph_builder() -> StateGraph:
    """Create the state graph builder for Virgo.

    Returns:
        StateGraph: The state graph builder for Virgo.
    """
    builder = StateGraph(state_schema=Answer)

    # Add the nodes to the graph
    builder.add_node(_VirgoNodes.DRAFT.value, _first_responder_node)
    builder.add_node(_VirgoNodes.EXECUTE_TOOLS.value, execute_tools)
    builder.add_node(_VirgoNodes.REVISE.value, _revisor_node)
    builder.add_node(_VirgoNodes.FORMAT.value, _format_node)

    # Define the edges between the nodes
    builder.add_edge(_VirgoNodes.DRAFT.value, _VirgoNodes.EXECUTE_TOOLS.value)
    builder.add_edge(_VirgoNodes.EXECUTE_TOOLS.value, _VirgoNodes.REVISE.value)

    # Define the entry point of the graph
    builder.set_entry_point(_VirgoNodes.DRAFT.value)

    builder.add_conditional_edges(
        _VirgoNodes.REVISE.value,
        _event_loop,
    )

    # Add edge from FORMAT to END
    builder.add_edge(_VirgoNodes.FORMAT.value, END)

    return builder


# Define the state graph for Virgo
builder = create_graph_builder()
"""The state graph builder for Virgo."""

__all__ = [
    "builder",
    "create_graph_builder",
]
