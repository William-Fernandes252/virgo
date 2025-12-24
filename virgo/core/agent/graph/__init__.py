"""Graph agent components for Virgo."""

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from virgo.core.agent.graph.builder import create_graph_builder
from virgo.core.agent.graph.nodes import draft, format, research, revise
from virgo.core.agent.graph.state import AnswerState

type VirgoGraph = CompiledStateGraph[AnswerState, None, AnswerState, AnswerState]


def create_graph(llm: BaseChatModel, researcher: research.Researcher) -> VirgoGraph:
    """Create the Virgo graph from a language model.

    Args:
        llm: The language model to be used by the agent.

    Returns:
        VirgoGraph: A configured instance of VirgoGraph.
    """
    builder = create_graph_builder(
        {
            "DRAFT": draft.create_node(llm),
            "RESEARCH": research.create_node(researcher),
            "REVISE": revise.create_node(llm),
            "FORMAT": format.create_node(llm),
        }
    )
    return builder.compile()
