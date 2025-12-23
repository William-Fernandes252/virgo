"""Revisor node implementation for the Virgo agent graph."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable
from langgraph.graph.state import StateNode

from virgo.core.agent.graph.nodes.chains import revisor
from virgo.core.agent.graph.state import AnswerState


def _create_node_from_chain(
    chain: RunnableSerializable,
) -> StateNode[AnswerState]:
    """Return a node that invokes the provided revisor chain."""

    def revise(state: AnswerState) -> AnswerState:
        """Revise the previous answer based on the current reflection."""
        output = chain.invoke({"messages": state["messages"]})
        return AnswerState(
            messages=[*state["messages"], output["raw"]],
            final_answer=output["parsed"],
            formatted_article=None,
        )

    return revise


def create_node(llm: BaseChatModel) -> StateNode[AnswerState]:
    """Create the revisor node that wraps the revisor chain."""

    chain = revisor.create_chain(llm)
    return _create_node_from_chain(chain)
