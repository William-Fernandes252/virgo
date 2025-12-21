"""Draft node for the Virgo agent graph."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable
from langgraph.graph.state import StateNode

from virgo.core.agent.graph.state import AnswerState


def _create_node_from_chain(
    chain: RunnableSerializable,
) -> StateNode[AnswerState]:
    """Create the draft node function.

    Args:
        chain: The first responder chain.

    Returns:
        callable: The first responder node function.
    """

    def draft(state: AnswerState) -> AnswerState:
        """The first responder node that generates detailed answers to questions.

        Args:
            state (AnswerState): The current state of the graph.

        Returns:
            AnswerState: The updated state of the graph with the first response.
        """
        output = chain.invoke(
            {"messages": state["messages"], "formatted_article": None}
        )
        return AnswerState(
            messages=[output["raw"]],
            final_answer=output["parsed"],
            formatted_article=None,
        )

    return draft


def create_node(llm: BaseChatModel) -> StateNode[AnswerState]:
    """Create the draft node.

    Args:
        llm: The language model to be used by the node.

    Returns:
        StateNode[AnswerState]: The draft state node.
    """
    from virgo.core.agent.graph.nodes.chains import first_responder

    chain = first_responder.create_chain(llm)
    return _create_node_from_chain(chain)
