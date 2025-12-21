from langchain_core.runnables import RunnableSerializable
from langgraph.graph.state import StateNode

from virgo.core.agent.graph.state import AnswerState


def create_node(
    chain: RunnableSerializable,
) -> StateNode[AnswerState]:
    """Create the revisor node function.

    Args:
        chain: The revisor chain.

    Returns:
        callable: The revisor node function.
    """

    def revise(state: AnswerState) -> AnswerState:
        """The revisor node that revises previous answers based on reflections and new information.

        Args:
            state (AnswerState): The current state of the graph.
        Returns:
            AnswerState: The updated state of the graph with the revised answer.
        """
        output = chain.invoke({"messages": state["messages"]})

        return AnswerState(
            messages=[output["raw"]],
            final_answer=output["parsed"],
            formatted_article=None,
        )

    return revise
