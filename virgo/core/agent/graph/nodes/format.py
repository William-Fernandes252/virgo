from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable
from langgraph.graph.state import StateNode

from virgo.core.agent.graph.state import AnswerState


def _create_node_from_chain(
    chain: RunnableSerializable[dict, dict],
) -> StateNode[AnswerState]:
    """Create the formatter node function.

    Args:
        chain: The markdown formatter chain.

    Returns:
        callable: The formatter node function.
    """

    def format(state: AnswerState) -> AnswerState:
        """The formatter node that converts the final answer to a well-formatted Markdown article.

        Args:
            state (AnswerState): The current state of the graph.

        Returns:
            AnswerState: The updated state with the formatted Markdown article.
        """
        latest_answer = state.get("final_answer")

        if not latest_answer:
            return AnswerState(messages=[], formatted_article=None, final_answer=None)

        article_content = latest_answer.value

        references = getattr(latest_answer, "references", [])

        output = chain.invoke(
            {
                "article": article_content,
                "references": "\n".join(references) if references else "None",
            }
        )

        formatted_article = output.get("parsed")

        return AnswerState(
            messages=[], formatted_article=formatted_article, final_answer=latest_answer
        )

    return format


def create_node(llm: BaseChatModel) -> StateNode[AnswerState]:
    """Create the formatter node.

    Args:
        llm: The language model to be used by the formatter chain.

    Returns:
        StateNode[AnswerState]: The formatter node.
    """
    from virgo.core.agent.graph.nodes.chains import markdown_formatter

    chain = markdown_formatter.create_chain(llm)
    return _create_node_from_chain(chain)
