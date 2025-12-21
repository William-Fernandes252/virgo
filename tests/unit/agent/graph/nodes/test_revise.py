"""Unit tests for the revise node module."""

from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from virgo.core.agent.graph.nodes.revise import create_node
from virgo.core.agent.graph.state import AnswerState
from virgo.core.agent.schemas import Reflection, Revised


class DescribeCreateNode:
    """Tests for the create_node function."""

    def it_returns_a_callable_state_node(self):
        """Verify create_node returns a callable state node."""
        mock_chain = MagicMock()

        node = create_node(mock_chain)

        assert callable(node)

    def it_invokes_chain_with_messages(self):
        """Verify the node invokes the chain with messages from state."""
        revised_answer = Revised(
            value="Revised answer with citations [1]",
            reflection=Reflection(
                missing="more sources",
                superfluous="none",
                search_queries=[],
            ),
            references=["[1] Source, 2024"],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": HumanMessage(content="raw revised"),
            "parsed": revised_answer,
        }

        node = create_node(mock_chain)

        state: AnswerState = {
            "messages": [
                HumanMessage(content="Original question"),
                HumanMessage(content="Research results"),
            ],
            "final_answer": None,
            "formatted_article": None,
        }

        node(state)

        # Verify chain was invoked with messages
        mock_chain.invoke.assert_called_once()
        call_args = mock_chain.invoke.call_args[0][0]
        assert "messages" in call_args

    def it_returns_updated_state_with_raw_message(self):
        """Verify the node returns state with raw message."""
        raw_message = HumanMessage(content="raw revised")
        revised_answer = Revised(
            value="Revised answer",
            reflection=Reflection(
                missing="",
                superfluous="",
                search_queries=[],
            ),
            references=[],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": raw_message,
            "parsed": revised_answer,
        }

        node = create_node(mock_chain)

        state: AnswerState = {
            "messages": [HumanMessage(content="Question")],
            "final_answer": None,
            "formatted_article": None,
        }

        result = node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0] == raw_message

    def it_returns_updated_state_with_revised_answer(self):
        """Verify the node returns state with Revised answer as final_answer."""
        revised_answer = Revised(
            value="Revised content with citations",
            reflection=Reflection(
                missing="more details",
                superfluous="irrelevant info",
                search_queries=["query"],
            ),
            references=["[1] Reference"],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": HumanMessage(content="raw"),
            "parsed": revised_answer,
        }

        node = create_node(mock_chain)

        state: AnswerState = {
            "messages": [HumanMessage(content="Question")],
            "final_answer": None,
            "formatted_article": None,
        }

        result = node(state)

        assert result["final_answer"] == revised_answer
        assert isinstance(result["final_answer"], Revised)

    def it_returns_updated_state_with_none_formatted_article(self):
        """Verify the node returns state with None for formatted_article."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": HumanMessage(content="raw"),
            "parsed": Revised(
                value="Answer",
                reflection=Reflection(
                    missing="",
                    superfluous="",
                    search_queries=[],
                ),
                references=[],
            ),
        }

        node = create_node(mock_chain)

        state: AnswerState = {
            "messages": [HumanMessage(content="Question")],
            "final_answer": None,
            "formatted_article": None,
        }

        result = node(state)

        assert result["formatted_article"] is None

    def it_handles_revised_answer_with_references(self):
        """Verify the node handles Revised answers with references."""
        revised_answer = Revised(
            value="Answer with multiple citations [1][2]",
            reflection=Reflection(
                missing="",
                superfluous="",
                search_queries=[],
            ),
            references=[
                "[1] Source A",
                "[2] Source B",
            ],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": HumanMessage(content="raw"),
            "parsed": revised_answer,
        }

        node = create_node(mock_chain)

        state: AnswerState = {
            "messages": [HumanMessage(content="Question")],
            "final_answer": None,
            "formatted_article": None,
        }

        result = node(state)

        assert isinstance(result["final_answer"], Revised)
        assert len(result["final_answer"].references) == 2
