"""Unit tests for the revise node module."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from tests.unit.factories import ReflectionFactory, RevisedFactory
from virgo.core.agent.graph.nodes.revise import create_node
from virgo.core.agent.graph.state import AnswerState
from virgo.core.agent.schemas import Revised


class DescribeCreateNode:
    """Tests for the create_node function."""

    def it_returns_a_callable_state_node(self):
        """Verify create_node returns a callable state node."""
        mock_chain = MagicMock()

        with patch(
            "virgo.core.agent.graph.nodes.revise.revisor.create_chain",
            return_value=mock_chain,
        ):
            node = create_node(MagicMock())

        assert callable(node)

    def it_invokes_chain_with_messages(self):
        """Verify the node invokes the chain with messages from state."""
        revised_answer = RevisedFactory.build(
            value="Revised answer with citations [1]",
            reflection=ReflectionFactory.build(search_queries=[]),
            references=["[1] Source, 2024"],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": HumanMessage(content="raw revised"),
            "parsed": revised_answer,
        }

        with patch(
            "virgo.core.agent.graph.nodes.revise.revisor.create_chain",
            return_value=mock_chain,
        ):
            node = create_node(MagicMock())

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
        revised_answer = RevisedFactory.build(
            value="Revised answer",
            reflection=ReflectionFactory.build(search_queries=[]),
            references=[],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": raw_message,
            "parsed": revised_answer,
        }

        state: AnswerState = {
            "messages": [HumanMessage(content="Question")],
            "final_answer": None,
            "formatted_article": None,
        }

        original_messages = state["messages"].copy()
        with patch(
            "virgo.core.agent.graph.nodes.revise.revisor.create_chain",
            return_value=mock_chain,
        ):
            node = create_node(MagicMock())
        result = node(state)

        expected_messages = [*original_messages, raw_message]
        assert result["messages"] == expected_messages

    def it_returns_updated_state_with_revised_answer(self):
        """Verify the node returns state with Revised answer as final_answer."""
        revised_answer = RevisedFactory.build(
            value="Revised content with citations",
            reflection=ReflectionFactory.build(search_queries=["query"]),
            references=["[1] Reference"],
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": HumanMessage(content="raw"),
            "parsed": revised_answer,
        }

        with patch(
            "virgo.core.agent.graph.nodes.revise.revisor.create_chain",
            return_value=mock_chain,
        ):
            node = create_node(MagicMock())

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
            "parsed": RevisedFactory.build(
                value="Answer",
                reflection=ReflectionFactory.build(search_queries=[]),
                references=[],
            ),
        }

        with patch(
            "virgo.core.agent.graph.nodes.revise.revisor.create_chain",
            return_value=mock_chain,
        ):
            node = create_node(MagicMock())

        state: AnswerState = {
            "messages": [HumanMessage(content="Question")],
            "final_answer": None,
            "formatted_article": None,
        }

        result = node(state)

        assert result["formatted_article"] is None

    def it_handles_revised_answer_with_references(self):
        """Verify the node handles Revised answers with references."""
        revised_answer = RevisedFactory.build(
            value="Answer with multiple citations [1][2]",
            reflection=ReflectionFactory.build(search_queries=[]),
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

        with patch(
            "virgo.core.agent.graph.nodes.revise.revisor.create_chain",
            return_value=mock_chain,
        ):
            node = create_node(MagicMock())

        state: AnswerState = {
            "messages": [HumanMessage(content="Question")],
            "final_answer": None,
            "formatted_article": None,
        }

        result = node(state)

        assert isinstance(result["final_answer"], Revised)
        assert len(result["final_answer"].references) == 2
