"""Unit tests for the draft node module."""

from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from virgo.core.agent.graph.nodes.draft import _create_node_from_chain, create_node
from virgo.core.agent.graph.state import AnswerState
from virgo.core.agent.schemas import Answer, Reflection


class DescribeCreateNodeFromChain:
    """Tests for the _create_node_from_chain internal function."""

    def it_returns_a_callable_node(self):
        """Verify _create_node_from_chain returns a callable."""
        mock_chain = MagicMock()
        node = _create_node_from_chain(mock_chain)

        assert callable(node)

    def it_invokes_chain_with_messages_and_formatted_article(self):
        """Verify the node invokes the chain with correct input."""
        # Create a mock chain that returns structured output
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": HumanMessage(content="raw response"),
            "parsed": Answer(
                value="Test answer",
                reflection=Reflection(
                    missing="nothing",
                    superfluous="nothing",
                    search_queries=[],
                ),
            ),
        }

        node = _create_node_from_chain(mock_chain)

        state: AnswerState = {
            "messages": [HumanMessage(content="What is AI?")],
            "final_answer": None,
            "formatted_article": None,
        }

        node(state)

        # Verify chain was invoked with correct parameters
        mock_chain.invoke.assert_called_once()
        call_args = mock_chain.invoke.call_args[0][0]
        assert "messages" in call_args
        assert "formatted_article" in call_args
        assert call_args["formatted_article"] is None

    def it_returns_updated_state_with_raw_message(self):
        """Verify the node returns state with raw message in messages list."""
        raw_message = HumanMessage(content="raw response")
        parsed_answer = Answer(
            value="Test answer",
            reflection=Reflection(
                missing="nothing",
                superfluous="nothing",
                search_queries=[],
            ),
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": raw_message,
            "parsed": parsed_answer,
        }

        node = _create_node_from_chain(mock_chain)

        state: AnswerState = {
            "messages": [HumanMessage(content="Question")],
            "final_answer": None,
            "formatted_article": None,
        }

        result = node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0] == raw_message

    def it_returns_updated_state_with_final_answer(self):
        """Verify the node returns state with parsed answer as final_answer."""
        parsed_answer = Answer(
            value="Test answer",
            reflection=Reflection(
                missing="details",
                superfluous="fluff",
                search_queries=["query 1"],
            ),
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": HumanMessage(content="raw"),
            "parsed": parsed_answer,
        }

        node = _create_node_from_chain(mock_chain)

        state: AnswerState = {
            "messages": [HumanMessage(content="Question")],
            "final_answer": None,
            "formatted_article": None,
        }

        result = node(state)

        assert result["final_answer"] == parsed_answer

    def it_returns_updated_state_with_none_formatted_article(self):
        """Verify the node returns state with None for formatted_article."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "raw": HumanMessage(content="raw"),
            "parsed": Answer(
                value="Answer",
                reflection=Reflection(
                    missing="",
                    superfluous="",
                    search_queries=[],
                ),
            ),
        }

        node = _create_node_from_chain(mock_chain)

        state: AnswerState = {
            "messages": [HumanMessage(content="Question")],
            "final_answer": None,
            "formatted_article": None,
        }

        result = node(state)

        assert result["formatted_article"] is None


class DescribeCreateNode:
    """Tests for the create_node function."""

    def it_returns_a_callable_state_node(self):
        """Verify create_node returns a callable state node."""
        mock_llm = MagicMock()

        # Mock the chain creation
        with MagicMock():
            node = create_node(mock_llm)

            assert callable(node)

    def it_accepts_a_chat_model(self):
        """Verify create_node accepts a BaseChatModel."""
        mock_llm = MagicMock()

        # Should not raise an error
        node = create_node(mock_llm)

        assert node is not None

    def it_returns_a_function_that_processes_state(self):
        """Verify the returned node is a function that processes AnswerState."""
        mock_llm = MagicMock()
        node = create_node(mock_llm)

        # The node should be callable and accept state
        assert callable(node)
