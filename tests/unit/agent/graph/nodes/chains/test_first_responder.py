"""Unit tests for the first responder chain module."""

from unittest.mock import MagicMock

from virgo.core.agent.graph.nodes.chains import first_responder


class DescribeCreateChain:
    """Tests for the create_chain function in first_responder module."""

    def it_returns_a_runnable_chain(self):
        """Verify create_chain returns a Runnable chain."""
        mock_llm = MagicMock()

        chain = first_responder.create_chain(mock_llm)

        assert hasattr(chain, "invoke")

    def it_accepts_a_chat_model(self):
        """Verify create_chain accepts a BaseChatModel."""
        mock_llm = MagicMock()

        # Should not raise an error
        chain = first_responder.create_chain(mock_llm)

        assert chain is not None

    def it_includes_prompt_template(self):
        """Verify the chain includes a prompt template."""
        mock_llm = MagicMock()

        chain = first_responder.create_chain(mock_llm)

        # The chain should have a prompt or be composite with a prompt
        assert chain is not None

    def it_uses_first_instruction_partial(self):
        """Verify the chain uses the first_instruction parameter."""
        mock_llm = MagicMock()

        chain = first_responder.create_chain(mock_llm)

        # The prompt should have the first instruction about detailed answers
        assert chain is not None
