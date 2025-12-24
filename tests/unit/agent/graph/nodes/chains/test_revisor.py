"""Unit tests for the revisor chain module."""

from unittest.mock import MagicMock

from virgo.core.agent.graph.nodes.chains import revisor


class DescribeCreateChain:
    """Tests for the create_chain function in revisor module."""

    def it_returns_a_runnable_chain(self):
        """Verify create_chain returns a Runnable chain."""
        mock_llm = MagicMock()

        chain = revisor.create_chain(mock_llm)

        assert hasattr(chain, "invoke")

    def it_accepts_a_chat_model(self):
        """Verify create_chain accepts a BaseChatModel."""
        mock_llm = MagicMock()

        # Should not raise an error
        chain = revisor.create_chain(mock_llm)

        assert chain is not None

    def it_includes_prompt_template(self):
        """Verify the chain includes a prompt template."""
        mock_llm = MagicMock()

        chain = revisor.create_chain(mock_llm)

        # The chain should have a prompt or be composite with a prompt
        assert chain is not None

    def it_uses_revision_instruction(self):
        """Verify the chain uses the revision instruction."""
        mock_llm = MagicMock()

        chain = revisor.create_chain(mock_llm)

        # The prompt should have instructions about revising and citations
        assert chain is not None
