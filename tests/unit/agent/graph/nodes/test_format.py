"""Unit tests for the format node module."""

from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from virgo.core.agent.graph.nodes.format import _create_node_from_chain, create_node
from virgo.core.agent.graph.state import AnswerState
from virgo.core.agent.schemas import Answer, MarkdownArticle, Reflection, Revised


class DescribeCreateNodeFromChain:
    """Tests for the _create_node_from_chain internal function."""

    def it_returns_a_callable_node(self):
        """Verify _create_node_from_chain returns a callable."""
        mock_chain = MagicMock()
        node = _create_node_from_chain(mock_chain)

        assert callable(node)

    def it_invokes_chain_with_article_and_references(self):
        """Verify the node invokes the chain with article and references."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "parsed": MarkdownArticle(
                title="Test",
                summary="Summary",
                content="Content",
            ),
        }

        node = _create_node_from_chain(mock_chain)

        answer = Answer(
            value="Test article content",
            reflection=Reflection(
                missing="",
                superfluous="",
                search_queries=[],
            ),
        )

        state: AnswerState = {
            "messages": [],
            "final_answer": answer,
            "formatted_article": None,
        }

        node(state)

        # Verify chain was invoked
        mock_chain.invoke.assert_called_once()
        call_args = mock_chain.invoke.call_args[0][0]
        assert "article" in call_args
        assert "references" in call_args

    def it_uses_final_answer_value_as_article(self):
        """Verify the node uses final_answer.value as the article."""
        article_content = "This is the article content"
        answer = Answer(
            value=article_content,
            reflection=Reflection(
                missing="",
                superfluous="",
                search_queries=[],
            ),
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "parsed": MarkdownArticle(
                title="Test",
                summary="Summary",
                content="Content",
            ),
        }

        node = _create_node_from_chain(mock_chain)

        state: AnswerState = {
            "messages": [],
            "final_answer": answer,
            "formatted_article": None,
        }

        node(state)

        call_args = mock_chain.invoke.call_args[0][0]
        assert call_args["article"] == article_content

    def it_includes_references_from_revised_answer(self):
        """Verify the node includes references from Revised answer."""
        references = ["[1] Source A", "[2] Source B"]
        revised = Revised(
            value="Answer with citations",
            reflection=Reflection(
                missing="",
                superfluous="",
                search_queries=[],
            ),
            references=references,
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "parsed": MarkdownArticle(
                title="Test",
                summary="Summary",
                content="Content",
            ),
        }

        node = _create_node_from_chain(mock_chain)

        state: AnswerState = {
            "messages": [],
            "final_answer": revised,
            "formatted_article": None,
        }

        node(state)

        call_args = mock_chain.invoke.call_args[0][0]
        # References should be joined with newlines
        assert "[1] Source A" in call_args["references"]
        assert "[2] Source B" in call_args["references"]

    def it_uses_none_for_references_when_not_available(self):
        """Verify the node uses 'None' string when references are not available."""
        answer = Answer(
            value="Article without references",
            reflection=Reflection(
                missing="",
                superfluous="",
                search_queries=[],
            ),
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "parsed": MarkdownArticle(
                title="Test",
                summary="Summary",
                content="Content",
            ),
        }

        node = _create_node_from_chain(mock_chain)

        state: AnswerState = {
            "messages": [],
            "final_answer": answer,
            "formatted_article": None,
        }

        node(state)

        call_args = mock_chain.invoke.call_args[0][0]
        assert call_args["references"] == "None"

    def it_returns_empty_messages(self):
        """Verify the node returns state with empty messages."""
        mock_chain = MagicMock()
        article = MarkdownArticle(
            title="Test",
            summary="Summary",
            content="Content",
        )
        mock_chain.invoke.return_value = {"parsed": article}

        node = _create_node_from_chain(mock_chain)

        answer = Answer(
            value="Content",
            reflection=Reflection(
                missing="",
                superfluous="",
                search_queries=[],
            ),
        )

        state: AnswerState = {
            "messages": [HumanMessage(content="question")],
            "final_answer": answer,
            "formatted_article": None,
        }

        result = node(state)

        assert result["messages"] == []

    def it_returns_formatted_article(self):
        """Verify the node returns the formatted article."""
        article = MarkdownArticle(
            title="Formatted Title",
            summary="Formatted summary",
            content="Formatted content",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"parsed": article}

        node = _create_node_from_chain(mock_chain)

        answer = Answer(
            value="Original content",
            reflection=Reflection(
                missing="",
                superfluous="",
                search_queries=[],
            ),
        )

        state: AnswerState = {
            "messages": [],
            "final_answer": answer,
            "formatted_article": None,
        }

        result = node(state)

        assert result["formatted_article"] == article

    def it_preserves_final_answer(self):
        """Verify the node preserves the final_answer in the returned state."""
        answer = Answer(
            value="Content",
            reflection=Reflection(
                missing="",
                superfluous="",
                search_queries=[],
            ),
        )

        article = MarkdownArticle(
            title="Test",
            summary="Summary",
            content="Content",
        )

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"parsed": article}

        node = _create_node_from_chain(mock_chain)

        state: AnswerState = {
            "messages": [],
            "final_answer": answer,
            "formatted_article": None,
        }

        result = node(state)

        assert result["final_answer"] == answer

    def it_handles_none_final_answer(self):
        """Verify the node handles None as final_answer."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "parsed": MarkdownArticle(
                title="Test",
                summary="Summary",
                content="Content",
            ),
        }

        node = _create_node_from_chain(mock_chain)

        state: AnswerState = {
            "messages": [],
            "final_answer": None,
            "formatted_article": None,
        }

        result = node(state)

        assert result["formatted_article"] is None
        assert result["final_answer"] is None
        assert result["messages"] == []


class DescribeCreateNode:
    """Tests for the create_node function."""

    def it_returns_a_callable_state_node(self):
        """Verify create_node returns a callable state node."""
        mock_llm = MagicMock()

        node = create_node(mock_llm)

        assert callable(node)

    def it_accepts_a_chat_model(self):
        """Verify create_node accepts a BaseChatModel."""
        mock_llm = MagicMock()

        # Should not raise an error
        node = create_node(mock_llm)

        assert node is not None
