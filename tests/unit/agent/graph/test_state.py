"""Unit tests for the Virgo agent graph state module."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from virgo.core.agent.graph.state import AnswerState
from virgo.core.agent.schemas import Answer, MarkdownArticle, Revised


class DescribeAnswerState:
    """Tests for the AnswerState TypedDict."""

    def it_accepts_empty_messages_list(self):
        """Verify AnswerState accepts an empty messages list."""
        state: AnswerState = {
            "messages": [],
            "final_answer": None,
            "formatted_article": None,
        }
        assert state["messages"] == []

    def it_accepts_single_message(self):
        """Verify AnswerState accepts a single message."""
        message = HumanMessage(content="test question")
        state: AnswerState = {
            "messages": [message],
            "final_answer": None,
            "formatted_article": None,
        }
        assert len(state["messages"]) == 1
        assert state["messages"][0] == message

    def it_accepts_multiple_messages(self):
        """Verify AnswerState accepts multiple messages."""
        messages = [
            HumanMessage(content="question"),
            AIMessage(content="answer"),
            ToolMessage(content="result", tool_call_id="1"),
        ]
        state: AnswerState = {
            "messages": messages,
            "final_answer": None,
            "formatted_article": None,
        }
        assert len(state["messages"]) == 3
        assert state["messages"] == messages

    def it_accepts_human_messages(self):
        """Verify AnswerState accepts HumanMessage objects."""
        message = HumanMessage(content="human query")
        state: AnswerState = {
            "messages": [message],
            "final_answer": None,
            "formatted_article": None,
        }
        assert isinstance(state["messages"][0], HumanMessage)

    def it_accepts_ai_messages(self):
        """Verify AnswerState accepts AIMessage objects."""
        message = AIMessage(content="ai response")
        state: AnswerState = {
            "messages": [message],
            "final_answer": None,
            "formatted_article": None,
        }
        assert isinstance(state["messages"][0], AIMessage)

    def it_accepts_tool_messages(self):
        """Verify AnswerState accepts ToolMessage objects."""
        message = ToolMessage(content="tool result", tool_call_id="123")
        state: AnswerState = {
            "messages": [message],
            "final_answer": None,
            "formatted_article": None,
        }
        assert isinstance(state["messages"][0], ToolMessage)

    def it_accepts_answer_final_answer(self, answer: Answer):
        """Verify AnswerState accepts an Answer object as final_answer."""
        state: AnswerState = {
            "messages": [],
            "final_answer": answer,
            "formatted_article": None,
        }
        assert state["final_answer"] == answer
        assert isinstance(state["final_answer"], Answer)

    def it_accepts_revised_final_answer(self, revised: Revised):
        """Verify AnswerState accepts a Revised object as final_answer."""
        state: AnswerState = {
            "messages": [],
            "final_answer": revised,
            "formatted_article": None,
        }
        assert state["final_answer"] == revised
        assert isinstance(state["final_answer"], Revised)

    def it_accepts_none_final_answer(self):
        """Verify AnswerState accepts None for final_answer."""
        state: AnswerState = {
            "messages": [],
            "final_answer": None,
            "formatted_article": None,
        }
        assert state["final_answer"] is None

    def it_accepts_markdown_article(self, markdown_article: MarkdownArticle):
        """Verify AnswerState accepts a MarkdownArticle object."""
        state: AnswerState = {
            "messages": [],
            "final_answer": None,
            "formatted_article": markdown_article,
        }
        assert state["formatted_article"] == markdown_article
        assert isinstance(state["formatted_article"], MarkdownArticle)

    def it_accepts_none_formatted_article(self):
        """Verify AnswerState accepts None for formatted_article."""
        state: AnswerState = {
            "messages": [],
            "final_answer": None,
            "formatted_article": None,
        }
        assert state["formatted_article"] is None

    def it_accepts_complete_state(
        self, answer: Answer, markdown_article: MarkdownArticle
    ):
        """Verify AnswerState accepts all fields populated."""
        state: AnswerState = {
            "messages": [HumanMessage(content="question")],
            "final_answer": answer,
            "formatted_article": markdown_article,
        }
        assert len(state["messages"]) == 1
        assert state["final_answer"] == answer
        assert state["formatted_article"] == markdown_article
