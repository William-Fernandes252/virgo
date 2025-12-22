"""Unit tests for the draft node module."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from virgo.core.agent.graph.nodes.chains import first_responder
from virgo.core.agent.graph.nodes.draft import create_node


class DescribeDraftNode:
    @pytest.fixture
    def mock_chain(self):
        return MagicMock()

    @pytest.fixture
    def draft_node(self, mock_chain):
        with pytest.MonkeyPatch.context() as m:
            m.setattr(first_responder, "create_chain", lambda _: mock_chain)
            yield create_node(llm=MagicMock())

    def it_updates_state_with_parsed_answer(
        self, draft_node, mock_chain, default_answer_state, make_answer
    ):
        """Verify the node takes chain output and updates final_answer."""

        parsed_answer = make_answer(value="The Answer")
        raw_msg = HumanMessage(content="raw")

        mock_chain.invoke.return_value = {
            "raw": raw_msg,
            "parsed": parsed_answer,
        }

        new_state = draft_node(default_answer_state)

        assert new_state["final_answer"] == parsed_answer
        assert new_state["messages"][-1] == raw_msg

    def it_resets_formatted_article(
        self, draft_node, mock_chain, default_answer_state, make_answer
    ):
        """Verify business rule: drafting invalidates previous formatting."""
        mock_chain.invoke.return_value = {
            "raw": HumanMessage(content="x"),
            "parsed": make_answer(),
        }

        default_answer_state["formatted_article"] = "Old Article"

        new_state = draft_node(default_answer_state)

        assert new_state["formatted_article"] is None
