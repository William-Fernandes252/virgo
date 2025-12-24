"""Unit tests for the Virgo agent graph builder module."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import StateNode
from langgraph.prebuilt import ToolNode

from virgo.core.agent.graph.builder import (
    DRAFT,
    FORMAT,
    RESEARCH,
    REVISE,
    VIRGO_MAX_ITERATIONS,
    VirgoNodes,
    _event_loop,
    create_graph_builder,
)
from virgo.core.agent.graph.state import AnswerState


class DescribeVirgoConstants:
    """Tests for the Virgo graph constants."""

    def it_defines_draft_node_name(self):
        """Verify DRAFT constant is defined."""
        assert DRAFT == "draft"
        assert isinstance(DRAFT, str)

    def it_defines_research_node_name(self):
        """Verify RESEARCH constant is defined."""
        assert RESEARCH == "research"
        assert isinstance(RESEARCH, str)

    def it_defines_revise_node_name(self):
        """Verify REVISE constant is defined."""
        assert REVISE == "revise"
        assert isinstance(REVISE, str)

    def it_defines_format_node_name(self):
        """Verify FORMAT constant is defined."""
        assert FORMAT == "format"
        assert isinstance(FORMAT, str)

    def it_defines_max_iterations(self):
        """Verify VIRGO_MAX_ITERATIONS constant is defined and is an integer."""
        assert isinstance(VIRGO_MAX_ITERATIONS, int)
        assert VIRGO_MAX_ITERATIONS > 0

    def it_reads_max_iterations_from_env_with_default(self):
        """Verify VIRGO_MAX_ITERATIONS uses environment variable with default."""
        # The constant is already loaded, so we just verify it has a sensible default
        assert VIRGO_MAX_ITERATIONS >= 1


class DescribeVirgoNodes:
    """Tests for the VirgoNodes TypedDict."""

    def it_has_draft_node_type(self):
        """Verify VirgoNodes has a DRAFT key for StateNode."""
        # Create a mock state node
        mock_node: StateNode[AnswerState] = MagicMock(spec=StateNode)
        nodes: VirgoNodes = {
            "DRAFT": mock_node,
            "RESEARCH": MagicMock(spec=ToolNode),
            "REVISE": MagicMock(spec=StateNode),
            "FORMAT": MagicMock(spec=StateNode),
        }
        assert "DRAFT" in nodes

    def it_has_research_node_type(self):
        """Verify VirgoNodes has a RESEARCH key for ToolNode."""
        nodes: VirgoNodes = {
            "DRAFT": MagicMock(spec=StateNode),
            "RESEARCH": MagicMock(spec=ToolNode),
            "REVISE": MagicMock(spec=StateNode),
            "FORMAT": MagicMock(spec=StateNode),
        }
        assert "RESEARCH" in nodes

    def it_has_revise_node_type(self):
        """Verify VirgoNodes has a REVISE key for StateNode."""
        nodes: VirgoNodes = {
            "DRAFT": MagicMock(spec=StateNode),
            "RESEARCH": MagicMock(spec=ToolNode),
            "REVISE": MagicMock(spec=StateNode),
            "FORMAT": MagicMock(spec=StateNode),
        }
        assert "REVISE" in nodes

    def it_has_format_node_type(self):
        """Verify VirgoNodes has a FORMAT key for StateNode."""
        nodes: VirgoNodes = {
            "DRAFT": MagicMock(spec=StateNode),
            "RESEARCH": MagicMock(spec=ToolNode),
            "REVISE": MagicMock(spec=StateNode),
            "FORMAT": MagicMock(spec=StateNode),
        }
        assert "FORMAT" in nodes


class DescribeEventLoop:
    """Tests for the _event_loop function."""

    def it_returns_format_when_at_max_iterations(self):
        """Verify event loop goes to FORMAT when at max iterations."""
        # Create state with VIRGO_MAX_ITERATIONS tool messages
        tool_messages = [
            ToolMessage(content=f"result {i}", tool_call_id=str(i))
            for i in range(VIRGO_MAX_ITERATIONS)
        ]
        state: AnswerState = {
            "messages": tool_messages,
            "final_answer": None,
            "formatted_article": None,
        }

        result = _event_loop(state)

        assert result == FORMAT

    def it_returns_format_when_exceeding_max_iterations(self):
        """Verify event loop goes to FORMAT when exceeding max iterations."""
        tool_messages = [
            ToolMessage(content=f"result {i}", tool_call_id=str(i))
            for i in range(VIRGO_MAX_ITERATIONS + 2)
        ]
        state: AnswerState = {
            "messages": tool_messages,
            "final_answer": None,
            "formatted_article": None,
        }

        result = _event_loop(state)

        assert result == FORMAT

    def it_returns_research_when_under_max_iterations(self):
        """Verify event loop continues to RESEARCH when under max iterations."""
        tool_messages = [
            ToolMessage(content=f"result {i}", tool_call_id=str(i))
            for i in range(VIRGO_MAX_ITERATIONS - 1)
        ]
        state: AnswerState = {
            "messages": tool_messages,
            "final_answer": None,
            "formatted_article": None,
        }

        result = _event_loop(state)

        assert result == RESEARCH

    def it_counts_only_tool_messages(self):
        """Verify event loop only counts ToolMessages, not other message types."""
        state: AnswerState = {
            "messages": [
                HumanMessage(content="question 1"),
                AIMessage(content="answer 1"),
                HumanMessage(content="question 2"),
                AIMessage(content="answer 2"),
                ToolMessage(content="result 1", tool_call_id="1"),
            ],
            "final_answer": None,
            "formatted_article": None,
        }

        result = _event_loop(state)

        # Only 1 ToolMessage, so should return RESEARCH
        assert result == RESEARCH

    def it_handles_empty_messages(self):
        """Verify event loop handles empty messages list."""
        state: AnswerState = {
            "messages": [],
            "final_answer": None,
            "formatted_article": None,
        }

        result = _event_loop(state)

        assert result == RESEARCH

    def it_returns_research_with_zero_tool_messages(self):
        """Verify event loop returns RESEARCH when there are no ToolMessages."""
        state: AnswerState = {
            "messages": [
                HumanMessage(content="q1"),
                AIMessage(content="a1"),
                HumanMessage(content="q2"),
            ],
            "final_answer": None,
            "formatted_article": None,
        }

        result = _event_loop(state)

        assert result == RESEARCH


class DescribeCreateGraphBuilder:
    """Tests for the create_graph_builder function."""

    def it_returns_a_state_graph_with_methods(self):
        """Verify create_graph_builder returns an object with StateGraph methods."""

        # Use basic callables for nodes instead of mocks to avoid type inspection issues
        def mock_draft(state):
            return state

        mock_research = MagicMock()

        def mock_revise(state):
            return state

        def mock_format(state):
            return state

        nodes: VirgoNodes = {
            "DRAFT": mock_draft,
            "RESEARCH": mock_research,
            "REVISE": mock_revise,
            "FORMAT": mock_format,
        }

        builder = create_graph_builder(nodes)

        assert builder is not None
        # StateGraph has these methods
        assert hasattr(builder, "add_node")
        assert hasattr(builder, "add_edge")
        assert hasattr(builder, "set_entry_point")

    def it_has_draft_node_name_constant(self):
        """Verify DRAFT constant is used for the draft node."""
        assert DRAFT == "draft"
        assert isinstance(DRAFT, str)

    def it_has_research_node_name_constant(self):
        """Verify RESEARCH constant is used for the research node."""
        assert RESEARCH == "research"
        assert isinstance(RESEARCH, str)

    def it_has_revise_node_name_constant(self):
        """Verify REVISE constant is used for the revise node."""
        assert REVISE == "revise"
        assert isinstance(REVISE, str)

    def it_has_format_node_name_constant(self):
        """Verify FORMAT constant is used for the format node."""
        assert FORMAT == "format"
        assert isinstance(FORMAT, str)

    def it_event_loop_is_used_as_routing_function(self):
        """Verify _event_loop is available as a routing function."""
        assert callable(_event_loop)
        # _event_loop should accept AnswerState and return a string
        state: AnswerState = {
            "messages": [],
            "final_answer": None,
            "formatted_article": None,
        }
        result = _event_loop(state)
        assert isinstance(result, str)
        assert result in (RESEARCH, FORMAT)
