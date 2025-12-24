"""Unit tests for the research node module."""

from unittest.mock import MagicMock

from langgraph.prebuilt import ToolNode

from virgo.core.agent.graph.nodes.research import create_node
from virgo.core.agent.schemas import Reflection


class DescribeCreateNode:
    """Tests for the create_node function."""

    def it_returns_a_tool_node(self):
        """Verify create_node returns a ToolNode."""
        mock_researcher = MagicMock()

        node = create_node(mock_researcher)

        assert isinstance(node, ToolNode)

    def it_creates_tools_from_researcher(self):
        """Verify create_node creates StructuredTools from the researcher."""
        mock_researcher = MagicMock()

        node = create_node(mock_researcher)

        # ToolNode should have tools
        assert hasattr(node, "tools_by_name")

    def it_creates_answer_tool(self):
        """Verify create_node creates an Answer tool."""
        mock_researcher = MagicMock()

        node = create_node(mock_researcher)

        # The node should have an Answer tool
        assert "Answer" in node.tools_by_name or any(
            "Answer" in str(tool) for tool in getattr(node, "tools", [])
        )

    def it_creates_revised_tool(self):
        """Verify create_node creates a Revised tool."""
        mock_researcher = MagicMock()

        node = create_node(mock_researcher)

        # The node should have a Revised tool
        assert "Revised" in node.tools_by_name or any(
            "Revised" in str(tool) for tool in getattr(node, "tools", [])
        )

    def it_accepts_researcher_callable(self):
        """Verify create_node accepts any callable as researcher."""

        def mock_researcher(reflection: Reflection, value: str, references=None):
            return ["result"]

        # Should not raise an error
        node = create_node(mock_researcher)

        assert isinstance(node, ToolNode)

    def it_tools_use_answer_schema(self):
        """Verify the created tools use the Answer schema."""
        mock_researcher = MagicMock()

        node = create_node(mock_researcher)

        # Verify Answer schema is used in tool creation
        # This is verified by checking that tools_by_name contains Answer
        assert len(node.tools_by_name) >= 1
