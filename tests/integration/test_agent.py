"""Integration tests for the Virgo agent workflow.

These tests validate the full graph execution using a deterministic fake LLM
and a mock researcher. This ensures the agent follows the expected trajectory:
Draft -> Research -> Revise -> (Loop) -> Format.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from virgo.core.agent.graph.nodes import draft, research, revise
from virgo.core.agent.graph.nodes import format as format_node
from virgo.core.agent.schemas import MarkdownArticle


class DescribeVirgoAgent:
    """Tests for the full agent execution workflow."""

    @pytest.fixture
    def mock_researcher(self):
        """A mock researcher that returns deterministic results."""

        def _research(reflection, value, references=None):
            return [
                f"Source A for query: {reflection.search_queries[0]}",
                "Source B: Python is a dynamic language.",
            ]

        return _research

    def it_should_execute_happy_path_trajectory(
        self, agent_builder, fake_llm_responses, mock_researcher
    ) -> None:
        """Test the standard flow: Draft -> Research -> Revise -> Format.

        We simulate a scenario where the agent is allowed 2 iterations (Draft + 1 Revision).
        """
        # 1. Prepare Fake LLM Responses
        # The agent needs 3 responses from the LLM for this path:
        #   1. Draft Node: Returns an 'Answer' tool call.
        #   2. Revise Node: Returns a 'Revised' tool call.
        #   3. Format Node: Returns a 'MarkdownArticle' tool call.

        # Response 1: Initial Draft
        draft_content = "Python is a programming language."
        draft_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Answer",
                    "args": {
                        "value": draft_content,
                        "reflection": {
                            "missing": "history",
                            "superfluous": "none",
                            "search_queries": ["python history"],
                        },
                    },
                    "id": "call_draft_1",
                }
            ],
        )

        # Response 2: Revision (after research)
        revised_content = "Python was created by Guido van Rossum."
        revise_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Revised",
                    "args": {
                        "value": revised_content,
                        "reflection": {
                            "missing": "none",
                            "superfluous": "none",
                            "search_queries": ["python version"],
                        },
                        "references": ["[1] History of Python"],
                    },
                    "id": "call_revise_1",
                }
            ],
        )

        # Response 3: Formatting
        final_article = {
            "title": "The History of Python",
            "summary": "A brief look at Python's origins.",
            "content": "# The History of Python\n\nPython was created by...",
            "references": ["[1] History of Python"],
        }
        format_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "MarkdownArticle",
                    "args": final_article,
                    "id": "call_format_1",
                }
            ],
        )

        # Create the deterministic LLM
        llm = fake_llm_responses([draft_msg, revise_msg, format_msg])

        # 2. Build the Agent
        nodes = {
            "DRAFT": draft.create_node(llm),
            "RESEARCH": research.create_node(mock_researcher),
            "REVISE": revise.create_node(llm),
            "FORMAT": format_node.create_node(llm),
        }

        # 3. Execute with patched Max Iterations
        # Setting max_iterations to 2 ensures:
        # Iteration 1: Draft (count=1) -> Research
        # Iteration 2: Revise (count=2) -> Loop Check (>=2?) -> Format
        with patch("virgo.core.agent.graph.builder.VIRGO_MAX_ITERATIONS", 2):
            agent = agent_builder(nodes)
            result = agent.generate("Tell me about Python history.")

        # 4. Assertions
        assert result is not None
        assert isinstance(result, MarkdownArticle)
        assert result.title == "The History of Python"
        assert result.content.startswith("# The History of Python")

    def it_should_handle_loops_correctly(
        self, agent_builder, fake_llm_responses, mock_researcher
    ) -> None:
        """Test that the agent loops back to research if max iterations is high."""

        # Verify: Draft -> Research -> Revise -> Research -> Revise -> Format
        # This requires max_iterations >= 3 (Draft=1, Revise1=2, Revise2=3)

        # Messages: Draft, Revise 1, Revise 2, Format
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Answer",
                        "args": {
                            "value": "v1",
                            "reflection": {
                                "missing": "x",
                                "superfluous": "",
                                "search_queries": ["q1"],
                            },
                        },
                        "id": "1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Revised",
                        "args": {
                            "value": "v2",
                            "reflection": {
                                "missing": "y",
                                "superfluous": "",
                                "search_queries": ["q2"],
                            },
                            "references": [],
                        },
                        "id": "2",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Revised",
                        "args": {
                            "value": "v3",
                            "reflection": {
                                "missing": "none",
                                "superfluous": "",
                                "search_queries": [],
                            },
                            "references": [],
                        },
                        "id": "3",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "MarkdownArticle",
                        "args": {
                            "title": "Final",
                            "summary": "S",
                            "content": "C",
                            "references": [],
                        },
                        "id": "4",
                    }
                ],
            ),
        ]

        llm = fake_llm_responses(msgs)

        # Spy on researcher to verify it's called twice
        research_spy = MagicMock(side_effect=mock_researcher)

        nodes = {
            "DRAFT": draft.create_node(llm),
            "RESEARCH": research.create_node(research_spy),
            "REVISE": revise.create_node(llm),
            "FORMAT": format_node.create_node(llm),
        }

        # Set max_iterations to 3.
        # Flow:
        # 1. Draft (tool_invocations=1) -> Research
        # 2. Research (does not increment tool invocations on message history usually, but produces ToolMessage)
        # 3. Revise (tool_invocations=2) -> Loop Check (2 < 3) -> Research
        # 4. Research
        # 5. Revise (tool_invocations=3) -> Loop Check (3 >= 3) -> Format
        # 6. Format -> End
        with patch("virgo.core.agent.graph.builder.VIRGO_MAX_ITERATIONS", 3):
            agent = agent_builder(nodes)
            agent.generate("Loop test")

        assert research_spy.call_count == 2, (
            "Researcher should be called twice in this loop scenario"
        )

    def it_should_produce_valid_markdown_from_real_execution(
        self, agent_builder, fake_llm_responses, mock_researcher
    ):
        """Verify the integration of the Format node producing the final artifact."""

        # Simulating a direct flow where Draft is good enough (unlikely in real logic but possible in graph)
        # If we set MAX_ITERATIONS = 1, Draft -> Research -> Revise -> Format?
        # Actually logic is: if count >= Max.
        # Draft is 1 call.
        # If Max=1: Draft(1) -> Research -> Revise(2) -> Loop(2>=1) -> Format.
        # So even with Max=1, we get one revision cycle because check is AFTER Revise.

        draft_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Answer",
                    "args": {
                        "value": "Draft",
                        "reflection": {
                            "missing": "",
                            "superfluous": "",
                            "search_queries": ["q"],
                        },
                    },
                    "id": "1",
                }
            ],
        )
        revise_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Revised",
                    "args": {
                        "value": "Revised",
                        "reflection": {
                            "missing": "",
                            "superfluous": "",
                            "search_queries": [],
                        },
                        "references": [],
                    },
                    "id": "2",
                }
            ],
        )
        format_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "MarkdownArticle",
                    "args": {
                        "title": "Title",
                        "summary": "Sum",
                        "content": "**Bold** Content",
                        "references": [],
                    },
                    "id": "3",
                }
            ],
        )

        llm = fake_llm_responses([draft_msg, revise_msg, format_msg])

        nodes = {
            "DRAFT": draft.create_node(llm),
            "RESEARCH": research.create_node(mock_researcher),
            "REVISE": revise.create_node(llm),
            "FORMAT": format_node.create_node(llm),
        }

        with patch("virgo.core.agent.graph.builder.VIRGO_MAX_ITERATIONS", 1):
            agent = agent_builder(nodes)
            article = agent.generate("Quick check.")

        assert article.content == "**Bold** Content"
        assert article.title == "Title"
