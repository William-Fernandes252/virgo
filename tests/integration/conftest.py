"""Pytest configuration for integration tests."""

from collections.abc import Iterator
from typing import Any, Self

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from virgo.core.agent import VirgoAgent


@pytest.fixture
def virgo_agent_stub():
    class VirgoAgentStub(VirgoAgent):
        def __init__(self):
            self._graph = None

    return VirgoAgentStub()


class FakeChatModel(BaseChatModel):
    """A Fake Chat Model that supports tool binding and returns pre-canned responses.

    This replaces GenericFakeChatModel to ensure compatibility with 'with_structured_output'
    and consistent behavior across test environments.
    """

    _responses: Iterator[BaseMessage] = PrivateAttr()

    def __init__(self, responses: list[BaseMessage]):
        super().__init__()
        self._responses = iter(responses)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> ChatResult:
        """Generate the next pre-canned response."""
        try:
            response = next(self._responses)
        except StopIteration:
            raise ValueError(
                "FakeChatModel: Not enough responses provided for execution."
            )

        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Self:
        """Mock bind_tools to allow the agent to 'bind' tools.

        We return self because the responses are already pre-canned with the
        expected tool calls, so we don't need real tool binding logic.
        """
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"


@pytest.fixture
def sample_question() -> str:
    """Provide a sample question for testing the agent."""
    return "What are the main differences between Python and JavaScript?"


@pytest.fixture
def simple_question() -> str:
    """Provide a simple question for quick testing."""
    return "What is Python?"


@pytest.fixture
def fake_llm_responses():
    """Returns a deterministic LLM that we can script."""

    def _create_fake_llm(responses: list[BaseMessage]):
        return FakeChatModel(responses=responses)

    return _create_fake_llm


@pytest.fixture
def agent_builder():
    """A factory to build the VirgoAgent with specific dependencies."""

    from virgo.core.agent import VirgoAgent
    from virgo.core.agent.graph.builder import VirgoNodes, create_graph_builder

    def _build(nodes: VirgoNodes):
        graph_builder = create_graph_builder(nodes)
        graph = graph_builder.compile()
        return VirgoAgent(graph=graph)

    return _build
