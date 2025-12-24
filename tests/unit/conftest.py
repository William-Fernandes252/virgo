import pytest

from tests.unit.factories import (
    AnswerFactory,
    AnswerStateFactory,
    MarkdownArticleFactory,
    ReflectionFactory,
    RevisedFactory,
)
from virgo.core.agent.graph.state import AnswerState
from virgo.core.agent.schemas import Answer, MarkdownArticle, Reflection, Revised


@pytest.fixture
def answer() -> Answer:
    return AnswerFactory.build()


@pytest.fixture
def revised() -> Revised:
    return RevisedFactory.build()


@pytest.fixture
def reflection() -> Reflection:
    return ReflectionFactory.build()


@pytest.fixture
def markdown_article() -> MarkdownArticle:
    return MarkdownArticleFactory.build()


@pytest.fixture
def answer_state() -> AnswerState:
    return AnswerStateFactory.build()
