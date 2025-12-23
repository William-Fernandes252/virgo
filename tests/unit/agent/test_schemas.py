"""Unit tests for the Virgo agent schema models."""

import pytest
from pydantic import ValidationError

from tests.unit.factories import (
    AnswerFactory,
    MarkdownArticleFactory,
    ReflectionFactory,
    RevisedFactory,
)
from virgo.core.agent.schemas import Answer, Reflection, Revised


class DescribeReflection:
    """Tests for the Reflection schema."""

    def it_creates_with_required_fields(self):
        reflection = ReflectionFactory.build(
            missing="Missing details",
            superfluous="Extra background",
        )

        assert reflection.missing == "Missing details"
        assert reflection.superfluous == "Extra background"

    def it_requires_missing_field(self):
        with pytest.raises(ValidationError):
            Reflection(superfluous="Only superfluous")  # type: ignore[call-arg]

    def it_requires_superfluous_field(self):
        with pytest.raises(ValidationError):
            Reflection(missing="Only missing")  # type: ignore[call-arg]


class DescribeAnswer:
    """Tests for the Answer schema."""

    def it_creates_with_required_fields(self):
        answer = AnswerFactory.build()

        assert isinstance(answer.value, str)
        assert isinstance(answer.reflection, Reflection)
        assert isinstance(answer.reflection.search_queries, list)

    def it_defaults_search_queries_to_empty_list(self):
        answer = AnswerFactory.build(
            reflection=ReflectionFactory.build(search_queries=[]),
        )

        assert answer.reflection.search_queries == []
        assert isinstance(answer.reflection.search_queries, list)

    def it_accepts_search_queries(self):
        queries = ["foo", "bar"]
        answer = AnswerFactory.build(
            reflection=ReflectionFactory.build(search_queries=queries),
        )

        assert answer.reflection.search_queries == queries
        assert len(answer.reflection.search_queries) == 2

    def it_requires_value_field(self):
        with pytest.raises(ValidationError):
            Answer(
                reflection=Reflection(missing="M", superfluous="S"),
            )  # type: ignore[call-arg]

    def it_requires_reflection_field(self):
        with pytest.raises(ValidationError):
            Answer(value="Just text")  # type: ignore[call-arg]


class DescribeRevised:
    """Tests for the Revised schema."""

    def it_extends_answer(self):
        assert issubclass(Revised, Answer)

    def it_creates_with_all_fields(self):
        revised = RevisedFactory.build(
            reflection=ReflectionFactory.build(search_queries=["extra"]),
            references=["[1] Source"],
        )

        assert revised.references == ["[1] Source"]
        assert revised.reflection.search_queries == ["extra"]

    def it_defaults_references_to_empty_list(self):
        revised = RevisedFactory.build(references=[])

        assert revised.references == []
        assert isinstance(revised.references, list)

    def it_inherits_search_queries_default(self):
        revised = RevisedFactory.build()

        assert isinstance(revised.reflection.search_queries, list)


class DescribeMarkdownArticle:
    """Tests for the MarkdownArticle schema."""

    def it_creates_with_required_fields(self):
        article = MarkdownArticleFactory.build(
            title="Test",
            summary="Summary",
            content="## Section\n\nBody",
            references=[],
        )

        assert article.title == "Test"
        assert article.content == "## Section\n\nBody"
        assert article.references == []

    def it_defaults_references_to_empty_list(self):
        article = MarkdownArticleFactory.build(
            title="Test",
            summary="Summary",
            content="Content",
            references=[],
        )

        assert article.references == []
        assert isinstance(article.references, list)

    def it_accepts_references(self):
        refs = ["[1] Source", "[2] Other"]
        article = MarkdownArticleFactory.build(
            title="Test",
            summary="Summary",
            content="Content",
            references=refs,
        )

        assert article.references == refs
        assert len(article.references) == 2


class DescribeMarkdownArticleToMarkdown:
    """Tests for MarkdownArticle.to_markdown."""

    def it_formats_article(self):
        article = MarkdownArticleFactory.build(
            title="My Article",
            summary="Nice summary",
            content="## Section\n\nContent",
            references=["[1] Source"],
        )

        result = article.to_markdown()

        assert result.startswith("# My Article")
        assert "*Nice summary*" in result
        assert "## Section\n\nContent" in result
        assert "## References" in result
        assert "[1] Source" in result

    def it_excludes_references_section_when_empty(self):
        article = MarkdownArticleFactory.build(
            title="No refs",
            summary="Summary",
            content="Content",
            references=[],
        )

        result = article.to_markdown()

        assert "## References" not in result

    def it_returns_string(self):
        article = MarkdownArticleFactory.build(
            title="Test",
            summary="Summary",
            content="Content",
            references=[],
        )

        assert isinstance(article.to_markdown(), str)
