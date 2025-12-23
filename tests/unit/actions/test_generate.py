from unittest.mock import create_autospec

import pytest

from tests.unit.factories import MarkdownArticleFactory
from virgo.core.actions.generate import GenerateArticleAction
from virgo.core.actions.protocols import ArticleGenerator
from virgo.core.agent.schemas import MarkdownArticle


class DescribeGenerateArticleAction:
    @pytest.fixture
    def mock_generator(self):
        return create_autospec(ArticleGenerator, instance=True)

    @pytest.fixture
    def action(self, mock_generator):
        """The Subject Under Test, pre-configured with dependencies."""
        return GenerateArticleAction(generator=mock_generator)

    def it_initializes_with_generator(self, action, mock_generator):
        # You verify the attribute, but you didn't have to instantiate it here
        assert action.generator is mock_generator

    def it_executes_generation_via_generator(self, action, mock_generator):
        # Use the factory from Step 1
        expected_article = MarkdownArticleFactory.build(title="AI Revolution")
        mock_generator.generate.return_value = expected_article

        result = action.execute("What is AI?")

        mock_generator.generate.assert_called_once_with("What is AI?")
        assert result == expected_article

    def it_returns_none_when_generator_fails(self, action, mock_generator):
        mock_generator.generate.return_value = None
        assert action.execute("Query") is None

    def it_supports_dependency_injection(self):
        """Verify action works with different generator implementations."""

        class CustomGenerator:
            def __init__(self, prefix: str):
                self.prefix = prefix

            def generate(self, question: str) -> MarkdownArticle | None:
                return MarkdownArticleFactory.build(
                    title=f"{self.prefix}: {question}",
                    summary="Generated summary.",
                    content="Generated content.",
                )

        generator = CustomGenerator(prefix="Article")
        action = GenerateArticleAction(generator=generator)
        result = action.execute("Test Question")

        assert result is not None
        assert result.title == "Article: Test Question"
