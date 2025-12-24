import factory
from langchain_core.messages.human import HumanMessage

from virgo.core.agent.graph.state import AnswerState
from virgo.core.agent.schemas import Answer, MarkdownArticle, Reflection, Revised


class MarkdownArticleFactory(factory.Factory[MarkdownArticle]):
    """Factory for creating MarkdownArticle instances."""

    class Meta:
        model = MarkdownArticle

    title = factory.Faker("sentence", nb_words=6)
    summary = factory.Faker("paragraph", nb_sentences=2)
    content = factory.Faker("text", max_nb_chars=500)
    references = factory.List(
        [
            "[1] [Advances in Renewable Energy Technologies](https://example.com/renewable-energy) - An in-depth look at recent advancements in renewable energy.",
            "[2] [Global Renewable Energy Statistics](https://example.com/energy-stats) - Comprehensive statistics on renewable energy adoption worldwide.",
        ]
    )


class ReflectionFactory(factory.Factory[Reflection]):
    """Factory for creating Reflection instances."""

    class Meta:
        model = Reflection

    missing = "The answer lacks information on recent advancements in renewable energy technologies."
    superfluous = "The answer includes outdated statistics that are no longer relevant."
    search_queries = [
        "Latest advancements in renewable energy technologies 2024",
        "Current statistics on renewable energy adoption worldwide",
        "Innovative solutions in solar and wind energy",
    ]


class AnswerFactory(factory.Factory[Answer]):
    """Factory for creating Answer instances."""

    class Meta:
        model = Answer

    value = factory.Faker(
        "paragraph",
        nb_sentences=5,
    )
    reflection = factory.SubFactory(ReflectionFactory)


class RevisedFactory(factory.Factory[Revised]):
    """Factory for creating Revised instances."""

    class Meta:
        model = Revised

    value = factory.Faker(
        "paragraph",
        nb_sentences=5,
    )
    reflection = factory.SubFactory(ReflectionFactory)
    references = factory.List(
        [
            "[1] Smith, J. (2024). Advances in Renewable Energy Technologies. Journal of Sustainable Energy.",
            "[2] Doe, A. (2023). Global Renewable Energy Statistics. Energy Reports.",
        ]
    )


class AnswerStateFactory(factory.Factory[AnswerState]):
    """Factory for creating AnswerState dictionaries."""

    class Meta:
        model = dict

    messages = factory.List(
        [HumanMessage(content="What are the latest trends in renewable energy?")]
    )
    final_answer = factory.SubFactory(AnswerFactory)
    formatted_article = factory.SubFactory(MarkdownArticleFactory)
