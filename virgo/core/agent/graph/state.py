import operator
from typing import Annotated, TypedDict

from virgo.core.agent.schemas import Answer, MarkdownArticle, Revised


class AnswerState(TypedDict):
    """The answer graph that produces detailed answers to questions."""

    messages: Annotated[list[Answer | Revised], operator.add]
    """History of answers and revisions."""

    final_answer: Answer | Revised | None
    """The final answer object, either an initial answer or a revised one."""

    formatted_article: MarkdownArticle | None
    """The formatted article produced by the formatter chain."""
