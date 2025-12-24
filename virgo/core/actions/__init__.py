"""Actions module containing use cases and protocols for Virgo."""

from virgo.core.actions.generate import GenerateArticleAction
from virgo.core.actions.protocols import ArticleGenerator

__all__ = [
    "ArticleGenerator",
    "GenerateArticleAction",
]
