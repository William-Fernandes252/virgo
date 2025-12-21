"""Schemas for the Virgo assistant's answers and reflections."""

from typing import Annotated

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    """Reflection model representing the assistant's reflection on a given content."""

    missing: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Critique of what is missing from the content.",
                "examples": [
                    "The answer lacks information on recent advancements in renewable energy technologies."
                ],
            }
        ),
    ]
    superfluous: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Critique of what is superfluous in the content.",
                "examples": [
                    "The answer includes outdated statistics that are no longer relevant."
                ],
            }
        ),
    ]
    search_queries: Annotated[
        list[str],
        Field(
            default_factory=list,
            json_schema_extra={
                "description": "1-3 search queries to research improvements addressing the critique of the current answer.",
                "examples": [
                    [
                        "Latest advancements in renewable energy technologies 2024",
                        "Current statistics on renewable energy adoption worldwide",
                        "Innovative solutions in solar and wind energy",
                    ]
                ],
            },
        ),
    ]


class Answer(BaseModel):
    """Answer model representing the assistant's answer to a question."""

    value: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "A detailed answer to the question.",
                "examples": [
                    "Renewable energy technologies have seen significant advancements in recent years, particularly in solar and wind power. Innovations such as perovskite solar cells have improved efficiency rates, while offshore wind farms are becoming more prevalent due to their higher energy output. Additionally, energy storage solutions like advanced battery systems are addressing the intermittency issues associated with renewable sources. Governments worldwide are also implementing policies to encourage the adoption of clean energy, leading to increased investment and research in this sector. Overall, these developments are paving the way for a more sustainable and environmentally friendly energy future."
                ],
            }
        ),
    ]
    reflection: Annotated[
        Reflection,
        Field(
            json_schema_extra={
                "description": "Reflection on the content of the answer.",
            }
        ),
    ]


class Revised(Answer):
    """Revised answer model representing the assistant's revised answer to a question."""

    references: Annotated[
        list[str],
        Field(
            default_factory=list,
            json_schema_extra={
                "description": "List of full citations for each of the numerical citations in your revised answer.",
                "examples": [
                    [
                        "[1] Smith, J. (2024). Advances in Renewable Energy Technologies. Journal of Sustainable Energy.",
                        "[2] Doe, A. (2023). Global Renewable Energy Statistics. Energy Reports.",
                    ]
                ],
            },
        ),
    ]


class MarkdownArticle(BaseModel):
    """A well-formatted Markdown article."""

    title: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "Article title (without # prefix, will be added during rendering).",
            }
        ),
    ]
    summary: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "A brief 1-2 sentence summary of the article.",
            }
        ),
    ]
    content: Annotated[
        str,
        Field(
            json_schema_extra={
                "description": "The main article content formatted in Markdown with proper headings (##, ###), "
                "**bold** for key terms, bullet points, and numbered lists where appropriate.",
            }
        ),
    ]
    references: Annotated[
        list[str],
        Field(
            default_factory=list,
            json_schema_extra={
                "description": "List of references formatted as Markdown links: [1] [Title](URL) - Description",
                "examples": [
                    [
                        "[1] [Advances in Renewable Energy Technologies](https://example.com/renewable-energy) - An in-depth look at recent advancements in renewable energy.",
                        "[2] [Global Renewable Energy Statistics](https://example.com/energy-stats) - Comprehensive statistics on renewable energy adoption worldwide.",
                    ]
                ],
            },
        ),
    ]

    def to_markdown(self) -> str:
        """Convert the article to a complete Markdown string.

        Returns:
            str: The full Markdown-formatted article.
        """
        parts = [
            f"# {self.title}",
            "",
            f"*{self.summary}*",
            "",
            self.content,
        ]
        if self.references:
            parts.extend(["", "## References", ""])
            parts.extend(self.references)
        return "\n".join(parts)


__all__ = [
    "Answer",
    "MarkdownArticle",
    "Reflection",
    "Revised",
]
