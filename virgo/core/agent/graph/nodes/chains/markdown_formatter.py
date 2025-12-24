"""Chain to format articles into Markdown."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate

from virgo.core.agent.schemas import MarkdownArticle

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert technical writer and Markdown formatter.

            Your task is to take an article and format it as a well-structured Markdown document.

            Formatting guidelines:
            - Create a clear, descriptive title
            - Write a brief 1-2 sentence summary
            - Use ## for main section headings and ### for subsections
            - Use **bold** for key terms and important concepts
            - Use bullet points (-) for unordered lists
            - Use numbered lists (1. 2. 3.) for sequential steps or rankings
            - Use > for important quotes or callouts
            - Format references as numbered Markdown links: [1] [Title](URL) - Brief description
            - Ensure proper spacing between sections
            - Keep the content faithful to the original - do not add or remove information
            """,
        ),
        (
            "human",
            """Please format the following article as Markdown:

            {article}

            References (if any):
            {references}""",
        ),
    ]
)
"""The prompt template for the Markdown formatter."""


def create_chain(llm: BaseChatModel):
    """Create a chain that formats articles into Markdown.

    Args:
        llm (BaseChatModel): The language model to use. It must support tool usage.

    Returns:
        RunnableSerializable: The markdown formatter chain.
    """
    return _PROMPT | llm.with_structured_output(MarkdownArticle, include_raw=True)
