"""Chain to generate detailed answers to questions."""

from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

from virgo.core.agent.schemas import Answer

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert researcher.

            Current time: {current_time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe, to maximize improvement.
            3. Recommend search queries to get information and improve your answer.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ],
).partial(current_time=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
"""The prompt template for the actor agent to generate answers and reflections."""


def create_chain(llm: BaseChatModel):
    """Create a chain that generates detailed answers to questions.

    Args:
        llm (BaseChatModel): The language model to use. It must support tool usage.

    Returns:
        RunnableSerializable: The first responder chain.
    """
    return _PROMPT.partial(
        first_instruction="Answer the question in detail, with ~250 words."
    ) | llm.with_structured_output(Answer, include_raw=True)
