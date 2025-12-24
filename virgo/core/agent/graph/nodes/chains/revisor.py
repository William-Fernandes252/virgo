from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from virgo.core.agent.schemas import Revised

_PROMPT = (
    ChatPromptTemplate.from_messages(
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
    )
    .partial(current_time=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    .partial(
        first_instruction="""
        Revise your previous answer using the new information.
            - You should use the previous critique to add important information to your answer;
            - You MUST include numerical citations in your revised answer, to ensure it can be verified;
            - Add a "References" section at the end of your answer (which does not count towards the word limit), listing the full citations for each of the numerical citations in your answer. For example:
                [1] Author Name, "Title of the Article", Source, Year. URL
                [2] "Title of the Article", Source, Year. URL
            - You should use the previous critique to remove any superfluous information from your answer, and make sure it does not contain more than ~250 words.
        """,
    )
)


def create_chain(llm: BaseChatModel):
    """Create a chain that revises previous answers based on reflections and new information.

    Args:
        llm (BaseChatModel): The language model to use. It must support tool usage.

    Returns:
        RunnableSerializable: The revisor chain.
    """
    return _PROMPT | llm.with_structured_output(Revised, include_raw=True)
